import time
import gym
# import gymnasium as gym
import d4rl
import mujoco_py
from matplotlib import pyplot as plt
from procgen import ProcgenEnv, ProcgenGym3Env

from AE.skill_network import *
from AE.configs import *
import numpy


def get_model(env: object, env_name: str) -> object:
    if env_name is 'procgen':
        print(env.observation_space)
        model = CnnLMP(latent_dim=config['latent_dim'], state_dim=env.observation_space.shape,
                       action_dim=15, hidden_dims=config['hidden_dims'],
                       tanh=config['tanh'], latent_reg=config['latent_reg'], ar=False)
    else:
        model = LMP(latent_dim=config['latent_dim'], state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0], hidden_dims=config['hidden_dims'],
                    goal_idxs=config['goal_idxs'], tanh=config['tanh'],
                    latent_reg=config['latent_reg'], ar=False)
    return model


def load_ae_model(env, path, env_name=None):
    model = get_model(env, env_name=env_name)
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['gp_aa_model'])
    return model


def get_trained_ae_model(action_dim, state_dim, path, env_name=None):
    if env_name is 'procgen':
        ae_model = CnnLMP(latent_dim=config['latent_dim'], state_dim=state_dim, action_dim=15,
                          hidden_dims=config['hidden_dims'], tanh=config['tanh'], latent_reg=config['latent_reg'],
                          ar=False)
    else:
        ae_model = LMP(latent_dim=config['latent_dim'], state_dim=state_dim, action_dim=action_dim,
                       hidden_dims=config['hidden_dims'], goal_idxs=config['goal_idxs'], tanh=config['tanh'],
                       latent_reg=config['latent_reg'], ar=False)
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    ae_model.load_state_dict(checkpoint['gp_aa_model'])
    return ae_model


def make_env(costume_map=False):
    if config['env_name'] is 'procgen':
        env = ProcgenEnv(num_envs=1, env_name='coinrun', num_levels=64, start_level=0, distribution_mode="easy")
        return env
    if costume_map:
        R = 'r'
        G = 'g'
        example_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 0, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
                       [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, G, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 0, G, 0, 0, 0, 1, G, 0, 0, 0, 0, 0, 0, 0, 0, 1, G, 0, 0, 1],
                       [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                       [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, G, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                       [1, 1, 0, 1, G, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, G, 1, 0, 1, 1, 1],
                       [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 1, 0, 0, G, 0, 0, 0, 0, G, 0, 1, 0, 0, 0, 0, G, 1],
                       [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 0, 0, G, 0, 0, 1, 0, 0, 0, 0, 0, 0, G, 0, 0, 1, 0, 0, 0, 1],
                       [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, G, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                       [1, 0, 0, 1, 0, 1, 0, G, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                       [1, 1, G, 1, 0, 1, 0, 1, 0, 1, 1, 1, G, 1, 0, 1, 0, 1, G, 1, 1, 1],
                       [1, 0, 0, 1, 0, 0, G, 1, 0, 0, 0, 0, 0, 1, 0, G, 0, 1, 0, 0, G, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        env = gym.make('antmaze-large-diverse-v0', maze_map=example_map, reward_type='dense', render_mode='human')
    else:
        env = gym.make(config['env_name'], reward_type='dense')
    return env


def play_policy(env, model, num_eval, traj_length, tanh, render=False, encode_state=False):
    model.eval()
    rewards = []
    for i in range(num_eval):
        print(i)
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        reward = 0
        num_steps = 0
        while not done:
            latent, _ = model.prior.act(latent=None, state=state, encode_state=True)
            if tanh:
                latent = torch.tanh(latent)
            for t in range(traj_length):
                action, _ = model.decoder.act(latent, state, encode_state=True)
                action = action.cpu().numpy().flatten()
                env.render()
                s, r, done, info = env.step(action[0])
                reward += r
                num_steps += 1
                state = torch.FloatTensor(s).unsqueeze(0)
                # done = done and 'TimeLimit.truncated' not in info
                # if done:
                #     print(reward, info)
                #     rewards.append(reward)
                #     break
        print(reward)
        rewards.append(reward)
    print(numpy.mean(rewards))
    return rewards


def evaluate(env, model):
    rewards = play_policy(env, model, config['num_eval'], config['traj_length'], config['tanh'], render=True)
    plt.plot(rewards, label='AE')
    plt.xlabel("Steps")
    plt.ylabel("reward")
    plt.legend(loc='lower right', frameon=True)
    plt.title(config['env_name'])
    plt.show()
