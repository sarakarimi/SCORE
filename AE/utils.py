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
                       tanh=config['tanh'], latent_reg=config['latent_reg'], ar=False, spirl=config['spirl'])
    else:
        model = LMP(latent_dim=config['latent_dim'], state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0], hidden_dims=config['hidden_dims'],
                    goal_idxs=config['goal_idxs'], tanh=config['tanh'],
                    latent_reg=config['latent_reg'], ar=False, spirl=config['spirl'])
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
                          ar=False, spirl=config['spirl'])
    else:
        ae_model = LMP(latent_dim=config['latent_dim'], state_dim=state_dim, action_dim=action_dim,
                       hidden_dims=config['hidden_dims'], goal_idxs=config['goal_idxs'], tanh=config['tanh'],
                       latent_reg=config['latent_reg'], ar=False, spirl=config['spirl'])
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    ae_model.load_state_dict(checkpoint['gp_aa_model'])
    return ae_model


def make_env(costume_map=False):
    if config['env_name'] is 'procgen':
        env = ProcgenEnv(num_envs=1, env_name='coinrun', num_levels=1, start_level=65, distribution_mode="easy")
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
    if config['spirl']:
        input_format = "spirl"
    else:
        input_format = 'opal'
    plt.title(config['env_name'] + " in format of " + input_format)
    plt.show()


if __name__ == '__main__':
    # env = make_env(costume_map=True)
    # model = load_ae_model(env,
    #                       "/home/sara/repositories/RL-skill-extraction/models/AE_models/antmaze-xl-diverse-v0-opal/600.pt")
    # evaluate(env, model)
    # import gymnasium as gym
    import gym3

    # example_map = [
    #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #             [1, 'r', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #             [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #             [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #             [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    #             [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    #             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    #             [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    #             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #             [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #             [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #             [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    #             [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    #             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 'g', 1, 1, 1],
    #             [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    #
    # env = gym.make('antmaze-large-diverse-v0', maze_map=example_map, reward_type='dense', render_mode='human')
    # # env = gym.make('antmaze-medium-diverse-v0',render_mode='human')
    #
    # env.reset()
    # env.set_xy((69, 48))
    # while True:
    #     env.render()
    #     s, r, d, i = env.step(env.action_space.sample())
    #     print(r, d, i)
    # env = ProcgenEnv(num_envs=1, env_name='coinrun', num_levels=1, start_level=65, distribution_mode="easy", render_mode='human')
    import gym
    import gym.wrappers

    env = gym.make("procgen-coinrun-v0", render_mode="human", distribution_mode='hard')

    model = load_ae_model(env, "/home/sara/repositories/RL-skill-extraction/models/AE_models/procgen-opal/40.pt",
                          env_name='procgen')
    rewards = play_policy(env, model, config['num_eval'], config['traj_length'], config['tanh'], encode_state=True)
    plt.plot(rewards, label='AE')
    plt.xlabel("Steps")
    plt.ylabel("reward")
    plt.legend(loc='lower right', frameon=True)
    if config['spirl']:
        input_format = "spirl"
    else:
        input_format = 'opal'
    plt.title(config['env_name'] + " in format of " + input_format)
    plt.show()
