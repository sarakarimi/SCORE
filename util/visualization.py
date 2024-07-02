import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
from matplotlib import pyplot as plt
import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from AE.utils import load_ae_model


class SkillConcatenated(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        state_dim = env.observation_space.shape[0] + 8
        self.observation_space = Box(shape=(state_dim,), low=-np.inf, high=np.inf)
        print(env.spec.name + "-v" + str(env.spec.version))
        # replace with path to trained VAE
        path = "/models/AE_models/" + env.spec.name + "-v" + str(env.spec.version) + "-opal/long-with-clamp-1000.pt"
        AE_model = load_ae_model(env, path).to('cuda')
        self.state_encoder = AE_model.prior

    def observation(self, obs):
        tensor_obs = torch.Tensor(np.expand_dims(obs, axis=0)).to('cuda')
        p_encoded_state = self.state_encoder.act(None, tensor_obs, deterministic=True)
        concat_obs = torch.cat([p_encoded_state, tensor_obs], dim=-1)
        return concat_obs[0].detach().cpu().numpy()


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 200)),
            nn.Tanh(),
            layer_init(nn.Linear(200, 200)),
            nn.Tanh(),
            layer_init(nn.Linear(200, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 200)),
            nn.Tanh(),
            layer_init(nn.Linear(200, 200)),
            nn.Tanh(),
            layer_init(nn.Linear(200, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), probs

    def load_weights(self, env_name, checkpoint_name=None):
        if checkpoint_name is None:
            # path = "/home/sara/repositories/RL-skill-extraction/models/self_contained_skill_ppo_models/antmaze-medium-diverse-v0/antmaze-medium-diverse-v0_seed_2_kl_True_initialized_TrueadaptiveTrue_weights.pth"
            path = "/models/self_contained_skill_ppo_models/antmaze-medium-diverse-v0/antmaze-medium-diverse-v0_seed_2_weights.pth"
        else:
            path = "/models/self_contained_skill_ppo_models/" + env_name + "/" + checkpoint_name
        checkpoint = torch.load(path, map_location=torch.device('cuda'))
        self.actor_mean.load_state_dict(checkpoint["policy_mean_state_dict"])
        self.actor_logstd = checkpoint["policy_std_state_dict"]
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        # optim.load_state_dict(checkpoint["optimizer_state_dict"])
        state_mean = checkpoint["state_rms_mean"]
        state_var = checkpoint["state_rms_var"]
        return self, optim, state_mean, state_var


class Play:
    def __init__(self, env, env_name, checkpoint_name, agent, max_episode=10000):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        _, _, state_rms_mean, state_rms_var = self.agent.load_weights(env_name, checkpoint_name)
        self.env.envs[0].obs_rms.mean = state_rms_mean
        self.env.envs[0].obs_rms.var = state_rms_var
        self.device = "cuda"

    def evaluate(self):
        eps_return = []
        traj = []
        i = 0
        while i <= 100000:
            s = self.env.reset()
            self.env.envs[0].set_target_goal()
            episode_reward = 0
            done = False
            traj.append(self.env.envs[0].get_xy())
            while not done:
                i += 1
                s = torch.FloatTensor(s).to(self.device)

                action, _, _, _, _ = self.agent.get_action_and_value(s)
                action = action.cpu().numpy()
                s_, r, done, info = self.env.step(action)
                xy = self.env.envs[0].get_xy()
                traj.append((xy[0], xy[1]))

                episode_reward += r
                s = s_
                if "episode" in info[0].keys():
                    eps_return.append(info[0]['episode']['r'])
                    print(i, self.env.envs[0].target_goal - self.env.envs[0].get_xy())
                    if i % 100 == 0:
                        print(info[0]['episode']['r'])
        print("Avg episodic return:", np.asarray(eps_return).mean())
        return eps_return, traj


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, reward_type='dense')  # , unwrap_time=True)  # gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        # env = SkillConcatenated(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def draw_maze(ax, trajectories, target_goal):
    traj_x = [x[0] for x in trajectories]
    traj_y = [x[1] for x in trajectories]
    ax.scatter(traj_x, traj_y, ls='dotted', linewidth=1, color='blue', alpha=0.03)
    ax.scatter(target_goal[0], target_goal[1], marker='x', c='red', )

    plt.show()
    return ax


if __name__ == '__main__':
    envs = gym.vector.SyncVectorEnv(
        [make_env("antmaze-medium-diverse-v0", 1 + i, i, False, "test") for i in range(1)])
    agent = Agent(envs).to('cuda')
    player = Play(envs, 'antmaze-medium-diverse-v0', checkpoint_name=None, agent=agent)
    _, traj = player.evaluate()
    target = envs.envs[0].target_goal
    fig, ax = plt.subplots()

    ax = draw_maze(ax, traj, target)
