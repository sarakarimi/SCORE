# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
from procgen import ProcgenEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-skill-learning",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="coinrun",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=7500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=5,
                        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=int(10000),
                        help="the replay memory buffer size")  # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="target smoothing coefficient (default: 1)")  # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=200,
                        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
                        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--update-frequency", type=int, default=4,
                        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
                        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
                        help="coefficient for scaling the autotune entropy target")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, num_envs, gamma, capture_video, run_name):
    def thunk():
        # env setup
        envs = ProcgenEnv(num_envs=num_envs, env_name=env_id, num_levels=64, start_level=0, distribution_mode="hard",
                          render_mode="rgb_array")
        envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space["rgb"]
        envs.is_vector_env = True
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        if capture_video:
            envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
        envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        # env.action_space.seed(seed)
        return envs

    return thunk


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)

#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env, skip=4)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 4)

#         env.action_space.seed(seed)
#         return env

#     return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


def permute_and_forward(conv, x):
    return conv.forward(x.permute((0, 3, 1, 2)) / 255.0)


class ReplayData:
    def __init__(self, observations, actions, rewards, next_observations, dones):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones


class ReplayBufferV2():
    def __init__(self, max_size, n_envs, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, n_envs, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, n_envs, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_envs))
        self.reward_memory = np.zeros((self.mem_size, n_envs))
        self.terminal_memory = np.zeros((self.mem_size, n_envs), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = torch.FloatTensor(self.state_memory[batch, 0])
        states_ = torch.FloatTensor(self.new_state_memory[batch, 0])
        actions = torch.Tensor(self.action_memory[batch, 0])
        rewards = torch.FloatTensor(self.reward_memory[batch, 0])
        dones = torch.Tensor(self.terminal_memory[batch, 0])
        return ReplayData(states, actions, rewards, states_, dones)


def get_conv(envs):
    h, w, c = envs.single_observation_space.shape
    shape = (c, h, w)
    conv_seqs = []
    for out_channels in [16, 32, 32]:
        conv_seq = ConvSequence(shape, out_channels)
        shape = conv_seq.get_output_shape()
        conv_seqs.append(conv_seq)
    conv_seqs += [
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
        nn.ReLU(),
    ]
    conv = nn.Sequential(*conv_seqs)
    return conv


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs, conv):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        # self.conv = nn.Sequential(
        #     layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=4, padding=1)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
        #     nn.Flatten(),
        # )
        self.conv = conv
        with torch.inference_mode():
            output_dim = \
            permute_and_forward(self.conv, torch.zeros(1, *obs_shape).to(next(self.conv.parameters()).device)).shape[1]
        # with torch.inference_mode():
        #     output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 200))
        self.fc_q = layer_init(nn.Linear(200, envs.single_action_space.n))

    def forward(self, x):
        # x = F.relu(self.conv(x / 255.0))
        # x = torch.FloatTensor(x)
        x = x.to(next(self.conv.parameters()).device)
        hidden = self.conv(x.permute((0, 3, 1, 2)) / 255.0)
        x = F.relu(self.fc1(hidden))
        q_vals = self.fc_q(x)
        return q_vals

    def save_weights(self, env_name, optim, seed, name=None):
        path = "../models/self_contained_sac_procgen_models/" + env_name + "/" + env_name + "_seed_" + str(
            seed) + "_" + name + "_QNet_weights.pth"
        torch.save({"fc1_state_dict": self.fc1.state_dict(),
                    "fc_q_state_dict": self.fc_q.state_dict(),
                    "conv_state_dict": self.conv.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    }, path)


class Actor(nn.Module):
    def __init__(self, envs, conv):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = conv
        # self.conv = nn.Sequential(
        #     layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=4, padding=1)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
        #     nn.Flatten(),
        # )

        with torch.inference_mode():
            output_dim = \
            permute_and_forward(self.conv, torch.zeros(1, *obs_shape).to(next(self.conv.parameters()).device)).shape[1]
            # _x = torch.zeros(1, *obs_shape)
            # output_dim = self.conv(_x.permute((0, 3, 1, 2)) / 255.0).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 200))
        self.fc_logits = layer_init(nn.Linear(200, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    # def get_action(self, x):
    #     logits = self(x / 255.0)
    #     policy_dist = Categorical(logits=logits)
    #     action = policy_dist.sample()
    #     # Action probabilities for calculating the adapted soft-Q loss
    #     action_probs = policy_dist.probs
    #     log_prob = F.log_softmax(logits, dim=1)
    #     return action, log_prob, action_probs

    def get_action(self, x):
        x = x.to(next(self.conv.parameters()).device)
        # x = torch.FloatTensor(x)
        hidden = self.conv(x.permute((0, 3, 1, 2)) / 255.0)
        logits = self(hidden)
        action_probs = torch.softmax(logits, dim=1)
        action = torch.argmax(action_probs, dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

    def save_weights(self, env_name, optim, seed):
        path = "../models/self_contained_sac_procgen_models/" + env_name + "/" + env_name + "_seed_" + str(seed) + "actor_weights.pth"
        torch.save({"fc_logits_state_dict": self.fc_logits.state_dict(),
                    "fc1_state_dict": self.fc1.state_dict(),
                    "conv_state_dict": self.conv.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    }, path)


def main(seed):
    #     import stable_baselines3 as sb3

    #     if sb3.__version__ < "2.0":
    #         raise ValueError(
    #             """Ongoing migration: run the following command to install the new dependencies:

    # poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1"
    # """
    #         )
    args = parse_args()
    args.seed = seed
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__hard__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            reinit=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args.env_id, args.num_envs, args.gamma, args.capture_video, run_name)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    conv = get_conv(envs).to(device)
    actor = Actor(envs, conv).to(device)
    qf1 = SoftQNetwork(envs, conv).to(device)
    qf2 = SoftQNetwork(envs, conv).to(device)
    qf1_target = SoftQNetwork(envs, conv).to(device)
    qf2_target = SoftQNetwork(envs, conv).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBufferV2(args.buffer_size, args.num_envs, envs.single_observation_space.shape,
                        envs.single_action_space.n)
    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     handle_timeout_termination=False,
    # )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    # scores = []
    global_step = 0

    for _ in range(args.total_timesteps // args. num_envs):
        global_step += 1 * args.num_envs
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs, rewards, terminations, infos = envs.step(actions)
        # scores.append(rewards.reshape(-1, 1))
        # scores = scores[-100:]
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for item in infos:
            if "episode" in item.keys():
                print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                break
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         # Skip the envs that are not done
        #         if "episode" not in info:
        #             continue
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        rb.store_transition(obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample_buffer(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten().to(device) + (
                                1 - data.dones.flatten().to(device)) * args.gamma * (min_qf_next_target.to(device))

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations).to(device)
                qf2_values = qf2(data.observations).to(device)
                qf1_a_values = qf1_values.gather(1, data.actions.long().view(-1, 1).to(device)).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long().view(-1, 1).to(device)).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (
                                action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)

                print("SPS:", int(global_step / (time.time() - start_time)))
                # print("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                # print("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                # print("losses/qf1_loss", qf1_loss.item(), global_step)
                # print("losses/qf2_loss", qf2_loss.item(), global_step)
                # print("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                # print("losses/actor_loss", actor_loss.item(), global_step)
                # print("losses/alpha", alpha, global_step)
                # print("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                # _scores = np.hstack(scores).reshape(-1)
                # _total = np.sum(_scores)
                # _len = _scores.shape[0]
                # _avg_score =  _total/_len
                # print("avg score: ", _avg_score)
                # writer.add_scalar("charts/avg_total_rewards", global_step, _avg_score)
                # print("-" * 50)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                if global_step % 100000 == 0:
                    print("saving checkpoint at ", global_step)
                    actor.save_weights(env_name=args.env_id, optim=a_optimizer, seed=args.seed)
                    qf1.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf1")
                    qf2.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf2")
                    qf1_target.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf1_target")
                    qf2_target.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf2_target")

    qf1.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf1")
    qf2.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf2")
    qf1_target.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf1_target")
    qf2_target.save_weights(env_name=args.env_id, optim=q_optimizer, seed=args.seed, name="qf2_target")
    actor.save_weights(env_name=args.env_id,  optim=a_optimizer, seed=args.seed)
    envs.close()
    writer.close()


if __name__ == "__main__":
    for seed in [1, 2, 3, 4, 5]:
        main(seed)
