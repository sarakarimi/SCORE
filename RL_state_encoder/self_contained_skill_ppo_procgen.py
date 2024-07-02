import argparse
import os
import random
import time
from collections import OrderedDict
from distutils.util import strtobool
import h5py
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import ObservationWrapper
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from AE.utils import load_ae_model, get_trained_ae_model


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="coinrun",
                        help="the id of the gym environment")
    parser.add_argument("--distribution-mode", type=str, default="hard",
                        help="the mode of the environment")
    parser.add_argument("--num-levels", type=int, default=64,
                        help="the number of levels to train on")
    parser.add_argument("--start-level", type=int, default=0,
                        help="the level to start from")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=int(7.5e6),
                        help="total timesteps of the experiments")
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
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--copy-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="initialize weights from trained AE")
    parser.add_argument("--decoder-kl", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use KL regularization using AE")
    parser.add_argument("--adaptive-kl-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use KL regularization using AE")
    parser.add_argument("--evaluate", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Set to evaluation mode")
    parser.add_argument("--model-path", type=str,
                        default="../models/self_contained_skill_ppo_procgen_models/coinrun/coinrun_easy_seed_1_weights.pth",
                        help="path to the saved model")
    parser.add_argument("--skill-model-path", type=str,
                        default="../models/AE_models/procgen-opal/600.pt",
                        help="path to the saved model")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
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


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
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
        self.network = nn.Sequential(*conv_seqs)
        # self.actor = layer_init(nn.Linear(256 + 8, envs.single_action_space.n), std=0.01)

        self.actor = nn.Sequential(nn.Linear(in_features=256 + 8, out_features=200),
                                   nn.ReLU(),
                                   nn.Linear(in_features=200, out_features=200),
                                   nn.ReLU(),
                                   layer_init(nn.Linear(in_features=200, out_features=envs.single_action_space.n),
                                              std=0.01))
        # self.critic = layer_init(nn.Linear(256, 1), std=1)
        self.critic = nn.Sequential(nn.Linear(in_features=256 + 8, out_features=200),
                                    nn.ReLU(),
                                    nn.Linear(in_features=200, out_features=200),
                                    nn.ReLU(),
                                    layer_init(nn.Linear(in_features=200, out_features=1),
                                               std=1.0))

    def get_value(self, x, skill=None):
        encoded_state = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        hidden = torch.cat([skill, encoded_state], dim=-1)
        return self.critic(hidden)  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None, skill=None, skill_prior=None):
        if skill is None:
            skill, _ = skill_prior.act(None, x, encode_state=True)
        encoded_state = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        hidden = torch.cat([skill, encoded_state], dim=-1)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, skill, probs.log_prob(action), probs.entropy(), self.critic(hidden), probs

    def initialize_weights(self, copy_model):
        state_dict = copy_model.state_dict()
        new_dict = OrderedDict()
        for key1, key2 in list(zip(list(copy_model.state_dict().keys())[1:], list(self.state_dict().keys())[:-6])):
            new_dict[key2] = state_dict.pop(key1)
        for key in list(self.state_dict().keys())[-6:]:
            new_dict[key] = self.state_dict().pop(key)
        self.load_state_dict(new_dict)

    def save_weights(self, env_name, env_mode, optim, seed, kl=True, initialized=True, adaptive_kl_weight=True):
        path = "../models/self_contained_skill_ppo_procgen_models/" + env_name + "/" + env_name + "_" + env_mode + "_seed_" + str(
            seed) + "_kl_" + str(kl) + "_initialized_" + str(initialized) + "adaptive" + str(
            adaptive_kl_weight) + "_weights.pth"
        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "network_state_dict": self.network.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cuda'))
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.network = checkpoint["network_state_dict"]
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        # optim.load_state_dict(checkpoint["optimizer_state_dict"])


class Play:
    def __init__(self, env, path, agent, max_steps=1000000):
        self.env = env
        self.max_steps = max_steps
        self.agent = agent
        self.agent.load_weights(path)
        self.device = "cuda"

    def evaluate(self, epsilon=None):
        eps_return, states, actions = [], [], []
        data = {'observations': [], 'actions': []}
        i = 0
        while i <= self.max_steps:
            s = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                i += 1
                action, _, _, _, _ = self.agent.get_action_and_value(torch.FloatTensor(s).unsqueeze(0).to(self.device))
                action = action.cpu().numpy()
                if epsilon:
                    action = [random.randint(0, 14)] if random.random() < epsilon else action
                data['observations'].append(s[0])
                data['actions'].append(action[0])
                s_, r, done, info = self.env.step(action[0])
                episode_reward += r
                s = s_
                if "episode" in info.keys():
                    eps_return.append(info['episode']['r'])
                    print(i, info['episode']['r'])
        print("Avg episodic return:", np.asarray(eps_return).mean(), i)
        return eps_return, data


def main(seed):
    args = parse_args()
    args.seed = seed
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{args.distribution_mode}_{int(time.time())}"
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
    envs = ProcgenEnv(num_envs=args.num_envs, env_name=args.gym_id, num_levels=args.num_levels,
                      start_level=args.start_level,
                      distribution_mode=args.distribution_mode)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.evaluate:  # evaluation loop
        envs = gym.make("procgen-coinrun-v0", distribution_mode=args.ditribution_mode)  # , render_mode="human")
        player = Play(envs, args.model_path, agent)
        player.evaluate()
        envs.close()
        writer.close()
    else:
        # Path to the trained skill VAW model to copy weights from the decoder to PPO policy net
        path = args.skill_model_path
        AE_model = get_trained_ae_model(state_dim=envs.observation_space['rgb'].shape,
                                        action_dim=envs.action_space.shape, path=path, env_name='procgen').to('cuda')
        prior = AE_model.prior

        if args.copy_weight:
            agent.initialize_weights(AE_model.decoder)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        skills = torch.zeros((args.num_steps, args.num_envs) + (8,)).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, skill, logprob, _, value, _ = agent.get_action_and_value(next_obs, skill_prior=prior)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                skills[step] = skill

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs, skill).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_skills = skills.reshape((-1,) + (8,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, _, newlogprob, entropy, newvalue, dist = agent.get_action_and_value(b_obs[mb_inds],
                                                                                           action=b_actions.long()[mb_inds],
                                                                                           skill=b_skills[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # regularize with decoder
                    if args.decoder_kl:
                        logits, _ = AE_model.decoder(latent=b_skills[mb_inds], state=b_obs[mb_inds], encode_state=True)
                        decoder_dist = Categorical(logits=logits)
                        uncertainty_weight = 1
                        if args.adaptive_kl_weight:
                            uncertainty_weight = decoder_dist.log_prob(b_actions[mb_inds])
                            uncertainty_weight = torch.exp(uncertainty_weight).sum()
                        kl_divergence = torch.distributions.kl_divergence(dist, decoder_dist).sum()
                        kl_loss = torch.clamp(kl_divergence, -100, 100)

                        p_loss = pg_loss - uncertainty_weight * 0.001 * kl_loss
                    else:
                        p_loss = pg_loss

                    entropy_loss = entropy.mean()
                    loss = p_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/p_loss", p_loss.item(), global_step)
            writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("losses/weights", uncertainty_weight.item(), global_step)

            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if update % 100 == 0:
                print("saving checkpoint at ", update)
                agent.save_weights(env_name=args.gym_id, env_mode=args.distribution_mode, optim=optimizer,
                                   seed=args.seed, kl=args.decoder_kl,
                                   initialized=args.copy_weight,
                                   adaptive_kl_weight=args.adaptive_kl_weight)
        agent.save_weights(env_name=args.gym_id, env_mode=args.distribution_mode, optim=optimizer, seed=args.seed,
                           kl=args.decoder_kl,
                           initialized=args.copy_weight,
                           adaptive_kl_weight=args.adaptive_kl_weight)

        envs.close()
        writer.close()


if __name__ == "__main__":
    for seed in [1]:
        main(seed)
