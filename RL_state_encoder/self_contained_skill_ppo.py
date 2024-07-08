import argparse
import os
import random
import time
from collections import OrderedDict
from distutils.util import strtobool
import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from gym import ObservationWrapper
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from AE.env_wrapper import antmaze_costume_env
from AE.utils import load_ae_model, get_trained_ae_model
from gym.spaces import Box
import d4rl


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="antmaze-xl-diverse-v0",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=3500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
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
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
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
    parser.add_argument("--evaluate", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Set to evaluation mode")
    parser.add_argument("--model-path", type=str,
                        default="../models/self_contained_skill_ppo_models/antmaze-xl-diverse-v0/antmaze-xl-diverse-v0_seed_5_kl_True_initialized_TrueadaptiveTrue_weights.pth",
                        help="path to the saved model")
    # "models/self_contained_skill_ppo_models/antmaze-large-diverse-v0/antmaze-large-diverse-v0_seed_5_kl_True_initialized_TrueadaptiveTrue_weights.pth"
    # "models/self_contained_skill_ppo_models/antmaze-medium-diverse-v0/antmaze-medium-diverse-v0_seed_2_kl_True_initialized_TrueadaptiveTrue_weights.pth"
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        # env = gym.make(gym_id, reward_type='dense')  # , unwrap_time=True)  # gym.make(gym_id)
        env = antmaze_costume_env()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = SkillConcatenated(env)
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


class SkillConcatenated(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        state_dim = env.observation_space.shape[0] + 8
        self.observation_space = Box(shape=(state_dim,), low=-np.inf, high=np.inf)
        path = "../models/AE_models/" + env.spec.name + "-v" + str(env.spec.version) + "-opal/600.pt"
        AE_model = load_ae_model(env, path).to('cuda')
        self.state_encoder = AE_model.prior
        self.steps = 0
        self.p_encoded_state = None

    def observation(self, obs):
        tensor_obs = torch.Tensor(np.expand_dims(obs, axis=0)).to('cuda')
        if self.steps == 0:
            self.p_encoded_state = self.state_encoder.act(None, tensor_obs, deterministic=True)
        # uncomment if you want to roll out policy 10 times using the same skill
        #     self.steps += 1
        # if 9 > self.steps > 0:
        #     self.steps += 1
        # if self.steps == 9:
        #     self.steps = 0
        concat_obs = torch.cat([self.p_encoded_state, tensor_obs], dim=-1)
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

    def initialize_weights(self, copy_model):
        state_dict = copy_model.state_dict()
        new_dict = OrderedDict()
        new_dict['actor_logstd'] = state_dict.pop('logstd')
        for key1, key2 in list(zip(list(copy_model.state_dict().keys())[1:], list(self.state_dict().keys())[7:])):
            new_dict[key2] = state_dict.pop(key1)
        for key in list(self.state_dict().keys())[1:7]:
            new_dict[key] = self.state_dict().pop(key)
        self.load_state_dict(new_dict)

    def save_weights(self, env_name, env, optim, seed, kl=True, initialized=True, adaptive_kl_weight=True):
        path = "../models/self_contained_skill_ppo_models/" + env_name + "/" + env_name + "_seed_" + str(
            seed) + "_kl_" + str(kl) + "_initialized_" + str(initialized) + "adaptive" + str(
            adaptive_kl_weight) + "_weights.pth"
        torch.save({"policy_mean_state_dict": self.actor_mean.state_dict(),
                    "policy_std_state_dict": self.actor_logstd,
                    "critic_state_dict": self.critic.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "state_rms_mean": env.obs_rms.mean,
                    "state_rms_var": env.obs_rms.var,
                    }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cuda'))
        self.actor_mean.load_state_dict(checkpoint["policy_mean_state_dict"])
        self.actor_logstd = checkpoint["policy_std_state_dict"]
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        # optim.load_state_dict(checkpoint["optimizer_state_dict"])
        state_mean = checkpoint["state_rms_mean"]
        state_var = checkpoint["state_rms_var"]
        return self, optim, state_mean, state_var


class Play:
    def __init__(self, env, path, agent, max_steps=100000):
        self.env = env
        self.max_steps = max_steps
        self.agent = agent
        _, _, state_rms_mean, state_rms_var = self.agent.load_weights(path)
        self.env.envs[0].obs_rms.mean = state_rms_mean
        self.env.envs[0].obs_rms.var = state_rms_var
        self.device = "cuda"

    def evaluate(self):
        eps_return = []
        success_rate = []
        i = 0
        while i <= self.max_steps:
            s = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                i += 1
                s = torch.FloatTensor(s).to(self.device)

                action, _, _, _, _ = self.agent.get_action_and_value(s)
                action = action.cpu().numpy()
                s_, r, done, info = self.env.step(action)
                episode_reward += r
                s = s_
                if "episode" in info[0].keys():
                    eps_return.append(info[0]['episode']['r'])
                    if done and 'TimeLimit.truncated' not in info[0]:
                        success_rate.append(1)
                    else:
                        success_rate.append(0)
                    # if i % 100 == 0:
                    #     print(i, info[0]['episode']['r'])
        print("Avg episodic return:", np.asarray(eps_return).mean(), np.asarray(success_rate).mean())
        return np.asarray(eps_return).mean()


def main(seed):
    args = parse_args()
    args.seed = seed
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__kl_{str(args.decoder_kl)}___initialized_{args.copy_weight}__adaptive_{args.adaptive_kl_weight}__{int(time.time())}"
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.evaluate:  # evaluation loop
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )

        player = Play(envs, args.model_path, agent)
        return_avg = player.evaluate()
        envs.close()
        writer.close()
        return return_avg

    else:  # training code
        # Path to the trained skill embedding model
        path = "../models/AE_models/" + args.gym_id + "-opal/600_old.pt"
        AE_model = get_trained_ae_model(state_dim=envs.single_observation_space.shape[0] - 8,
                                        action_dim=envs.single_action_space.shape[0], path=path).to('cuda')
        if args.copy_weight:
            agent.initialize_weights(AE_model.decoder)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size

        success_rate = []
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
                    action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                #  execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        if done and 'TimeLimit.truncated' not in item:
                            writer.add_scalar("charts/win_rate", 1, global_step)
                            success_rate.append(1)
                        else:
                            writer.add_scalar("charts/win_rate", 0, global_step)
                            success_rate.append(0)

                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
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

                    _, newlogprob, entropy, newvalue, dist = agent.get_action_and_value(b_obs[mb_inds],
                                                                                        b_actions[mb_inds])
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

                    # regularize with decoder and adaptive weighting of the kl term
                    if args.decoder_kl:
                        mean, std = AE_model.decoder(latent=b_obs[mb_inds], state=None)
                        decoder_dist = Normal(mean, std)
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
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/decoder_kl", kl_loss.item(), global_step)
            # writer.add_scalar("losses/weights", uncertainty_weight.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/ma_win_rate", np.mean(success_rate[-100:]), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if update % 10 == 0:
                print("saving checkpoint at ", update)
                agent.save_weights(env_name=args.gym_id, env=envs.envs[0], optim=optimizer, seed=args.seed,
                                   kl=args.decoder_kl, initialized=args.copy_weight,
                                   adaptive_kl_weight=args.adaptive_kl_weight)
        agent.save_weights(env_name=args.gym_id, env=envs.envs[0], optim=optimizer, seed=args.seed, kl=args.decoder_kl,
                           initialized=args.copy_weight,
                           adaptive_kl_weight=args.adaptive_kl_weight)
        envs.close()
        writer.close()
        return None


if __name__ == "__main__":
    avg_returns = []
    for seed in [1, 2, 3, 4, 5]:
        avg_return = main(seed)
