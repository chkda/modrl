from modrl.core.trainer import  Base, TrainConfig
from modrl.utils.scheduler import linear_scheduler
from modrl.utils.buffers import ReplayBuffer

import wandb
import random
import  numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from  stable_baselines3.common.buffers import ReplayBuffer


class QNetwork(nn.Module):

    def __init__(self, action_dims: int, obs_dims: int):
        super().__init__()
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.layer1 = nn.Linear(self.obs_dims,120)
        self.layer2 = nn.Linear(120, 84)
        self.layer3 = nn.Linear(84, self.action_dims)

    def forward(self, inp):
        x = self.layer1(inp)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        action_values = self.layer3(x)

        return  action_values


class DQN(Base):
    def __init__(self, config: TrainConfig, env: gym.vector.VectorEnv):
        self.config = config
        self.actions_dims = np.array(env.single_action_space.shape).prod(dtype="uint8")
        self.obs_dims = np.array(env.single_observation_space.shape).prod(dtype="uint8")
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = QNetwork(self.actions_dims, self.obs_dims)
        self.target_network = QNetwork(self.actions_dims, self.obs_dims)
        self.optimiser = optim.Adam(self.network.parameters(), lr=self.config.lr)
        self.target_network.load_state_dict(self.network.state_dict())

    def train(self):
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project_name,
                config=self.config.to_dict(),
                monitor_gym=self.config.monitor_gym,
                sync_tensorboard=self.config.sync_tensorboard,
                name=self.config.exp_name
            )

        writer = SummaryWriter(f"runs/{self.config.exp_name}")

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.network = self.network.to(self.device)
        self.target_network = self.target_network.to(self.device)

        obs, _ = self.env.reset(seed=self.config.seed)

        rb = ReplayBuffer(self.config.buffer_size)

        for global_step in range(self.config.num_steps):
            epsilon = linear_scheduler(self.config.epsilon_start, self.config.epsilon_end, global_step, self.config.num_steps)
            if random.random() < epsilon:
                actions = np.array([self.env.single_action_space.sample() for _ in range(self.config.num_envs)])
            else:
                q_values = self.network(torch.Tensor(obs).to(self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            real_next_obs = next_obs.copy()

            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            rb.add(obs, real_next_obs, actions, rewards, terminations)

            if global_step > self.config.learning_starts:
                if global_step % self.config.train_frequency == 0:
                    data = rb.sample(self.config.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.config.gamma * target_max * (1 - data.dones.flatten())

                    old_val = self.network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(old_val, td_target)

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/td_target", td_target.mean().item(), global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()

                if global_step % self.config.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                                     self.network.parameters()):
                        target_network_param.data.copy_(
                            self.config.tau * q_network_param.data + (1 - self.config.tau) * target_network_param.data
                        )

    def eval(self):
        episodic_returns = []
        epsilon = self.config.epsilon_end
        obs, _ = self.env.reset(seed=self.config.seed)
        for step in range(self.config.eval_steps):
            if random.random() < epsilon:
                actions = self.env.single_action_space.sample()
            else:
                q_values = self.target_network(torch.Tensor(obs).to(self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, _, _, _, infos = self.env.step(actions)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]

            obs = next_obs

        return episodic_returns

    def load(self, path: str):
        self.network = QNetwork(self.actions_dims, self.obs_dims)
        self.target_network = QNetwork(self.actions_dims, self.obs_dims)
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))
        print("Weights loaded")
        return

    def save(self, path: str):
        torch.save(self.target_network.parameters(), path)
        print("Model saved")
        return