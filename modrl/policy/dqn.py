from modrl.core.trainer import  Base, TrainConfig
from modrl.utils import linear_scheduler

import wandb
import random
import  numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


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
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space is not of type gym.spaces.Discrete")
        self.actions_dims = np.array(env.single_action_space.shape).prod()
        self.obs_dims = np.array(env.single_observation_space.shape).prod()
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


    def eval(self):
        pass

    def load(self):
        pass

    def save(self):
        pass