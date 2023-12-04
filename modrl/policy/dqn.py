from modrl.core.trainer import  Base

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def load(self):
        pass

    def save(self):
        pass