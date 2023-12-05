import torch
import random
import numpy as np
from collections import  deque, namedtuple

Transition = namedtuple('Transition', ('observations', 'next_observations', 'actions', 'rewards', 'dones'))


class ReplayBuffer:

    def __init__(self, size=10000):
        self.storage = deque(maxlen=size)

    def add(self, s, s_, a, r, d):
        data = (s, s_, a, r, d)
        self.storage.append(Transition(*data))

    def sample(self, batch_size=32):
        batch_idxs = random.sample(range(len(self.storage)), batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []
        for idx in batch_idxs:
            data = self.storage[idx]
            print(data.observations)
            states.append(data.observations)
            next_states.append(data.next_observations)
            actions.append(data.actions)
            rewards.append(data.rewards)
            dones.append(data.dones)

        states = torch.from_numpy(np.array(states))
        next_states = torch.from_numpy(np.array(next_states))
        rewards = torch.from_numpy(np.array(rewards))
        actions = torch.from_numpy(np.array(actions))
        dones = np.array(dones)

        return Transition(states, next_states, actions, rewards, dones)