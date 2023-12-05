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
        return [self.storage[idx] for idx in batch_idxs]