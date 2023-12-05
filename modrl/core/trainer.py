from dataclasses import dataclass
from abc import ABC, abstractmethod


class Base(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


@dataclass
class TrainConfig:

    num_steps: int = 1000
    lr: float = 0.001
    buffer_size: int = 10000
    gamma: float = 0.99
    target_network_frequency: int = 10
    max_grad_norm: float = 0.5
    batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.3
    learning_starts: int = 1000
    train_frequency: int = 100
    seed: int = 1
    tau: float = 0.25
    num_envs: int = 1
    vectorise: bool = True
    use_wandb: bool = False
    wandb_project_name: str = "default"
    monitor_gym: bool = False
    sync_tensorboard: bool = False
    exp_name: str = "default-exp"
    eval_steps: int = 100

    def to_dict(self):
        return {attr: getattr(self, attr, None) for attr in self.__annotations__}
