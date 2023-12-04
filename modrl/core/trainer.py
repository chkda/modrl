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


class TrainConfig:

    num_steps: int
    lr: float
    buffer_size: int
    gamma: float
    target_network_frequency: int
    max_grad_norm: float
    batch_size: int
    epsilon_start: float
    epsilon_end: float
    exploration_fraction: float
    learning_starts: int
    train_frequency: int
    seed: int
    tau: int
    num_envs: int
    vectorise: bool
    use_wandb: bool = False
    wandb_project_name: str
    monitor_gym: bool = False
    sync_tensorboard: bool = False
    exp_name: str
    eval_steps: int

    def to_dict(self):
        return {attr: getattr(self, attr, None) for attr in self.__annotations__}
