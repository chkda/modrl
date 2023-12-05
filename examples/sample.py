from modrl.policy.dqn import DQN
from modrl.core.trainer import TrainConfig

import gymnasium as gym

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"video/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

config = TrainConfig(num_steps=1000,
                     lr=0.001,
                     exp_name="test-1",
                     wandb_project_name="test-exp"
                     )



envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, True, "test-1")])

policy = DQN(config=config, env=envs)

policy.train()