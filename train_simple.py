import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from poker_wrapper import PokerGymWrapper
from poker_model import POLICY_KWARGS

def make_env():
    return PokerGymWrapper(opponent_policy="random")

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=POLICY_KWARGS,
    verbose=1,
    device='cpu',
    tensorboard_log="./logs/tb/",
    n_steps=256,
    batch_size=64,
    n_epochs=5,
)

model.learn(total_timesteps=5000)
model.save("test_model")
print("训练完成！")
