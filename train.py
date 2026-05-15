import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from poker_wrapper import PokerGymWrapper
from poker_model import POLICY_KWARGS
from callbacks import PokerMetricsCallback, EarlyStopOnWinRate

# 设备（CPU 稳定模式）
device = 'cpu'
print("使用 CPU 训练")

# 超参数
PPO_PARAMS = dict(
    policy          = "MlpPolicy",
    policy_kwargs   = POLICY_KWARGS,
    n_steps         = 512,
    batch_size      = 256,
    n_epochs        = 10,
    learning_rate   = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    ent_coef        = 0.05,       # 提高探索，鼓励弃牌
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
    tensorboard_log = "./logs/tb/",
    device          = device,
)

N_ENVS = 1

def make_env(rank: int, opponent: str = "random"):
    def _init():
        env = PokerGymWrapper(opponent_policy=opponent)
        env.reset(seed=42 + rank)
        return env
    return _init

def train():
    os.makedirs("./logs/tb",   exist_ok=True)
    os.makedirs("./models",    exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # 第一阶段：vs 随机对手
    envs = DummyVecEnv([make_env(i, "random") for i in range(N_ENVS)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True)

    model = PPO(env=envs, **PPO_PARAMS)
    print(f"模型參數量: {sum(p.numel() for p in model.policy.parameters()):,}")

    callbacks = CallbackList([
        PokerMetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=50_000,
            save_path="./checkpoints/",
            name_prefix="poker_stage1"
        ),
        EarlyStopOnWinRate(
            target_win_rate=0.65,
            check_freq=50_000,
            verbose=1
        ),
    ])

    print("開始第一階段訓練（隨機對手）...")
    model.learn(
        total_timesteps=1_000_000,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
        tb_log_name="stage1_vs_random",
    )
    model.save("./models/stage1_vs_random")
    envs.save("vec_normalize_stage1.pkl")
    print("第一階段完成！")

    # 第二阶段：vs 跟注站对手
    envs2 = DummyVecEnv([make_env(i, "call_always") for i in range(N_ENVS)])
    envs2 = VecNormalize(envs2, norm_obs=True, norm_reward=True)
    model.set_env(envs2)

    print("開始第二階段訓練（跟注站對手）...")
    model.learn(
        total_timesteps=500_000,
        callback=PokerMetricsCallback(verbose=1),
        reset_num_timesteps=False,
        tb_log_name="stage2_vs_caller",
    )
    model.save("./models/stage2_final")
    envs2.save("vec_normalize_stage2.pkl")
    print("第二階段完成！最終模型已儲存")

    envs.close()
    envs2.close()

if __name__ == "__main__":
    train()
