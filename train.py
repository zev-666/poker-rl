# train.py
import os
import torch
import torch_directml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from poker_wrapper import PokerGymWrapper
from poker_model   import POLICY_KWARGS
from callbacks     import PokerMetricsCallback, EarlyStopOnWinRate

# ── 設備設置 ──────────────────────────────────────────────────
dml = torch_directml.device()
print(f"DirectML 設備就緒: {dml}")

# ── 超參數（針對撲克調整過的版本）────────────────────────────
PPO_PARAMS = dict(
    # 策略網路
    policy      = "MlpPolicy",
    policy_kwargs = POLICY_KWARGS,

    # 環境
    # n_envs 個並行環境加速採樣（見下方 make_env）

    # Rollout 設置
    # 撲克每局約 20~40 步，n_steps=2048 約等於 50~100 局
    n_steps     = 2048,

    # 訓練設置
    batch_size  = 256,      # 比預設 64 大，撲克狀態變化多
    n_epochs    = 10,       # 每次 rollout 重複訓練 10 次
    learning_rate = 3e-4,

    # PPO clip（預設 0.2 通常不用動）
    clip_range  = 0.2,

    # 折扣因子：撲克回合短，用 0.99 即可
    gamma       = 0.99,

    # GAE lambda：0.95 平衡偏差/方差
    gae_lambda  = 0.95,

    # Entropy 係數：撲克需要探索，用 0.01 鼓勵策略多樣性
    # 若動作分佈坍縮到全 fold 或全 call，可提高到 0.05
    ent_coef    = 0.01,

    # Value function 係數
    vf_coef     = 0.5,

    # 梯度裁剪
    max_grad_norm = 0.5,

    # 監控
    verbose     = 1,
    tensorboard_log = "./logs/tb/",

    # DirectML 設備
    device      = dml,
)

N_ENVS = 4   # 並行環境數，根據你的記憶體調整


def make_env(rank: int, opponent: str = "random"):
    """工廠函數，建立帶有獨立 seed 的環境"""
    def _init():
        env = PokerGymWrapper(opponent_policy=opponent)
        env.reset(seed=42 + rank)
        return env
    return _init


def train():
    os.makedirs("./logs/tb",   exist_ok=True)
    os.makedirs("./models",    exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # ── 建立並行環境 ──────────────────────────────────────────
    # 第一階段：對手全是隨機 agent
    envs = DummyVecEnv([make_env(i, "random") for i in range(N_ENVS)])
    envs = VecMonitor(envs, "./logs/monitor")  # 自動記錄 ep_rew_mean

    # ── 建立模型 ─────────────────────────────────────────────
    model = PPO(env=envs, **PPO_PARAMS)

    print(f"模型參數量: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"觀測空間:   {envs.observation_space}")
    print(f"動作空間:   {envs.action_space}")
    print("開始訓練...\n")

    # ── 回調函數 ─────────────────────────────────────────────
    callbacks = CallbackList([
        PokerMetricsCallback(verbose=1),

        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = "./checkpoints/",
            name_prefix = "poker_ppo",
            verbose     = 1,
        ),

        EarlyStopOnWinRate(
            target_win_rate = 0.60,  # 對隨機對手勝率達 60% 就進入第二階段
            check_freq      = 50_000,
        ),
    ])

    # ── 第一階段訓練：對抗隨機對手 ───────────────────────────
    model.learn(
        total_timesteps = 1_000_000,
        callback        = callbacks,
        progress_bar    = True,
        reset_num_timesteps = True,
        tb_log_name     = "stage1_vs_random",
    )
    model.save("./models/stage1_vs_random")
    print("\n第一階段完成，模型已儲存")

    # ── 第二階段訓練：對抗「永遠跟注」對手 ──────────────────
    # 換一個更強的對手，繼續訓練（不重置網路權重）
    envs2 = DummyVecEnv([make_env(i, "call_always") for i in range(N_ENVS)])
    envs2 = VecMonitor(envs2, "./logs/monitor2")
    model.set_env(envs2)

    model.learn(
        total_timesteps = 500_000,
        callback        = PokerMetricsCallback(),
        reset_num_timesteps = False,   # 接續 step 計數
        tb_log_name     = "stage2_vs_caller",
    )
    model.save("./models/stage2_final")
    print("\n第二階段完成，最終模型已儲存")

    envs.close()
    envs2.close()


if __name__ == "__main__":
    train()