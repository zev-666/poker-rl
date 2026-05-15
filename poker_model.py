# poker_model.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class PokerFeatureExtractor(BaseFeaturesExtractor):
    """
    針對撲克設計的特徵提取器。
    把 72 維觀測先過兩層 FC，輸出 128 維特徵給 Actor/Critic 頭使用。

    為什麼不用預設 MlpPolicy 的預設網路？
    預設是 [64, 64]，對撲克這種需要長期記憶的遊戲偏小。
    這裡用 [256, 256] 並加入 LayerNorm 穩定訓練。
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input = observation_space.shape[0]  # 72

        self.network = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),          # 比 BatchNorm 更適合 RL（小 batch）
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


# 傳給 PPO policy_kwargs 的配置
POLICY_KWARGS = dict(
    features_extractor_class  = PokerFeatureExtractor,
    features_extractor_kwargs = dict(features_dim=128),
    # Actor 和 Critic 共享特徵提取器，各自獨立的頭
    net_arch = dict(pi=[128, 64], vf=[128, 64]),
)