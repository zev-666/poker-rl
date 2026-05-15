import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class PokerMetricsCallback(BaseCallback):
    """
    記錄撲克專用指標：
    - 各動作使用頻率（fold/call/raise 比例）
    - 勝率（episode 結束時 reward > 0 的比例）
    - 平均底池倍數
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = np.zeros(5)   # fold/call/rs/rb/allin
        self.episode_wins  = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 記錄每步動作
        actions = self.locals.get("actions", [])
        for a in actions:
            self.action_counts[int(a)] += 1

        # 記錄每局結果
        dones   = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for done, rew in zip(dones, rewards):
            if done:
                self.episode_wins.append(1 if rew > 0 else 0)
                self.episode_rewards.append(rew)

        # 每 10000 步印一次統計
        if self.n_calls % 10000 == 0 and self.action_counts.sum() > 0:
            total = self.action_counts.sum()
            labels = ["fold", "call", "raise_s", "raise_b", "allin"]
            dist   = {l: f"{c/total:.1%}"
                      for l, c in zip(labels, self.action_counts)}

            win_rate = np.mean(self.episode_wins[-200:]) if self.episode_wins else 0
            avg_rew  = np.mean(self.episode_rewards[-200:]) if self.episode_rewards else 0

            print(f"\n[Step {self.n_calls:>8,}]")
            print(f"  動作分佈: {dist}")
            print(f"  近 200 局勝率: {win_rate:.1%}")
            print(f"  近 200 局平均獎勵: {avg_rew:+.4f}")

            # 寫入 TensorBoard
            self.logger.record("poker/win_rate",   win_rate)
            self.logger.record("poker/avg_reward", avg_rew)
            for l, c in zip(labels, self.action_counts):
                self.logger.record(f"poker/action_{l}", c / total)

        return True  # 返回 False 會提前停止訓練


class EarlyStopOnWinRate(BaseCallback):
    """當勝率穩定超過閾值時自動停止訓練"""

    def __init__(self, target_win_rate=0.65, check_freq=50000, verbose=1):
        super().__init__(verbose)
        self.target  = target_win_rate
        self.check_freq = check_freq
        self._wins   = []

    def _on_step(self) -> bool:
        dones   = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for done, rew in zip(dones, rewards):
            if done:
                self._wins.append(1 if rew > 0 else 0)

        if self.n_calls % self.check_freq == 0:
            if len(self._wins) >= 500:
                recent = np.mean(self._wins[-500:])
                if recent >= self.target:
                    if self.verbose:
                        print(f"[EarlyStop] 勝率 {recent:.1%} ≥ 目標 {self.target:.1%}，停止訓練")
                    return False  # 停止

        return True
