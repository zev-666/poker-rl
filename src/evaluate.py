import sys
sys.path.insert(0, '/app/src')

import numpy as np
from stable_baselines3 import PPO
from poker_wrapper import PokerGymWrapper

def evaluate(model_path: str, n_episodes: int = 500, opponent: str = "random"):
    model = PPO.load(model_path)
    env   = PokerGymWrapper(opponent_policy=opponent)

    wins, rewards, action_hist = 0, [], np.zeros(5)
    action_labels = ["fold", "call", "raise_s", "raise_b", "all_in"]

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_hist[int(action)] += 1
            obs, reward, done, _, _ = env.step(int(action))

        rewards.append(reward)
        if reward > 0:
            wins += 1

    print(f"\n評估結果（{n_episodes} 局 vs {opponent}）")
    print(f"  勝率:       {wins/n_episodes:.1%}")
    print(f"  平均獎勵:   {np.mean(rewards):+.4f}")
    print(f"  最大獎勵:   {np.max(rewards):+.4f}")
    print(f"  最小獎勵:   {np.min(rewards):+.4f}")
    print(f"\n  動作分佈:")
    total = action_hist.sum()
    for label, count in zip(action_labels, action_hist):
        bar = "█" * int(count / total * 30)
        print(f"    {label:>9}: {bar} {count/total:.1%}")

if __name__ == "__main__":
    evaluate("./models/ppo_poker_cpu", n_episodes=500)
