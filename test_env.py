import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action

class RandomAgent:
    def get_action(self, obs):
        valid_actions = np.argwhere(obs[indices.VALID_ACTIONS] == 1).flatten()
        chosen = PlayerAction(np.random.choice(valid_actions))
        # 新版中 BET 涵盖了所有下注/加注行为
        if chosen == PlayerAction.BET:
            bet = np.random.uniform(
                obs[indices.VALID_BET_LOW],
                obs[indices.VALID_BET_HIGH]
            )
        else:
            bet = 0
        return Action(chosen, bet)

env = Table(n_players=6)
agents = [RandomAgent() for _ in range(6)]

obs = env.reset()
while True:
    acting_seat = env.current_player_i
    action = agents[acting_seat].get_action(obs)
    obs, rewards, done, _ = env.step(action)
    if done:
        print("一局结束，奖励：", rewards)
        break