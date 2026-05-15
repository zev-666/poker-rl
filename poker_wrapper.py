import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action
import pokerenv.obs_indices as indices

OBS_DIM    = 72
NUM_PLAYERS = 6
BIG_BLIND  = 5
STACK_SIZE = 200

class PokerGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_policy="random"):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.opponent_policy = opponent_policy
        self.table = Table(n_players=NUM_PLAYERS)
        self.trained_seat = 0
        self.other_seats = [1,2,3,4,5]
        self.done = False
        self.final_reward = 0.0  # 累积到游戏结束的奖励

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.table.reset()
        self.done = False
        self.final_reward = 0.0
        self._skip_to_our_turn()
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int):
        # 1. 训练智能体行动
        env_action = self._decode_action(action)
        _, reward_all, done_agent, _ = self.table.step(env_action)
        self.done = done_agent
        # 用 step 返回的奖励，不用 get_reward()
        self.final_reward = float(reward_all[self.trained_seat])

        # 2. 如果未结束，让对手行动，同时捕获对手回合的结束奖励
        if not self.done:
            self.final_reward += self._run_opponents()

        # 3. 最终奖励：游戏结束时是累积的 reward，否则为 0
        if self.done:
            reward = self.final_reward
        else:
            reward = 0.0

        obs = self._encode_obs()
        shaped_reward = self._shape_reward(reward, action, self.done)
        terminated = self.done
        truncated = False
        return obs, shaped_reward, terminated, truncated, {}

    def _encode_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        player = self.table.players[self.trained_seat]
        pot = self.table.pot
        call_amount = self.table.bet_to_match
        my_stack = player.stack
        street = self.table.street.value

        obs[0] = self._estimate_hand_strength(player.cards, self.table.cards)
        idx = 1

        for card in self.table.cards:
            if isinstance(card, int):
                rank = card % 13
                suit = card // 13
                card_id = suit * 13 + rank
                if 0 <= card_id < 52:
                    obs[idx + card_id] = 1.0
        idx += 52

        obs[53] = min(pot / (BIG_BLIND * 200), 1.0)
        obs[54] = min(my_stack / STACK_SIZE, 1.0)
        obs[55] = min(call_amount / (BIG_BLIND * 50), 1.0)
        idx = 56

        obs[56 + min(self.trained_seat, 5)] = 1.0
        idx = 62

        obs[62 + min(street, 3)] = 1.0
        idx = 66

        for i in range(NUM_PLAYERS):
            if i != self.trained_seat:
                last_bet = self.table.players[i].bet_this_street
                obs[idx + i] = min(last_bet / (BIG_BLIND * 20), 1.0)
        return obs

    def _decode_action(self, action: int) -> Action:
        player = self.table.players[self.trained_seat]
        valid = self.table._get_valid_actions(player)
        valid_actions = valid['actions_list']
        bet_range = valid['bet_range']

        action = int(action)

        if action == 0:  # fold
            act = PlayerAction.FOLD if PlayerAction.FOLD in valid_actions else PlayerAction.CHECK
            return Action(act, 0)
        elif action == 1:  # check/call
            if PlayerAction.CHECK in valid_actions:
                return Action(PlayerAction.CHECK, 0)
            else:
                return Action(PlayerAction.CALL, 0)
        elif action == 2:  # small raise
            amount = min(bet_range[1], max(bet_range[0], self.table.pot * 0.5))
            return Action(PlayerAction.BET, amount)
        elif action == 3:  # big raise
            amount = min(bet_range[1], max(bet_range[0], self.table.pot * 1.0))
            return Action(PlayerAction.BET, amount)
        elif action == 4:  # all-in
            return Action(PlayerAction.BET, player.stack)
        else:
            return Action(PlayerAction.CHECK, 0)

    def _skip_to_our_turn(self):
        while self.table.current_player_i != self.trained_seat and not self.done:
            self._random_opponent_step()

    def _run_opponents(self) -> float:
        """让对手行动直到轮到我方或游戏结束，返回对手回合中产生的我方奖励"""
        opponent_reward = 0.0
        while self.table.current_player_i != self.trained_seat and not self.done:
            _, r_all, self.done, _ = self._random_opponent_step()
            if r_all is not None:
                opponent_reward += float(r_all[self.trained_seat])
        return opponent_reward

    def _random_opponent_step(self):
        seat = self.table.current_player_i
        valid = self.table._get_valid_actions(self.table.players[seat])
        actions = valid['actions_list']
        chosen = np.random.choice(actions)
        chosen_action = PlayerAction(int(chosen))
        if chosen_action == PlayerAction.BET:
            low, high = valid['bet_range']
            bet = np.random.uniform(low, high)
        else:
            bet = 0.0
        return self.table.step(Action(chosen_action, bet))

    def _shape_reward(self, raw_reward, action, done) -> float:
        shaped = raw_reward / 100.0
        if action == 0 and not done:
            if self.table.bet_to_match == 0:
                shaped -= 0.05
        return float(shaped)

    def _estimate_hand_strength(self, hole_cards, community_cards) -> float:
        return 0.5
