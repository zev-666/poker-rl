# poker_wrapper.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 觀測空間設計（共 72 維）────────────────────────────────
# 維度分配：
#   手牌強度估算        :  1 維  (0.0~1.0，preflop 勝率)
#   公共牌 one-hot      : 52 維  (5張牌疊加，有牌=1 無牌=0)
#   底池大小            :  1 維  (歸一化，除以大盲×200)
#   自身籌碼            :  1 維  (歸一化)
#   當前需跟注金額      :  1 維  (歸一化)
#   位置 one-hot        :  6 維  (seat 0~5)
#   下注輪次 one-hot    :  4 維  (preflop/flop/turn/river)
#   對手最後動作        :  6 維  (各對手上輪動作，折/跟/加)
# 合計 = 1+52+1+1+1+6+4+6 = 72 ✓

OBS_DIM    = 72
NUM_PLAYERS = 6
BIG_BLIND  = 20     # 你的環境大盲金額，按實際調整
STACK_SIZE = 1000   # 初始籌碼


class PokerGymWrapper(gym.Env):
    """
    將 Poker-Env 1.0.1 包裝成 SB3 相容的單 agent Gym 環境。
    訓練策略：控制 seat 0 的 agent，其餘 5 名玩家使用可配置對手策略。
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_policy="random", render_mode=None):
        super().__init__()

        # Gym 必要屬性
        self.observation_space = spaces.Box(
            low  = -1.0,
            high =  1.0,
            shape= (OBS_DIM,),
            dtype= np.float32,
        )

        # 離散動作空間：5 個
        # 0=fold  1=check/call  2=raise_small(0.5pot)
        # 3=raise_big(1.0pot)   4=all_in
        self.action_space = spaces.Discrete(5)

        self.opponent_policy = opponent_policy

        # 延遲引入，避免 import 時就崩潰
        self._init_poker_env()

    def _init_poker_env(self):
        """初始化底層 Poker-Env，按你的 API 調整"""
        try:
            from poker_env.envs import TexasHoldemEnv   # 按實際模組路徑改
            self.env = TexasHoldemEnv(n_player=NUM_PLAYERS)
        except ImportError as e:
            raise ImportError(
                f"找不到 poker_env：{e}\n"
                "確認 Poker-Env 1.0.1 已正確安裝"
            )

    # ── reset ──────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置底層環境
        raw_obs = self.env.reset()

        # 快速跳過前幾個不是 seat 0 的行動輪
        # （Poker-Env 會讓所有玩家依序行動）
        raw_obs = self._skip_to_our_turn(raw_obs)

        obs = self._encode_obs(raw_obs)
        self._episode_reward = 0.0
        return obs, {}

    # ── step ───────────────────────────────────────────────────
    def step(self, action: int):
        # 把離散動作轉換成環境接受的格式
        env_action = self._decode_action(action)

        raw_obs, reward, done, info = self.env.step(env_action)

        # 回合未結束：讓對手行動直到輪到我們或回合結束
        if not done:
            raw_obs, opponent_reward, done, info = self._run_opponents(raw_obs)
            reward += opponent_reward   # 通常為 0，結算時才有非零獎勵

        obs = self._encode_obs(raw_obs)

        # ── Reward Shaping ──────────────────────────────────────
        # 撲克的稀疏獎勵問題：整局只有最後才有獎勵
        # 這裡加入輕微的中間獎勵，讓訓練更穩定
        shaped_reward = self._shape_reward(reward, action, done, info)

        self._episode_reward += shaped_reward

        # Gymnasium 要求 terminated 和 truncated 分開
        terminated = done
        truncated  = False

        return obs, shaped_reward, terminated, truncated, info

    # ── 觀測編碼 ────────────────────────────────────────────────
    def _encode_obs(self, raw_obs) -> np.ndarray:
        """
        將 Poker-Env 原始觀測轉換為 72 維 float32 向量。
        raw_obs 的結構按你的 Poker-Env 版本調整。
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # [0] 手牌強度估算（0~1）
        # 用簡化版 preflop 勝率表，真實實作可用 treys 庫計算
        obs[idx] = self._estimate_hand_strength(raw_obs)
        idx += 1

        # [1:53] 公共牌 one-hot（52 張牌）
        community_cards = getattr(raw_obs, "community_cards", [])
        for card in community_cards:
            if card is not None and hasattr(card, "to_int"):
                card_idx = card.to_int()  # 0~51
                if 0 <= card_idx < 52:
                    obs[idx + card_idx] = 1.0
        idx += 52

        # [53] 底池大小（歸一化到 0~1）
        pot = getattr(raw_obs, "pot", 0)
        obs[idx] = min(pot / (BIG_BLIND * 200), 1.0)
        idx += 1

        # [54] 自身籌碼（歸一化）
        my_stack = getattr(raw_obs, "stacks", [STACK_SIZE] * NUM_PLAYERS)
        if isinstance(my_stack, (list, np.ndarray)):
            my_stack = my_stack[0]
        obs[idx] = min(my_stack / STACK_SIZE, 1.0)
        idx += 1

        # [55] 需跟注金額（歸一化）
        call_amount = getattr(raw_obs, "call_amount", 0)
        obs[idx] = min(call_amount / (BIG_BLIND * 50), 1.0)
        idx += 1

        # [56:62] 位置 one-hot（6人桌）
        my_seat = getattr(raw_obs, "current_player", 0)
        obs[idx + min(my_seat, 5)] = 1.0
        idx += NUM_PLAYERS

        # [62:66] 下注輪次 one-hot
        street_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        street = getattr(raw_obs, "street", "preflop")
        street_idx = street_map.get(str(street).lower(), 0)
        obs[idx + street_idx] = 1.0
        idx += 4

        # [66:72] 對手最後動作（6 個位置）
        last_actions = getattr(raw_obs, "last_actions", [0] * NUM_PLAYERS)
        for i, act in enumerate(last_actions[:NUM_PLAYERS]):
            obs[idx + i] = float(act) / 4.0   # 歸一化到 0~1
        idx += NUM_PLAYERS

        assert idx == OBS_DIM, f"觀測維度錯誤：{idx} ≠ {OBS_DIM}"
        return obs

    # ── 動作解碼 ────────────────────────────────────────────────
    def _decode_action(self, action: int) -> dict:
        """
        將 0~4 的離散動作轉為 Poker-Env 接受的格式。
        按你的環境 API 調整。
        """
        pot = getattr(getattr(self.env, "state", None), "pot", BIG_BLIND * 10)

        action_map = {
            0: {"action": "fold",  "amount": 0},
            1: {"action": "call",  "amount": 0},         # call 金額環境自動計算
            2: {"action": "raise", "amount": int(pot * 0.5)},   # 0.5 pot
            3: {"action": "raise", "amount": int(pot * 1.0)},   # 1.0 pot
            4: {"action": "raise", "amount": STACK_SIZE},        # all-in
        }
        return action_map.get(action, action_map[1])

    # ── 對手行動 ────────────────────────────────────────────────
    def _run_opponents(self, raw_obs):
        """讓 seat 1~5 的對手按策略行動，直到輪到 seat 0 或局結束"""
        total_reward = 0.0
        done = False

        max_steps = 20  # 防止無限迴圈
        for _ in range(max_steps):
            current = getattr(raw_obs, "current_player", 0)
            if current == 0 or done:
                break

            if self.opponent_policy == "random":
                opp_action = self._random_opponent_action()
            elif self.opponent_policy == "call_always":
                opp_action = {"action": "call", "amount": 0}
            else:
                opp_action = self._random_opponent_action()

            raw_obs, reward, done, info = self.env.step(opp_action)
            total_reward += reward

        return raw_obs, total_reward, done, {}

    def _skip_to_our_turn(self, raw_obs):
        """reset 後跳過前面不是 seat 0 的行動（preflop 時莊家先行動）"""
        raw_obs, _, done, info = self._run_opponents(raw_obs)
        return raw_obs

    def _random_opponent_action(self) -> dict:
        """隨機對手：70% call，20% fold，10% raise"""
        r = self.np_random.random() if hasattr(self, "np_random") else np.random.random()
        if r < 0.70:
            return {"action": "call",  "amount": 0}
        elif r < 0.90:
            return {"action": "fold",  "amount": 0}
        else:
            return {"action": "raise", "amount": BIG_BLIND * 3}

    # ── Reward Shaping ──────────────────────────────────────────
    def _shape_reward(self, raw_reward, action, done, info) -> float:
        """
        撲克稀疏獎勵問題的解法。
        raw_reward 通常只在 done=True 時非零（輸贏籌碼差）。
        """
        shaped = raw_reward / STACK_SIZE  # 歸一化到 -1~+1

        # 懲罰：在應該 check 的情況下無腦 fold（preflop 大盲可免費看）
        # 這需要你根據環境狀態判斷，這裡給個示意
        if action == 0 and not done:  # fold 但局還沒結束
            call_amount = getattr(
                getattr(self.env, "state", None), "call_amount", 1
            )
            if call_amount == 0:  # 可以免費 check 卻選擇 fold
                shaped -= 0.05

        # 輕微鼓勵 all-in 用強牌
        # （避免 AI 永遠只 call，永不加注）
        # 真實實作應依據手牌強度決定，這裡先略去

        return float(shaped)

    def _estimate_hand_strength(self, raw_obs) -> float:
        """
        簡化版手牌強度估算，回傳 0.0~1.0。
        preflop 用查表，flop 以後用 Monte Carlo 抽樣（需要 treys 庫）。
        """
        # 暫用隨機值佔位，後續換成真實計算
        return float(np.random.random() * 0.5 + 0.25)