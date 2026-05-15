import numpy as np
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action

# 模拟你的训练环境：智能体控制座位0，其余随机
table = Table(n_players=6)
trained_seat = 0
total_rewards = []

for ep in range(20):
    table.reset()
    done = False
    while not done:
        seat = table.current_player_i
        if seat == trained_seat:
            # 智能体随机动作（0-4），按你的动作空间
            action_int = np.random.randint(0, 5)
            # 解码动作，复用你的逻辑（简化版，只做基本映射）
            valid = table._get_valid_actions(table.players[seat])
            valid_actions = valid['actions_list']
            bet_range = valid['bet_range']
            if action_int == 0:  # fold
                act = PlayerAction.FOLD if PlayerAction.FOLD in valid_actions else PlayerAction.CHECK
                bet = 0
            elif action_int == 1:  # check/call
                act = PlayerAction.CHECK if PlayerAction.CHECK in valid_actions else PlayerAction.CALL
                bet = 0
            elif action_int == 2:  # small raise
                act = PlayerAction.BET
                bet = min(bet_range[1], max(bet_range[0], table.pot * 0.5))
            elif action_int == 3:  # big raise
                act = PlayerAction.BET
                bet = min(bet_range[1], max(bet_range[0], table.pot * 1.0))
            elif action_int == 4:  # all-in
                act = PlayerAction.BET
                bet = table.players[seat].stack
            else:
                act = PlayerAction.CHECK
                bet = 0
            _, _, done, _ = table.step(Action(act, bet))
        else:
            # 对手随机
            valid = table._get_valid_actions(table.players[seat])
            actions = valid['actions_list']
            chosen = np.random.choice(actions)
            chosen_action = PlayerAction(int(chosen))
            if chosen_action == PlayerAction.BET:
                low, high = valid['bet_range']
                bet = np.random.uniform(low, high)
            else:
                bet = 0.0
            _, _, done, _ = table.step(Action(chosen_action, bet))
    
    # 游戏结束，获取座位0玩家的最终奖励（用get_reward）
    reward = table.players[trained_seat].get_reward()
    total_rewards.append(reward)
    print(f"局 {ep+1}: 奖励 = {reward:.2f}")

print("\n总览:", total_rewards)
print("平均奖励:", np.mean(total_rewards))
print("非零奖励次数:", sum(1 for r in total_rewards if r != 0))
