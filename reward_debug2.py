import numpy as np
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action

table = Table(n_players=6)
trained_seat = 0

for ep in range(3):
    table.reset()
    done = False
    step_count = 0
    while not done:
        seat = table.current_player_i
        # 统一用随机动作
        valid = table._get_valid_actions(table.players[seat])
        actions = valid['actions_list']
        chosen = np.random.choice(actions)
        chosen_action = PlayerAction(int(chosen))
        if chosen_action == PlayerAction.BET:
            low, high = valid['bet_range']
            bet = np.random.uniform(low, high)
        else:
            bet = 0.0
        _, rewards, done, _ = table.step(Action(chosen_action, bet))
        step_count += 1
        if done:
            print(f"第 {ep+1} 局结束, 步数: {step_count}")
            print(f"  done=True, rewards 数组: {rewards}")
            print(f"  玩家0的 reward: {rewards[trained_seat]}")
            print(f"  玩家0的 get_reward(): {table.players[trained_seat].get_reward()}")
            print(f"  玩家0的 winnings: {table.players[trained_seat].winnings}")
            print(f"  底池: {table.pot}")
            print("")
        if step_count > 100:  # 防止死循环
            print("步数超过100，强制退出")
            break
