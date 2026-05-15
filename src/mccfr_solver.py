import numpy as np
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
from treys import Card, Deck
from hand_evaluator import HandEvaluator
from info_set_abstractor import InfoSetAbstractor

class MCCFRSolver:
    def __init__(self, num_iterations=10000, discount_interval=1000, pruning_threshold=0.01):
        self.num_iterations = num_iterations
        self.discount_interval = discount_interval
        self.pruning_threshold = pruning_threshold
        self.evaluator = HandEvaluator()
        self.abstractor = InfoSetAbstractor()
        
        self.regret_sum = defaultdict(lambda: np.zeros(3))  # fold, call, raise
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.deck = Deck()
        
    def get_info_set(self, round_idx, board, hole, bet_history, pot, stack):
        board_cards = [Card.new(c) for c in board] if board else []
        return self.abstractor.encode(round_idx, bet_history, board_cards, pot, stack)
    
    def get_strategy(self, info_set, reach_prob=1.0):
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        total = np.sum(positive_regrets)
        if total > 0:
            strategy = positive_regrets / total
        else:
            strategy = np.ones(3) / 3.0
        self.strategy_sum[info_set] += reach_prob * strategy
        return strategy
    
    def cfr(self, round_idx, board, hole, bet_history, pot, stack, p0, p1):
        if round_idx > 3 or stack <= 0:
            return 0
        
        info_set = self.get_info_set(round_idx, board, hole, bet_history, pot, stack)
        strategy = self.get_strategy(info_set, p0)
        
        actions = [0, 1, 2]
        node_value = 0
        action_values = np.zeros(3)
        
        for a in actions:
            if a == 0:  # fold
                reward = -pot * 0.5 if len(bet_history) % 2 == 0 else pot * 0.5
                action_values[a] = reward
            elif a == 1:  # call
                action_values[a] = 0
            else:  # raise
                action_values[a] = 0.1 * pot
            
            node_value += strategy[a] * action_values[a]
        
        for a in actions:
            regret = action_values[a] - node_value
            self.regret_sum[info_set][a] += p1 * regret
        
        return node_value
    
    def train(self):
        print(f"開始 MCCFR 訓練，共 {self.num_iterations} 次迭代")
        for it in tqdm(range(1, self.num_iterations + 1)):
            deck = Deck()
            hole = deck.draw(2)
            board = []
            for _ in range(5):
                board.extend(deck.draw(1))
            
            self.cfr(0, board, hole, [20, 20], 40, 100, 1.0, 1.0)
            
            if it % self.discount_interval == 0:
                for k in self.regret_sum:
                    self.regret_sum[k] *= (it - self.discount_interval) / it
        
        print("訓練完成，儲存策略模型...")
        self.save_strategy("strategy.pkl")
    
    def save_strategy(self, path):
        model_data = {
            'strategy_sum': dict(self.strategy_sum),
            'regret_sum': dict(self.regret_sum),
            'abstractor': self.abstractor,
            'num_iterations': self.num_iterations
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"策略已儲存至 {path}")

if __name__ == "__main__":
    solver = MCCFRSolver(num_iterations=1000, discount_interval=200, pruning_threshold=0.05)
    solver.train()
