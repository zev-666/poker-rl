import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
from treys import Card, Deck
from hand_evaluator import HandEvaluator
from info_set_abstractor import InfoSetAbstractor

class BlueprintNet(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 3)

    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.input_layer(x)
        for block in self.blocks:
            identity = out
            out = block(out)
            out = out + identity
        return self.output_layer(out)

class PokerDataset(Dataset):
    def __init__(self, strategy_dict, evaluator, abstractor, num_samples=10000):
        self.samples = []
        self.evaluator = evaluator
        self.abstractor = abstractor
        self._generate(strategy_dict, num_samples)

    def _encode_features(self, round_idx, board, hole, bet_history, pot, stack):
        features = []
        round_oh = [0]*4
        round_oh[min(round_idx,3)] = 1
        features.extend(round_oh)
        if bet_history and pot > 0:
            features.append(bet_history[-1] / pot)
        else:
            features.append(0.0)
        features.append(stack / max(pot, 1))
        if board and len(board) >= 3:
            score = self.evaluator.evaluate(hole, board)
            features.append(score / 7462.0)
        else:
            features.append(0.0)
        if board and len(board) >= 3:
            texture = self.abstractor.classify_board(board[:3]).value
            features.append(texture / 10.0)
        else:
            features.append(0.0)
        while len(features) < 9:
            features.append(0.0)
        return np.array(features[:9], dtype=np.float32)

    def _generate(self, strategy_dict, num_samples):
        deck = Deck()
        for _ in range(num_samples):
            deck.shuffle()
            hole_ints = deck.draw(2)
            hole = [Card.int_to_str(c) for c in hole_ints]     # ← 改用 int_to_str
            board = []
            for _ in range(5):
                card_int = deck.draw(1)[0]
                board.append(Card.int_to_str(card_int))        # ← 改用 int_to_str
            round_idx = np.random.randint(0, 4)
            bet_history = [np.random.choice([0, 20, 40, 80])]
            pot = 100
            stack = np.random.randint(50, 200)
            info_set = self.abstractor.encode(round_idx, bet_history, board[:3], pot, stack)
            strategy = strategy_dict.get(info_set, np.ones(3)/3)
            x = self._encode_features(round_idx, board, hole, bet_history, pot, stack)
            self.samples.append((x, strategy.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_blueprint(strategy_path="strategy.pkl", model_path="blueprint_net.pth", epochs=10, batch_size=64):
    if not os.path.exists(strategy_path):
        print(f"找不到 {strategy_path}，請先執行 mccfr_solver.py 產生策略。")
        return
    with open(strategy_path, 'rb') as f:
        data = pickle.load(f)
    strategy_sum = data['strategy_sum']
    strategy_dict = {}
    for info_set, sums in strategy_sum.items():
        total = np.sum(sums)
        if total > 0:
            strategy_dict[info_set] = sums / total
        else:
            strategy_dict[info_set] = np.ones(3)/3
    evaluator = HandEvaluator()
    abstractor = InfoSetAbstractor()
    dataset = PokerDataset(strategy_dict, evaluator, abstractor, num_samples=20000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BlueprintNet(input_dim=9, hidden_dim=64, num_blocks=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    print("開始訓練 BlueprintNet...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader):.6f}")
    torch.save(model.state_dict(), model_path)
    print(f"模型已儲存至 {model_path}")

if __name__ == "__main__":
    train_blueprint("strategy.pkl", "blueprint_net.pth", epochs=10, batch_size=64)
