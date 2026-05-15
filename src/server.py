from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import pickle
from hand_evaluator import HandEvaluator
from info_set_abstractor import InfoSetAbstractor
from blueprint_net import BlueprintNet

app = FastAPI(title="Texas Hold'em AI Decision Server")

# ---------- 載入模型 ----------
MODEL_PATH = "blueprint_net.pth"
STRATEGY_PATH = "strategy.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlueprintNet(input_dim=9, hidden_dim=64, num_blocks=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 載入輔助工具
evaluator = HandEvaluator()
abstractor = InfoSetAbstractor()

# 載入策略表（可選，供評估用）
try:
    with open(STRATEGY_PATH, 'rb') as f:
        strategy_data = pickle.load(f)
except:
    strategy_data = None

# ---------- 請求 / 回應格式 ----------
class GameState(BaseModel):
    round_idx: int               # 0=preflop,1=flop,2=turn,3=river
    hole_cards: List[str]        # e.g. ["Ah","Kh"]
    community_cards: List[str]   # e.g. ["Qh","Jh","Th","2c","3d"]
    bet_history: List[float]     # 過去下注金額
    pot_size: float
    stack_size: float

class DecisionResponse(BaseModel):
    action: str                  # "fold", "call", "raise"
    confidence: float
    action_values: List[float]   # [fold_value, call_value, raise_value]

# ---------- 特徵工程（與訓練時一致） ----------
def extract_features(gs: GameState):
    features = []
    # 回合 one-hot (4)
    round_oh = [0]*4
    round_oh[min(gs.round_idx, 3)] = 1
    features.extend(round_oh)
    
    # 下注比例
    if gs.bet_history and gs.pot_size > 0:
        features.append(gs.bet_history[-1] / gs.pot_size)
    else:
        features.append(0.0)
    
    # 籌碼深度
    features.append(gs.stack_size / max(gs.pot_size, 1))
    
    # 手牌強度歸一化
    if len(gs.community_cards) >= 3:
        score = evaluator.evaluate(gs.hole_cards, gs.community_cards)
        features.append(score / 7462.0)
    else:
        features.append(0.0)
    
    # 公共牌紋理歸一化
    if len(gs.community_cards) >= 3:
        texture = abstractor.classify_board(gs.community_cards[:3]).value
        features.append(texture / 10.0)
    else:
        features.append(0.0)
    
    # 補到 9 維
    while len(features) < 9:
        features.append(0.0)
    return np.array(features[:9], dtype=np.float32)

# ---------- 決策端點 ----------
@app.post("/decide", response_model=DecisionResponse)
def decide(gs: GameState):
    try:
        x = torch.from_numpy(extract_features(gs)).unsqueeze(0).to(device)
        with torch.no_grad():
            values = model(x).cpu().numpy().flatten()  # [fold, call, raise]
        
        # 簡單規則：最大值的動作；若 call 和 raise 接近時可自行調整策略
        action_idx = int(np.argmax(values))
        action_map = {0: "fold", 1: "call", 2: "raise"}
        action = action_map[action_idx]
        
        # 信心值 = softmax 最大值
        exp_vals = np.exp(values - np.max(values))
        probs = exp_vals / exp_vals.sum()
        confidence = float(probs[action_idx])
        
        return DecisionResponse(
            action=action,
            confidence=confidence,
            action_values=values.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- 健康檢查 ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}
