from fastapi import FastAPI, BackgroundTasks
from db import init_db, save_decision
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import pickle
from hand_evaluator import HandEvaluator
from info_set_abstractor import InfoSetAbstractor
from blueprint_net import BlueprintNet

app = FastAPI(title="Texas Hold'em AI Decision Server")

@app.on_event("startup")
async def startup_event():
    await init_db()

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

with open(STRATEGY_PATH, "rb") as f:
    strategy = pickle.load(f)

# ---------- 請求/回應格式 ----------
class DecisionRequest(BaseModel):
    round_idx: int
    hole_cards: List[str]
    community_cards: List[str]
    bet_history: List[int]
    pot_size: int
    stack_size: int

class DecisionResponse(BaseModel):
    action: str

# ---------- 決策端點 ----------
@app.post("/decide", response_model=DecisionResponse)
async def decide(request: DecisionRequest, background_tasks: BackgroundTasks):
    hole = request.hole_cards
    community = request.community_cards
    history = request.bet_history
    pot = request.pot_size
    stack = request.stack_size

    # 使用 encode 產生資訊集
    info_set = abstractor.encode(
        game_round=request.round_idx,
        bet_history=history,
        board_cards=community,
        pot_size=pot,
        stack_size=stack
    )

    # 查詢策略表
    if info_set in strategy:
        probs = strategy[info_set]
    else:
        # fallback: 使用神經網路
        features = np.array([len(community), pot / (pot + stack), len(history)], dtype=np.float32)
        features = np.concatenate([features, np.zeros(6, dtype=np.float32)])  # 補齊 9 維
        tensor = torch.tensor(features, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor).softmax(dim=-1).cpu().numpy().flatten()
        probs = {"fold": float(out[0]), "call": float(out[1]), "raise": float(out[2])}

    # 選擇最大機率動作
    action = max(probs, key=probs.get)

    # ----- 背景寫入資料庫 -----
    log_data = {
        "round_idx": request.round_idx,
        "hole_cards": hole,
        "community_cards": community,
        "bet_history": history,
        "pot_size": pot,
        "stack_size": stack,
        "action_taken": action,
        "strategy_used": "mccfr_blueprint"
    }
    background_tasks.add_task(save_decision, log_data)

    return DecisionResponse(action=action)
