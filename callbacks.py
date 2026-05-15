cat > ~/poker-project/callbacks.py << 'EOF'
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

def get_callbacks(save_dir="./checkpoints", eval_env=None):
    """
    返回训练中常用的回调列表。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_poker",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    callbacks = [checkpoint_callback]
    
    # 如果有评估环境，添加评估回调
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_dir, "best_model"),
            log_path=os.path.join(save_dir, "eval_log"),
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    return callbacks
EOF