import subprocess
import pandas as pd
import os

seeds = [42, 123, 2026]
summary = []

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

for seed in seeds:
    print(f"\n{'='*40}")
    print(f" TESTING SEED: {seed}")
    print(f"{'='*40}")
    
    # 1. Train
    run_cmd(f"./.venv/Scripts/python train.py --model_arch layer0 --epochs 3 --seed {seed}")
    
    # 2. Negate Predictions
    pred_file = "outputs/pred_layer0.csv"
    neg_file = f"outputs/neg_pred_seed_{seed}.csv"
    df = pd.read_csv(pred_file)
    df["pred"] = -df["pred"]
    df.to_csv(neg_file, index=False)
    print(f"Generated negated predictions: {neg_file}")
    
    # 3. Evaluate and Capture Output
    # 这里我们直接手动调用 evaluate 逻辑或者简单获取 Sharpe
    # 为了简单起见，我们直接运行 evaluate.py 并在日志中查看结果
    run_cmd(f"./.venv/Scripts/python evaluate.py --pred_path {neg_file}")

print("\n>>> Robustness Check Completed.")
