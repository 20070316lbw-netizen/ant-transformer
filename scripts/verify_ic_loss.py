import subprocess
import pandas as pd
import os
import shutil
import sys

seeds = [42, 123, 2026]

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

for seed in seeds:
    print(f"\n{'='*40}")
    print(f" TESTING SEED: {seed} with IC Loss")
    print(f"{'='*40}")
    
    # 1. Train with IC Loss
    run_cmd([sys.executable, "train.py", "--model_arch", "layer0", "--epochs", "3", "--seed", str(seed), "--loss_type", "ic"])
    
    # 2. 保存并评估原始预测 (不再进行手动取反!)
    pred_file = "outputs/pred_layer0.csv"
    final_pred_file = f"outputs/pred_ic_loss_seed_{seed}.csv"
    
    if os.path.exists(pred_file):
        shutil.copy(pred_file, final_pred_file)
        print(f"Saved predictions to: {final_pred_file}")
        
    run_cmd([sys.executable, "evaluate.py", "--pred_path", final_pred_file])

print("\n>>> IC Loss Robustness Check Completed.")
