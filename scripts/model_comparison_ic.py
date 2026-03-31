import subprocess
import os

# 实验配置
models = ["layer0", "layer0_layer2", "full"]
epochs = 20
seed = 42

def run_cmd(cmd):
    print(f"\n>>> Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

for model in models:
    print(f"\n{'='*60}")
    print(f" EXPERIMENT: Model={model} | Epochs={epochs} | Loss=IC")
    print(f"{'='*60}")
    
    # 1. 训练 (使用 IC Loss)
    train_cmd = f"./.venv/Scripts/python train.py --model_arch {model} --epochs {epochs} --seed {seed} --loss_type ic"
    run_cmd(train_cmd)
    
    # 2. 评估
    pred_file = f"outputs/pred_{model}.csv"
    if os.path.exists(pred_file):
        eval_cmd = f"./.venv/Scripts/python evaluate.py --pred_path {pred_file}"
        run_cmd(eval_cmd)
    else:
        print(f"Error: Prediction file {pred_file} not found!")

print("\n" + "="*60)
print("  All Experiments Completed!")
print("="*60)
