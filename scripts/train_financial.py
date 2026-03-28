import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
import numpy as np

# 添加项目根目录
sys.path.append(os.getcwd())

from model.config import AntConfig
from model.ant import AntTransformer
from data.data_prep import prepare_data
from data.financial_dataset import FinancialDataset

def train_one_epoch(model, loader, optimizer, criterion, device, gate_lambda):
    model.train()
    total_loss = 0
    total_gate_loss = 0
    
    gate_sums = [0.0] * model.config.num_layers
    batch_count = 0
    
    for batch in tqdm(loader, desc="  train", leave=False):
        x = batch['x'].to(device) # [B, T, D]
        y = batch['y'].to(device) # [B]
        
        optimizer.zero_grad()
        # Financial mode: input_ids is actually the feature tensor
        logits, _, all_gates = model(x) 
        
        # logits shape is [B, 1], y is [B]
        task_loss = criterion(logits.squeeze(-1), y)
        
        # Gate L1 Regularization
        gate_loss = torch.stack([g.abs().mean() for g in all_gates]).mean()
        
        loss = task_loss + gate_lambda * gate_loss
        loss.backward()
        
        optimizer.step()
        
        total_loss += task_loss.item() * x.size(0)
        total_gate_loss += gate_loss.item() * x.size(0)
        
        # 取 token 维度均值（financial 模式无 CLS，直接对 T 维平均）
        for i, g in enumerate(all_gates):
            gate_sums[i] += g.mean(dim=(1, 2)).mean().item()
        batch_count += 1
        
    avg_gates = [s / batch_count for s in gate_sums]
    return total_loss / len(loader.dataset), total_gate_loss / len(loader.dataset), avg_gates

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    for batch in tqdm(loader, desc="  val", leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        logits, _, _ = model(x)
        loss = criterion(logits.squeeze(-1), y)
        total_loss += loss.item() * x.size(0)
        preds.append(logits.squeeze(-1).cpu())
        targets.append(y.cpu())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = total_loss / len(loader.dataset)
    # 简单计算一个相关系数作为评测
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
    return mse, corr

def main():
    # 1. 加载并清理数据
    train_df, val_df, test_df, features, target = prepare_data()
    
    config = AntConfig(
        model_type="financial",
        input_dim=len(features), # 自动适配特征数
        num_classes=1,           # Regression
        max_seq_len=6,
        d_model=64,
        num_layers=4,
        batch_size=512,          # 增大以适应全量数据
        epochs=5,
        lr=1e-3,
        gate_lambda=0.08         # Slightly higher to see divergence faster
    )
    
    # 执行配置校验
    config.validate()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 创建 Dataset/Loader
    train_ds = FinancialDataset(train_df, features, target, seq_len=config.max_seq_len)
    val_ds   = FinancialDataset(val_df, features, target, seq_len=config.max_seq_len)
    
    print(f"Dataset created: Train={len(train_ds)} samples, Val={len(val_ds)} samples")
    
    if len(val_ds) == 0:
        print("Warning: Validation dataset is empty! Check your seq_len and date range.")
        return
        
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    # 3. 初始化模型
    model = AntTransformer(config).to(device)
    print(f"Model: AntTransformer (Financial Mode)")
    print(f"Parameters: {model.count_parameters():,}\n")
    
    optimizer = AdamW(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    
    # 4. 训练
    for epoch in range(1, config.epochs + 1):
        t_loss, g_loss, avg_gates = train_one_epoch(model, train_loader, optimizer, criterion, device, config.gate_lambda)
        v_mse, v_corr = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d} | Train MSE: {t_loss:.6f} | Gate Loss: {g_loss:.4f} | Val MSE: {v_mse:.6f} | Val Corr: {v_corr:.4f}")
        print("  Gate Values:", " | ".join([f"L{i}: {g:.4f}" for i, g in enumerate(avg_gates)]))

if __name__ == "__main__":
    main()
