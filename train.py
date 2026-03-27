"""
train.py — 小蚂蚁训练入口

用法：
    python train.py             # 使用默认 config
    python train.py --epochs 5   # 覆盖部分超参
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from config import AntConfig
from model.ant import AntTransformer
from data.dataset import get_dataloaders


# ─────────────────────────────────────────────────────────────
# 单 epoch 训练
# ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model: AntTransformer,
    loader,
    optimizer,
    scheduler,
    criterion,
    device: torch.device,
    max_grad_norm: float,
    gate_lambda: float = 0.0,
) -> tuple[float, float, float, list[float]]:
    model.train()
    total_loss = total_correct = total_samples = total_gate_loss = 0
    # 用于记录每层 gate 的累加值和数量
    gate_sums = [0.0] * model.config.num_layers
    batch_count = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _, all_gates = model(input_ids, attention_mask)
        
        # ① 原生任务 Loss
        task_loss = criterion(logits, labels)
        
        # ② 门控正则 Loss (L1 鼓励稀疏，即 gate -> 0，跳过层)
        gate_loss = torch.stack([g.abs().mean() for g in all_gates]).mean()
        
        loss = task_loss + gate_lambda * gate_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        bs = labels.size(0)
        total_loss      += loss.item() * bs
        total_gate_loss += gate_loss.item() * bs
        total_correct   += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples   += bs

        # 累加 gate 值
        for i, g in enumerate(all_gates):
            gate_sums[i] += g.mean().item()
        batch_count += 1

    avg_gates = [s / batch_count for s in gate_sums]
    return total_loss / total_samples, total_gate_loss / total_samples, total_correct / total_samples, avg_gates


# ─────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: AntTransformer,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_samples = 0

    for batch in tqdm(loader, desc="  eval ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits, _, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main(config: AntConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 数据
    train_loader, val_loader = get_dataloaders(config)

    # 模型
    model = AntTransformer(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Layers: {config.num_layers}")
    print(f"d_model: {config.d_model}\n")

    # 优化器 + 学习率调度
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * config.epochs
    pct_start = min(config.warmup_steps / total_steps, 0.99) if total_steps > 0 else 0.3
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch:02d}/{config.epochs}")

        train_loss, train_gate_loss, train_acc, avg_gates = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, config.max_grad_norm, config.gate_lambda
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"  Train loss={train_loss:.4f} (gate={train_gate_loss:.4f}) acc={train_acc:.4f}\n"
            f"  Val   loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # 输出每层 Gate 平均值
        print("  Gate Monitoring (Average per Layer):")
        for i, g in enumerate(avg_gates):
            print(f"    Layer {i+1}: {g:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"  [OK] Saved best (val_acc={val_acc:.4f})")
        print()

    print(f"Training done. Best val acc: {best_val_acc:.4f}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AntTransformer")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--d_model",    type=int,   default=None)
    parser.add_argument("--num_layers", type=int,   default=None)
    parser.add_argument("--use_dummy_data", action="store_true")
    args = parser.parse_args()

    cfg = AntConfig()
    # 覆盖指定参数
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    main(cfg)
