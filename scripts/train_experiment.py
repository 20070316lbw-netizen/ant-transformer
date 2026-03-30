"""
train_experiment.py — 模型对比实验

比较三个模型:
1. 模型A: 仅 Layer0
2. 模型B: Layer0 + 新Layer (重新初始化)
3. 模型C: 原始4层 + gate

评估指标: IC, RankIC, Sharpe (按时间分组计算)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
import os
import random

sys.path.append(os.getcwd())

from model.config import AntConfig
from model.ant import AntTransformer
from model.encoder import AntEncoder
from model.layer import AntLayer
from data.data_prep import prepare_data
from data.financial_dataset import FinancialDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return {
        "x": torch.stack([torch.from_numpy(np.array(item["x"])) for item in batch]),
        "y": torch.stack([torch.tensor(item["y"]) for item in batch]),
        "date": [item["date"] for item in batch],
        "ticker": [item["ticker"] for item in batch],
    }


def create_model_a(config):
    """模型A: 仅 Layer0"""
    cfg = AntConfig(
        model_type="financial",
        input_dim=config.input_dim,
        num_classes=1,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        num_layers=1,
        num_heads=config.num_heads,
        cross_layer_heads=config.cross_layer_heads,
        d_ff=config.d_ff,
        gate_hidden_dim=config.gate_hidden_dim,
        dropout=config.dropout,
        use_grouped_freq_attention=config.use_grouped_freq_attention,
        num_head_groups=config.num_head_groups,
        group_mix_coeff=config.group_mix_coeff,
        batch_size=config.batch_size,
        lr=config.lr,
        epochs=config.epochs,
    )
    return AntTransformer(cfg)


def create_model_b(config):
    """模型B: Layer0 + Layer2（剪枝后保留两层）"""

    class Layer0Layer2Encoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layer0 = AntLayer(
                cfg.d_model,
                cfg.num_heads,
                cfg.cross_layer_heads,
                cfg.d_ff,
                cfg.gate_hidden_dim,
                cfg.dropout,
                cfg.use_grouped_freq_attention,
                cfg.num_head_groups,
                cfg.group_mix_coeff,
            )
            self.layer2 = AntLayer(
                cfg.d_model,
                cfg.num_heads,
                cfg.cross_layer_heads,
                cfg.d_ff,
                cfg.gate_hidden_dim,
                cfg.dropout,
                cfg.use_grouped_freq_attention,
                cfg.num_head_groups,
                cfg.group_mix_coeff,
            )

        def forward(self, x, key_padding_mask=None, enable_pruning=True):
            all_hiddens = []
            all_gates = []

            h, gate0 = self.layer0(
                x,
                all_hiddens,
                key_padding_mask=key_padding_mask,
                enable_pruning=enable_pruning,
            )
            all_hiddens.append(h)
            all_gates.append(gate0)

            h, gate2 = self.layer2(
                h,
                all_hiddens,
                key_padding_mask=key_padding_mask,
                enable_pruning=enable_pruning,
            )
            all_hiddens.append(h)
            all_gates.append(gate2)
            return h, all_hiddens, all_gates

    class TwoLayerTransformer(AntTransformer):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.encoder = Layer0Layer2Encoder(cfg)

    cfg = AntConfig(
        model_type="financial",
        input_dim=config.input_dim,
        num_classes=1,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        num_layers=2,
        num_heads=config.num_heads,
        cross_layer_heads=config.cross_layer_heads,
        d_ff=config.d_ff,
        gate_hidden_dim=config.gate_hidden_dim,
        dropout=config.dropout,
        use_grouped_freq_attention=config.use_grouped_freq_attention,
        num_head_groups=config.num_head_groups,
        group_mix_coeff=config.group_mix_coeff,
        batch_size=config.batch_size,
        lr=config.lr,
        epochs=config.epochs,
    )
    return TwoLayerTransformer(cfg)


def create_model_c(config):
    """模型C: 原始4层 + gate"""
    cfg = AntConfig(
        model_type="financial",
        input_dim=config.input_dim,
        num_classes=1,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        num_layers=4,
        num_heads=config.num_heads,
        cross_layer_heads=config.cross_layer_heads,
        d_ff=config.d_ff,
        gate_hidden_dim=config.gate_hidden_dim,
        dropout=config.dropout,
        use_grouped_freq_attention=config.use_grouped_freq_attention,
        num_head_groups=config.num_head_groups,
        group_mix_coeff=config.group_mix_coeff,
        batch_size=config.batch_size,
        lr=config.lr,
        epochs=config.epochs,
    )
    return AntTransformer(cfg)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, gate_lambda
):
    model.train()
    total_loss = 0
    gate_sums = [0.0] * model.config.num_layers
    batch_count = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits, _, all_gates = model(x)

        task_loss = criterion(logits.squeeze(-1), y)
        gate_loss = torch.stack([g.abs().mean() for g in all_gates]).mean()

        loss = task_loss + gate_lambda * gate_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += task_loss.item() * x.size(0)

        for i, g in enumerate(all_gates):
            gate_sums[i] += g.mean(dim=(1, 2)).mean().item()
        batch_count += 1

    avg_gates = [s / batch_count for s in gate_sums]
    return total_loss / len(loader.dataset), avg_gates


@torch.no_grad()
def evaluate_quant(model, loader, device):
    """评估: IC, RankIC, Sharpe (按月分组)"""
    model.eval()
    preds, labels, dates = [], [], []

    for batch in tqdm(loader, desc="  eval", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits, _, _ = model(x)
        preds.extend(logits.squeeze(-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        dates.extend(batch["date"])

    preds = np.array(preds)
    labels = np.array(labels)
    dates = pd.to_datetime(dates)

    df = pd.DataFrame({"pred": preds, "label": labels, "date": dates})
    df["year_month"] = df["date"].dt.to_period("M")

    ic_list = []
    rankic_list = []
    returns = []

    for period, group in df.groupby("year_month"):
        if len(group) < 10:
            continue

        pred_vals = group["pred"].values
        label_vals = group["label"].values

        ic = np.corrcoef(pred_vals, label_vals)[0, 1] if len(pred_vals) > 1 else 0
        ic_list.append(ic)

        if len(np.unique(pred_vals)) > 1 and len(np.unique(label_vals)) > 1:
            rankic, _ = spearmanr(pred_vals, label_vals)
            rankic_list.append(rankic)

        top_pct = 0.2
        n_select = max(1, int(len(group) * top_pct))
        top_indices = np.argsort(pred_vals)[-n_select:]
        ret = np.mean(label_vals[top_indices])
        returns.append(ret)

    ic = np.mean(ic_list) if ic_list else 0
    rankic = np.mean(rankic_list) if rankic_list else 0
    sharpe = (
        np.mean(returns) / np.std(returns)
        if len(returns) > 1 and np.std(returns) > 0
        else 0
    )

    return {
        "IC": ic,
        "RankIC": rankic,
        "Sharpe": sharpe,
        "IC_IR": np.mean(ic_list) / np.std(ic_list) if len(ic_list) > 1 else 0,
    }


def run_experiment(
    model_name, create_fn, train_loader, val_loader, test_loader, config, device
):
    print(f"\n{'=' * 60}")
    print(f"训练模型: {model_name}")
    print(f"{'=' * 60}")

    model = create_fn(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    total_steps = len(train_loader) * config.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=total_steps, pct_start=0.1
    )
    criterion = nn.MSELoss()

    best_sharpe = -float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss, avg_gates = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            config.gate_lambda,
        )

        metrics = evaluate_quant(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | Train MSE: {train_loss:.6f} | "
            f"IC: {metrics['IC']:.4f} | RankIC: {metrics['RankIC']:.4f} | Sharpe: {metrics['Sharpe']:.4f}"
        )
        print(
            f"  Gate Values: {' | '.join([f'L{i}: {g:.4f}' for i, g in enumerate(avg_gates)])}"
        )

        if metrics["Sharpe"] > best_sharpe:
            best_sharpe = metrics["Sharpe"]
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    final_metrics = evaluate_quant(model, test_loader, device)

    print(f"\n>>> {model_name} 最佳结果:")
    print(f"    IC: {final_metrics['IC']:.4f}")
    print(f"    RankIC: {final_metrics['RankIC']:.4f}")
    print(f"    Sharpe: {final_metrics['Sharpe']:.4f}")
    print(f"    IC_IR: {final_metrics['IC_IR']:.4f}")

    return final_metrics


def main():
    set_seed(42)
    print(">>> 加载数据...")
    train_df, val_df, test_df, features, target = prepare_data(
        db_path="data/quant_lab.duckdb", train_end="2023-12-31", val_end="2024-12-31"
    )

    test_df = test_df[test_df["date"] >= "2025-01-01"]

    config = AntConfig(
        model_type="financial",
        input_dim=len(features),
        num_classes=1,
        max_seq_len=6,
        d_model=64,
        num_layers=4,
        batch_size=1024,
        epochs=20,
        lr=1e-3,
        gate_lambda=0.08,
        cross_layer_heads=4,
        d_ff=256,
        gate_hidden_dim=64,
        dropout=0.1,
    )

    print(f"\n数据切分: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_ds = FinancialDataset(train_df, features, target, seq_len=config.max_seq_len)
    val_ds = FinancialDataset(val_df, features, target, seq_len=config.max_seq_len)
    test_ds = FinancialDataset(test_df, features, target, seq_len=config.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    results["A_Layer0"] = run_experiment(
        "模型A: 仅 Layer0",
        create_model_a,
        train_loader,
        val_loader,
        test_loader,
        config,
        device,
    )

    results["B_Layer0_Layer2"] = run_experiment(
        "模型B: Layer0 + Layer2（剪枝）",
        create_model_b,
        train_loader,
        val_loader,
        test_loader,
        config,
        device,
    )

    results["C_Original4Layer"] = run_experiment(
        "模型C: 原始4层+gate",
        create_model_c,
        train_loader,
        val_loader,
        test_loader,
        config,
        device,
    )

    print("\n" + "=" * 60)
    print("实验结果汇总 (Test Set)")
    print("=" * 60)
    print(f"{'模型':<25} {'IC':>8} {'RankIC':>8} {'Sharpe':>8} {'IC_IR':>8}")
    print("-" * 60)
    for name, metrics in results.items():
        print(
            f"{name:<25} {metrics['IC']:>8.4f} {metrics['RankIC']:>8.4f} {metrics['Sharpe']:>8.4f} {metrics['IC_IR']:>8.4f}"
        )
    print("=" * 60)

    best_model = max(results.items(), key=lambda x: x[1]["Sharpe"])
    print(f"\n最佳模型: {best_model[0]} (Sharpe: {best_model[1]['Sharpe']:.4f})")


if __name__ == "__main__":
    main()
