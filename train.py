import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import random
from loguru import logger

# 添加项目根目录
sys.path.append(os.getcwd())

from model.config import AntConfig
from model.ant import AntTransformer
from data.data_prep import prepare_data
from data.financial_dataset import FinancialDataset


def resolve_config_path(config_path: str) -> str:
    """兼容不同工作目录，优先用传入路径，其次回退到项目根目录下同名文件。"""
    if os.path.exists(config_path):
        return config_path
    repo_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    if os.path.exists(repo_relative):
        return repo_relative
    raise FileNotFoundError(
        f"找不到配置文件: {config_path}。请确认当前工作目录或通过 --config 传入绝对路径。"
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model, loader, optimizer, criterion, device, gate_lambda, enable_pruning
):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _, all_gates = model(x, enable_pruning=enable_pruning)
        task_loss = criterion(logits.squeeze(-1), y)
        gate_loss = torch.stack([g.abs().mean() for g in all_gates]).mean()
        loss = task_loss + gate_lambda * gate_loss
        loss.backward()
        optimizer.step()
        total_loss += task_loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_predictions(model, loader, device, features, target):
    model.eval()
    all_preds = []
    all_targets = []
    all_dates = []
    all_tickers = []

    for batch in tqdm(loader, desc="  predicting", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        logits, _, _ = model(x, enable_pruning=model.config.enable_layer_pruning)

        # 记录预测
        all_preds.append(logits.squeeze(-1).cpu().numpy())
        all_targets.append(y)
        all_dates.extend(batch["date"])
        all_tickers.extend(batch["ticker"])

    return pd.DataFrame(
        {
            "ticker": all_tickers,
            "date": all_dates,
            "target": np.concatenate(all_targets),
            "pred": np.concatenate(all_preds),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Ant-Transformer 统一训练入口")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径 (config.yaml)"
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default=None,
        choices=["full", "layer0_layer2", "layer0"],
        help="覆盖配置文件中的模型架构",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="覆盖配置文件中的训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="覆盖配置文件中的批次大小"
    )
    parser.add_argument("--lr", type=float, default=None, help="覆盖配置文件中的学习率")
    parser.add_argument(
        "--gate_lambda", type=float, default=None, help="覆盖配置文件中的门控正则化权重"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="仅用于快速测试的样本数"
    )
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--val_end", type=str, default=None)
    parser.add_argument("--no_pruning", action="store_true", help="禁用层裁剪功能")
    args = parser.parse_args()

    # 1. 加载配置文件
    config_path = resolve_config_path(args.config)
    logger.info(f"正在加载配置文件: {config_path}")
    config = AntConfig.load_from_yaml(config_path)

    # 2. 命令行参数覆盖
    if args.model_arch:
        config.model_arch = args.model_arch
        config.update_by_arch()
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.gate_lambda:
        config.gate_lambda = args.gate_lambda
    if args.train_end:
        config.train_end = args.train_end
    if args.val_end:
        config.val_end = args.val_end
    if args.no_pruning:
        config.enable_layer_pruning = False

    config.validate()

    # 3. 设备配置
    set_seed(getattr(config, "seed", 42))
    device = torch.device(
        "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
    )
    if not config.use_cuda:
        device = torch.device("cpu")
    logger.info(f"使用设备: {device}")

    # 4. 数据准备
    train_df, val_df, test_df, features, target_col = prepare_data(
        train_end=config.train_end, val_end=config.val_end
    )

    if config.use_subset and config.subset_size:
        logger.info(f"使用数据子集，大小: {config.subset_size}")
        train_df = train_df.head(config.subset_size)
        val_df = val_df.head(config.subset_size // 2)

    train_ds = FinancialDataset(train_df, features, target_col, seq_len=config.seq_len)
    val_ds = FinancialDataset(val_df, features, target_col, seq_len=config.seq_len)
    config.input_dim = len(features)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # 5. 模型
    logger.info(
        f"初始化模型: {config.model_arch} (num_layers={config.num_layers}, pruning={config.enable_layer_pruning})"
    )
    model = AntTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    # 6. 训练
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.gate_lambda,
            config.enable_layer_pruning,
        )
        logger.info(f"Epoch {epoch:02d} | Loss: {loss:.6f}")

    # 7. 生成预测并保存
    logger.info("正在生成验证集预测结果...")
    pred_df = get_predictions(model, val_loader, device, features, target_col)

    os.makedirs(config.output_prefix, exist_ok=True)
    out_path = f"{config.output_prefix}/pred_{config.model_arch}.csv"
    pred_df.to_csv(out_path, index=False)

    logger.success(f"实验结束！预测结果已保存至: {out_path}")
    print(f"\n>>> 请运行: python evaluate.py --pred_path {out_path}")


if __name__ == "__main__":
    main()
