import duckdb
import pandas as pd
import numpy as np
from loguru import logger
import os

def prepare_data(db_path='data/quant_lab.duckdb', train_end='2023-12-31', val_end='2024-12-31', use_dummy_data=False):
    if use_dummy_data:
        logger.info("使用虚拟数据进行测试...")
        # 1. 设置随机种子，保证每次生成的虚拟数据一致
        np.random.seed(42)

        # 2. 生成特征数据
        dates = pd.date_range(start="2022-01-01", end="2025-01-01", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
        data = []

        # 每个 ticker 每天生成一条记录
        for date in dates:
            for ticker in tickers:
                record = {
                    "date": date,
                    "ticker": ticker,
                    "mom_20d": np.random.randn(),
                    "mom_60d": np.random.randn(),
                    "mom_12m_minus_1m": np.random.randn(),
                    "vol_60d_res": np.random.randn(),
                    "sp_ratio": np.random.randn(),
                    "turn_20d": np.random.randn(),
                    "label_next_month": np.random.randn(),
                }
                data.append(record)

        df = pd.DataFrame(data)
    else:
        logger.info(f"正在从 {db_path} 加载数据...")

        # 使用 context manager，确保连接异常时也能关闭
        with duckdb.connect(db_path, read_only=True) as conn:
            df = conn.execute("SELECT * FROM features_cn").df()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])

    feature_cols = [
        'mom_20d', 'mom_60d', 'mom_12m_minus_1m',
        'vol_60d_res', 'sp_ratio', 'turn_20d'
    ]
    target_col = 'label_next_month'

    # 丢弃标签缺失的行
    initial_count = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(f"丢弃标签缺失行: {initial_count - len(df)} 行")

    # 先切分，再填充 —— 防止用未来数据的统计量填充训练集（look-ahead bias）
    train_df = df[df['date'] <= train_end].copy()
    val_df   = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df  = df[df['date'] > val_end].copy()

    # 只在 train 集上计算中位数，再应用到 val/test
    for col in feature_cols:
        if train_df[col].isnull().any():
            filler = train_df[col].median()   # 仅用训练集统计量
            train_df[col] = train_df[col].fillna(filler)
            val_df[col]   = val_df[col].fillna(filler)
            test_df[col]  = test_df[col].fillna(filler)
            logger.warning(f"字段 {col} 存在缺失，已使用训练集中位数 {filler:.4f} 填充")

    logger.info(f"切分完成: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 空值二次检查
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        null_count = d[feature_cols + [target_col]].isnull().sum().sum()
        if null_count > 0:
            logger.error(f"{name} 集合中仍存在 {null_count} 个空值！")
        else:
            logger.success(f"{name} 集合空值检查通过。")

    return train_df, val_df, test_df, feature_cols, target_col

if __name__ == "__main__":
    train_df, val_df, test_df, features, target = prepare_data()
    print("\n[Data Preparation Summary]")
    print(f"Features: {features}")
    print(f"Target: {target}")
    print(f"Total Rows: {len(train_df) + len(val_df) + len(test_df)}")
