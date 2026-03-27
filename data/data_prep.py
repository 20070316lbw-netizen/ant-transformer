import duckdb
import pandas as pd
import numpy as np
from loguru import logger
import os

def prepare_data(db_path='data/quant_lab.duckdb', train_end='2023-12-31', val_end='2024-12-31'):
    logger.info(f"正在从 {db_path} 加载数据...")
    conn = duckdb.connect(db_path)
    
    # 1. 提取全量数据
    query = "SELECT * FROM features_cn"
    df = conn.execute(query).df()
    conn.close()
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    
    # 2. 定义特征和标签
    feature_cols = [
        'mom_20d', 'mom_60d', 'mom_12m_minus_1m', 
        'vol_60d_res', 'sp_ratio', 'turn_20d'
    ]
    target_col = 'label_next_month'
    
    # 3. 基础清洗
    # 丢弃标签缺失的行
    initial_count = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(f"丢弃标签缺失行: {initial_count - len(df)} 行")
    
    # 特征缺失值填充 (中位数)
    for col in feature_cols:
        if df[col].isnull().any():
            filler = df[col].median()
            df[col] = df[col].fillna(filler)
            logger.warning(f"字段 {col} 存在缺失，已使用中位数 {filler:.4f} 填充")

    # 4. 时间序列切分
    train_df = df[df['date'] <= train_end]
    val_df   = df[(df['date'] > train_end) & (df['date'] <= val_end)]
    test_df  = df[df['date'] > val_end]
    
    logger.info(f"切分完成: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 5. 空值二次检查
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
