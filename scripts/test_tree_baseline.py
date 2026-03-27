import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from loguru import logger

# 添加项目根目录到 sys.path
sys.path.append(os.getcwd())

from data.data_prep import prepare_data

def calculate_ic(df, feature='pred', target='label_next_month'):
    """计算 IC (秩相关系数)"""
    # 按日期汇总
    ics = []
    for date, group in df.groupby('date'):
        if len(group) < 2: continue
        ic, _ = spearmanr(group[feature], group[target])
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else 0.0

def main():
    logger.info("开始 LightGBM 基准测试流程...")
    
    # 1. 准备数据
    train_df, val_df, test_df, features, target = prepare_data()
    
    # 2. 转换数据格式
    dtrain = lgb.Dataset(train_df[features], label=train_df[target])
    dval   = lgb.Dataset(val_df[features], label=val_df[target], reference=dtrain)
    
    # 3. 设置 LightGBM 参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'verbose': -1
    }
    
    # 4. 训练模型
    logger.info("正在训练 LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # 5. 预测与评估
    logger.info("正在评估测试集...")
    test_df['pred'] = model.predict(test_df[features])
    
    ic = calculate_ic(test_df)
    mse = np.mean((test_df['pred'] - test_df[target])**2)
    
    print("\n" + "="*40)
    print("LightGBM 基准测试结果")
    print("="*40)
    print(f"训练集行数: {len(train_df)}")
    print(f"验证集行数: {len(val_df)}")
    print(f"测试集行数: {len(test_df)}")
    print(f"测试集 MSE: {mse:.6f}")
    print(f"测试集 Rank IC: {ic:.4f}")
    
    if ic > 0.02:
        print(">>> 结论: 因子具有显著预测效能 (IC > 2%)")
    else:
        print(">>> 结论: 因子信号较弱或存在噪声")
    print("="*40)

if __name__ == "__main__":
    main()
