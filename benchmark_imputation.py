import time
import pandas as pd
import numpy as np

# Generate large synthetic data
N = 1_000_000
feature_cols = [f'feature_{i}' for i in range(20)]
data = np.random.randn(N, 20)

# Introduce missing values
mask = np.random.rand(N, 20) < 0.1
data[mask] = np.nan

df = pd.DataFrame(data, columns=feature_cols)

train_df = df.iloc[:600000].copy()
val_df = df.iloc[600000:800000].copy()
test_df = df.iloc[800000:].copy()

# Baseline method
def baseline(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for col in feature_cols:
        if train_df[col].isnull().any():
            filler = train_df[col].median()
            train_df[col] = train_df[col].fillna(filler)
            val_df[col]   = val_df[col].fillna(filler)
            test_df[col]  = test_df[col].fillna(filler)
    return train_df, val_df, test_df

# Optimized method
def optimized(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    medians = train_df[feature_cols].median()
    missing_mask = train_df[feature_cols].isnull().any()
    missing_cols = missing_mask[missing_mask].index

    if len(missing_cols) > 0:
        fill_dict = medians[missing_cols].to_dict()
        train_df.fillna(value=fill_dict, inplace=True)
        val_df.fillna(value=fill_dict, inplace=True)
        test_df.fillna(value=fill_dict, inplace=True)

    return train_df, val_df, test_df

start = time.time()
baseline(train_df, val_df, test_df)
t_baseline = time.time() - start

start = time.time()
optimized(train_df, val_df, test_df)
t_optimized = time.time() - start

print(f"Baseline: {t_baseline:.4f} s")
print(f"Optimized: {t_optimized:.4f} s")
print(f"Speedup: {t_baseline / t_optimized:.2f}x")
