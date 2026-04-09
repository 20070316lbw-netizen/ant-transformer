import time
import pandas as pd
import numpy as np

# Create mock dataframe
num_tickers = 500
rows_per_ticker = 1000
total_rows = num_tickers * rows_per_ticker

tickers = [f"TICKER_{i}" for i in range(num_tickers)]
ticker_col = np.repeat(tickers, rows_per_ticker)
date_col = pd.date_range("2020-01-01", periods=rows_per_ticker).tolist() * num_tickers
feature_col = np.random.randn(total_rows)
target_col = np.random.randn(total_rows)

df = pd.DataFrame({
    "ticker": ticker_col,
    "date": date_col,
    "feature1": feature_col,
    "target": target_col
})

feature_cols = ["feature1"]
target_col = "target"
seq_len = 6

# Benchmark 1: Current implementation
start_time = time.time()
samples_1 = []
for ticker, group in df.groupby("ticker"):
    if len(group) < seq_len:
        continue

    features = group[feature_cols].values.astype(np.float32)
    labels = group[target_col].values.astype(np.float32)

    # Slow part: string formatting inside the loop
    dates = group["date"].dt.strftime("%Y-%m-%d").values
    tickers = group["ticker"].values

    num_samples = len(group) - seq_len + 1
    for i in range(num_samples):
        samples_1.append(
            {
                "x": features[i : i + seq_len],
                "y": labels[i + seq_len - 1],
                "date": dates[i + seq_len - 1],
                "ticker": tickers[i + seq_len - 1],
            }
        )
end_time = time.time()
print(f"Current Implementation: {end_time - start_time:.4f} seconds")

# Benchmark 2: Optimized implementation
start_time = time.time()

# OPTIMIZATION: Format date to string once on the entire dataframe BEFORE groupby
df_copy = df.copy()
df_copy["date_str"] = df_copy["date"].dt.strftime("%Y-%m-%d")

samples_2 = []
for ticker, group in df_copy.groupby("ticker"):
    if len(group) < seq_len:
        continue

    features = group[feature_cols].values.astype(np.float32)
    labels = group[target_col].values.astype(np.float32)

    # OPTIMIZATION: Use the pre-formatted string
    dates = group["date_str"].values
    tickers = group["ticker"].values

    num_samples = len(group) - seq_len + 1
    for i in range(num_samples):
        samples_2.append(
            {
                "x": features[i : i + seq_len],
                "y": labels[i + seq_len - 1],
                "date": dates[i + seq_len - 1],
                "ticker": tickers[i + seq_len - 1],
            }
        )
end_time = time.time()
print(f"Optimized Implementation: {end_time - start_time:.4f} seconds")

# Verify correctness
# assert len(samples_1) == len(samples_2)
# for i in range(10):
#     assert samples_1[i]["date"] == samples_2[i]["date"]
