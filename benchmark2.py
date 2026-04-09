import time
import pandas as pd
import numpy as np

# Create mock dataframe
num_tickers = 2000
rows_per_ticker = 1000
total_rows = num_tickers * rows_per_ticker

tickers = [f"TICKER_{i}" for i in range(num_tickers)]
ticker_col = np.repeat(tickers, rows_per_ticker)
date_col = pd.date_range("2020-01-01", periods=rows_per_ticker).tolist() * num_tickers

df = pd.DataFrame({
    "ticker": ticker_col,
    "date": date_col,
})

# Benchmark 1: Current implementation
start_time = time.time()
for ticker, group in df.groupby("ticker"):
    dates = group["date"].dt.strftime("%Y-%m-%d").values
end_time = time.time()
print(f"Current Implementation Formatting Only: {end_time - start_time:.4f} seconds")

# Benchmark 2: Optimized implementation
start_time = time.time()
date_strs = df["date"].dt.strftime("%Y-%m-%d")
df = df.assign(date_str=date_strs)
for ticker, group in df.groupby("ticker"):
    dates = group["date_str"].values
end_time = time.time()
print(f"Optimized Implementation Formatting Only: {end_time - start_time:.4f} seconds")
