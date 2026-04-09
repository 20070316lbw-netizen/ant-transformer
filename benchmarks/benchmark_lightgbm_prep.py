import pandas as pd
import numpy as np
import time
from models.lightgbm_model import LightGBMAdapter

def create_dummy_data(num_tickers, num_days, num_features):
    data = []
    dates = pd.date_range("2020-01-01", periods=num_days)
    for i in range(num_tickers):
        ticker = f"T{i}"
        for j in range(num_days):
            row = {"ticker": ticker, "date": dates[j], "target": np.random.randn()}
            for k in range(num_features):
                row[f"f{k}"] = np.random.randn()
            data.append(row)
    df = pd.DataFrame(data)
    features = [f"f{k}" for k in range(num_features)]
    return df, features

def run_benchmark():
    df, features = create_dummy_data(100, 500, 20)
    print(f"Data shape: {df.shape}")

    adapter = LightGBMAdapter()

    start_time = time.time()
    all_x, all_y, all_dates, all_tickers = adapter._prepare_tabular_data(df, features, "target", seq_len=6)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Shapes: X: {all_x.shape}, y: {all_y.shape}")

if __name__ == "__main__":
    run_benchmark()
