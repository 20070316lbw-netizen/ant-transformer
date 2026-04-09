import pandas as pd
import numpy as np
import time

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

def orig_prep(df, features, target_col, seq_len):
    all_x = []
    all_y = []
    all_dates = []
    all_tickers = []

    for ticker, group in df.groupby("ticker"):
        if len(group) < seq_len:
            continue

        feats = group[features].values
        labels = group[target_col].values
        dates = group["date"].dt.strftime("%Y-%m-%d").values
        tickers = group["ticker"].values

        num_samples = len(group) - seq_len + 1
        for i in range(num_samples):
            # Flatten seq_len * num_features
            x_flat = feats[i:i + seq_len].flatten()
            all_x.append(x_flat)
            all_y.append(labels[i + seq_len - 1])
            all_dates.append(dates[i + seq_len - 1])
            all_tickers.append(tickers[i + seq_len - 1])

    return np.array(all_x), np.array(all_y), all_dates, all_tickers

def new_prep(df, features, target_col, seq_len):
    all_x = []
    all_y = []
    all_dates = []
    all_tickers = []

    for ticker, group in df.groupby("ticker"):
        if len(group) < seq_len:
            continue

        feats = group[features].values
        labels = group[target_col].values
        dates = group["date"].dt.strftime("%Y-%m-%d").values
        tickers = group["ticker"].values

        # vectorized
        windowed_feats = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)
        windowed_feats = windowed_feats.swapaxes(1, 2)
        x_flat = windowed_feats.reshape(windowed_feats.shape[0], -1)

        all_x.append(x_flat)
        all_y.append(labels[seq_len - 1:])
        all_dates.append(dates[seq_len - 1:])
        all_tickers.append(tickers[seq_len - 1:])

    if not all_x:
        return np.array([]), np.array([]), [], []

    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # keeping lists as per original
    all_dates_flat = []
    for sublist in all_dates:
        all_dates_flat.extend(sublist)

    all_tickers_flat = []
    for sublist in all_tickers:
        all_tickers_flat.extend(sublist)

    return all_x, all_y, all_dates_flat, all_tickers_flat

df, features = create_dummy_data(200, 1000, 20)

print("Starting original...")
t0 = time.time()
x1, y1, d1, t1 = orig_prep(df, features, "target", 6)
t1_time = time.time() - t0
print(f"Original took {t1_time:.4f}s")

print("Starting new...")
t0 = time.time()
x2, y2, d2, t2 = new_prep(df, features, "target", 6)
t2_time = time.time() - t0
print(f"New took {t2_time:.4f}s")

print(f"Speedup: {t1_time / t2_time:.2f}x")

print("Checking equality...")
print("X equal:", np.allclose(x1, x2))
print("Y equal:", np.allclose(y1, y2))
print("Dates equal:", d1 == d2)
print("Tickers equal:", t1 == t2)
