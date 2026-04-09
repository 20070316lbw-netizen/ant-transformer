import torch
from torch.utils.data import Dataset
import numpy as np


class FinancialDataset(Dataset):
    """
    将原始 DataFrame 转换为适合 Transformer 的序列数据。
    每个样本为 (seq_len, input_dim) 的特征矩阵和对应的标量 label。
    """

    def __init__(self, df, feature_cols, target_col, seq_len=6):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Pre-format dates for performance
        if "date" in df.columns:
            df_working = df.assign(_formatted_date=df["date"].dt.strftime("%Y-%m-%d"))
        else:
            df_working = df.copy()

        # 按照 ticker 分组并生成序列索引
        self.samples = []
        for ticker, group in df_working.groupby("ticker"):
            if len(group) < seq_len:
                continue

            # 转换为 numpy 提高索引性能
            features = group[feature_cols].values.astype(np.float32)
            labels = group[target_col].values.astype(np.float32)

            dates = group["_formatted_date"].values if "_formatted_date" in group else group["date"].values
            tickers = group["ticker"].values

            # 滑动窗口
            num_samples = len(group) - seq_len + 1
            for i in range(num_samples):
                # 特征是 i 到 i+seq_len
                # 标签取 i+seq_len-1 位置的 (因为 label_next_month 已经是对应当前日的未来收益)
                self.samples.append(
                    {
                        "x": features[i : i + seq_len],
                        "y": labels[i + seq_len - 1],
                        "date": dates[i + seq_len - 1],
                        "ticker": tickers[i + seq_len - 1],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "x": torch.from_numpy(sample["x"]),
            "y": torch.tensor(sample["y"]),
            "date": sample["date"],
            "ticker": sample["ticker"],
        }
