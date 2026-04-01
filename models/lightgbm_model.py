import pandas as pd
import numpy as np
from typing import List
from loguru import logger
import lightgbm as lgb
from .base import BaseModelAdapter


class LightGBMAdapter(BaseModelAdapter):
    """
    LightGBM model adapter acting as a baseline.
    Tabularize sequence data by flattening or using only the last step's features.
    For simplicity, we will flatten the sequence data to preserve all information.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.model = None

    def _prepare_tabular_data(self, df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6):
        """
        Flatten the sequences into tabular form for LightGBM.
        We iterate per ticker and create sliding windows.
        """
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

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6):
        logger.info("Preparing tabular data for LightGBM...")
        X_train, y_train, _, _ = self._prepare_tabular_data(train_df, features, target_col, seq_len)
        X_val, y_val, _, _ = self._prepare_tabular_data(val_df, features, target_col, seq_len)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        num_workers = getattr(self.config, 'num_workers', -1)
        n_jobs = num_workers if num_workers != 0 else -1

        # Basic LightGBM parameters (can be tuned or set via config)
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': getattr(self.config, 'lr', 0.05),
            'num_leaves': getattr(self.config, 'num_leaves', 31),
            'verbose': -1,
            'seed': getattr(self.config, 'seed', 42),
            'n_jobs': n_jobs
        }

        logger.info("Training LightGBM model...")
        epochs = getattr(self.config, 'epochs', 100)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=epochs,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False), lgb.log_evaluation(period=0)]
        )
        logger.info(f"LightGBM training completed. Best iteration: {self.model.best_iteration}")

    def predict(self, test_df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6) -> pd.DataFrame:
        logger.info("Preparing test tabular data for LightGBM...")
        X_test, y_test, dates, tickers = self._prepare_tabular_data(test_df, features, target_col, seq_len)

        if len(X_test) == 0:
            from .base import get_empty_prediction_df
            return get_empty_prediction_df()

        logger.info("Generating predictions...")
        preds = self.model.predict(X_test, num_iteration=self.model.best_iteration)

        return pd.DataFrame({
            "ticker": tickers,
            "date": dates,
            "target": y_test,
            "pred": preds,
        })
