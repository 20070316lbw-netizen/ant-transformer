import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List
from loguru import logger

from model.config import AntConfig
from model.ant import AntTransformer
from data.financial_dataset import FinancialDataset
from model.losses import PearsonCorrLoss
from .base import BaseModelAdapter


class StandardTransformerAdapter(BaseModelAdapter):
    """
    Standard Transformer without soft gating or history lookback (cross-layer attention).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device("cuda" if getattr(config, "use_cuda", True) and torch.cuda.is_available() else "cpu")
        self.model = None

    def _init_model(self, input_dim: int):
        # Update config to act as a standard transformer
        self.config.input_dim = input_dim
        # Disable layer pruning (soft gating)
        self.config.enable_layer_pruning = False
        # Disable history lookback (cross-layer attention)
        # We can simulate this by setting cross_layer_heads to 0 or similar, but AntLayer implementation
        # might fail if cross_layer_heads is 0 (d_model % cross_layer_heads == 0 check).
        # We'll rely on our modified AntEncoder if needed, but let's see if setting gate_lambda = 0
        # and not using the cross-layer output helps, or just setting cross_layer_heads=0
        # But `config.validate()` has: `if self.d_model % self.cross_layer_heads != 0: raise ValueError(...)`
        # Thus cross_layer_heads cannot be 0.
        # We will set a flag in the config to disable it, which we'll need to patch in model/encoder.py or just use the model as is but ignore gates.
        # Wait, the instruction says "标准 Transformer". We can dynamically inject a parameter `use_cross_layer=False`.
        self.model = AntTransformer(self.config).to(self.device)

        # Disable ablation flags for standard transformer
        real_model = self.model.module if hasattr(self.model, 'module') else self.model
        real_model.encoder.set_ablation_flags(use_cross_layer=False, use_soft_gating=False)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6):
        self._init_model(input_dim=len(features))

        train_ds = FinancialDataset(train_df, features, target_col, seq_len=seq_len)
        val_ds = FinancialDataset(val_df, features, target_col, seq_len=seq_len)

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=getattr(self.config, "num_workers", 0))
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)

        loss_type = getattr(self.config, "loss_type", "mse")
        if loss_type == "ic":
            criterion = PearsonCorrLoss()
        else:
            criterion = nn.MSELoss()

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{self.config.epochs}", leave=False):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self.model(x, enable_pruning=False)  # Force no pruning
                loss = criterion(logits.squeeze(-1), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)

            avg_loss = total_loss / len(train_loader.dataset)
            logger.info(f"Epoch {epoch:02d} | Loss: {avg_loss:.6f}")

    @torch.no_grad()
    def predict(self, test_df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6) -> pd.DataFrame:
        test_ds = FinancialDataset(test_df, features, target_col, seq_len=seq_len)
        if len(test_ds) == 0:
            from .base import get_empty_prediction_df
            return get_empty_prediction_df()

        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_targets = []
        all_dates = []
        all_tickers = []

        for batch in tqdm(test_loader, desc="  predicting", leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].numpy()
            logits, _, _ = self.model(x, enable_pruning=False)

            all_preds.append(logits.squeeze(-1).cpu().numpy())
            all_targets.append(y)
            all_dates.extend(batch["date"])
            all_tickers.extend(batch["ticker"])

        return pd.DataFrame({
            "ticker": all_tickers,
            "date": all_dates,
            "target": np.concatenate(all_targets),
            "pred": np.concatenate(all_preds),
        })
