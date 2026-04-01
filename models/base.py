import pandas as pd
from typing import List

class BaseModelAdapter:
    """
    Base class for all model adapters to ensure a consistent interface.
    """
    def __init__(self, config=None):
        self.config = config

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6):
        """
        Trains the model on the provided data.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            features: List of feature column names.
            target_col: Target column name.
            seq_len: Sequence length for time series models.
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(self, test_df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 6) -> pd.DataFrame:
        """
        Generates predictions for the test data.

        Args:
            test_df: Test DataFrame.
            features: List of feature column names.
            target_col: Target column name.
            seq_len: Sequence length for time series models.

        Returns:
            DataFrame with at least 'date', 'ticker', 'target', and 'pred' columns.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

def get_empty_prediction_df() -> pd.DataFrame:
    """Helper method to return an empty DataFrame when predictions fail or dataset is empty."""
    return pd.DataFrame({
        "ticker": [],
        "date": [],
        "target": [],
        "pred": []
    })
