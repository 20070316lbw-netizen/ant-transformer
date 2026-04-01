from .base import BaseModelAdapter
from .transformer_base import StandardTransformerAdapter
from .transformer_gating import GatingTransformerAdapter
from .transformer_lookback import LookbackTransformerAdapter
from .ant_transformer_adapter import AntTransformerAdapter
from .lightgbm_model import LightGBMAdapter

MODEL_REGISTRY = {
    "standard_transformer": StandardTransformerAdapter,
    "transformer_gating": GatingTransformerAdapter,
    "transformer_lookback": LookbackTransformerAdapter,
    "ant_transformer": AntTransformerAdapter,
    "lightgbm": LightGBMAdapter,
}

def get_model(model_name: str, config=None) -> BaseModelAdapter:
    """
    Factory method to retrieve a model instance based on string configuration.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](config=config)
