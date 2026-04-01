import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import torch
import numpy as np
import random
from loguru import logger

sys.path.append(os.getcwd())

from model.config import AntConfig
from data.data_prep import prepare_data
from evaluate import calculate_metrics
from models import get_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config_path(config_path: str) -> str:
    if os.path.exists(config_path):
        return config_path
    repo_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", config_path)
    if os.path.exists(repo_relative):
        return repo_relative
    raise FileNotFoundError(f"找不到配置文件: {config_path}。")


def main():
    parser = argparse.ArgumentParser(description="量化研究实验室可扩展模型对比实验平台")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # Load configuration
    config_path = resolve_config_path(args.config)
    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Initialize Base Config, then override with custom fields if any
    config = AntConfig()

    # We load standard AntConfig sections
    for section in ["model", "data", "training", "experiment", "advanced"]:
        if section in config_dict:
            for k, v in config_dict[section].items():
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    # Dynamically add custom config keys like `model_name`
                    setattr(config, k, v)

    # Force validate standard options (AntConfig does this)
    config.validate()

    # Check model name explicitly required by ablation lab
    model_name = getattr(config, "model_name", "ant_transformer")
    logger.info(f"Running Experiment with model: {model_name}")

    set_seed(getattr(config, "seed", 42))

    # Data Preparation
    logger.info("Step 1/5: Loading Data...")
    train_df, val_df, test_df, features, target_col = prepare_data(
        train_end=config.train_end,
        val_end=config.val_end,
        use_dummy_data=getattr(config, "use_dummy_data", False)
    )

    if getattr(config, "use_subset", False) and getattr(config, "subset_size", None):
        logger.info(f"Using subset of size: {config.subset_size}")
        train_df = train_df.head(config.subset_size)
        val_df = val_df.head(config.subset_size // 2)
        test_df = test_df.head(config.subset_size // 2)

    seq_len = getattr(config, "seq_len", 6)

    # Model Initialization via Registry
    logger.info("Step 2/5: Initializing Model...")
    model_adapter = get_model(model_name, config)

    # Training
    logger.info("Step 3/5: Training Model...")
    model_adapter.fit(train_df, val_df, features, target_col, seq_len=seq_len)

    # Prediction
    logger.info("Step 4/5: Generating Predictions...")
    pred_df = model_adapter.predict(test_df, features, target_col, seq_len=seq_len)

    output_prefix = getattr(config, "output_prefix", "outputs")
    os.makedirs(output_prefix, exist_ok=True)
    pred_path = os.path.join(output_prefix, f"pred_{model_name}.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")

    # Evaluation
    logger.info("Step 5/5: Evaluating Metrics...")
    # Calculate metrics uses the file path
    metrics_monthly, metrics_combo = calculate_metrics(pred_path, verbose=True)

    # Results Logging
    logger.info("Saving results to logs...")
    os.makedirs("logs", exist_ok=True)

    # We gather the core metrics to save in JSON format
    result_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "config": {
            "model": config_dict.get("model", {}),
            "data": config_dict.get("data", {}),
            "training": config_dict.get("training", {}),
        },
        "metrics": {
            "mean_ic": metrics_monthly["IC"].mean() if metrics_monthly is not None else None,
            "mean_rank_ic": metrics_monthly["RankIC"].mean() if metrics_monthly is not None else None,
            "ic_ir": metrics_monthly["IC"].mean() / metrics_monthly["IC"].std() if metrics_monthly is not None and len(metrics_monthly) > 1 else None,
            "sharpe": metrics_combo.get("sharpe", None) if metrics_combo else None,
            "max_dd": metrics_combo.get("max_dd", None) if metrics_combo else None,
        }
    }

    log_file = f"logs/exp_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=4, ensure_ascii=False)

    logger.success(f"Experiment Finished Successfully! Log saved at: {log_file}")

if __name__ == "__main__":
    main()
