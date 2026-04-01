import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import random
from loguru import logger

sys.path.append(os.getcwd())

from model.config import AntConfig
from data.data_prep import prepare_data
from evaluate import calculate_metrics
from models import get_model, MODEL_REGISTRY


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


def run_pipeline(model_name: str, config: AntConfig, config_dict: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, features: list, target_col: str):
    logger.info(f"--- Running Pipeline for model: {model_name} ---")
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
    metrics_monthly, metrics_combo = calculate_metrics(pred_path, verbose=True)

    # Results Logging
    logger.info("Saving results to logs...")
    os.makedirs("logs", exist_ok=True)

    mean_ic = metrics_monthly["IC"].mean() if metrics_monthly is not None and "IC" in metrics_monthly else None
    mean_rank_ic = metrics_monthly["RankIC"].mean() if metrics_monthly is not None and "RankIC" in metrics_monthly else None

    ic_std = metrics_monthly["IC"].std() if metrics_monthly is not None and "IC" in metrics_monthly else None
    ic_ir = mean_ic / ic_std if mean_ic is not None and ic_std is not None and ic_std > 0 else None

    result_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "config": {
            "model": config_dict.get("model", {}),
            "data": config_dict.get("data", {}),
            "training": config_dict.get("training", {}),
        },
        "metrics": {
            "mean_ic": mean_ic,
            "mean_rank_ic": mean_rank_ic,
            "ic_ir": ic_ir,
            "sharpe": metrics_combo.get("sharpe", None) if metrics_combo else None,
            "max_dd": metrics_combo.get("max_dd", None) if metrics_combo else None,
        }
    }

    log_file = f"logs/exp_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=4, ensure_ascii=False)

    logger.success(f"Pipeline finished for {model_name}! Log saved at: {log_file}")

    # Return metrics for summary
    return result_record["metrics"]


def main():
    parser = argparse.ArgumentParser(description="量化研究实验室可扩展模型对比实验平台")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--run_all", action="store_true", help="一次性跑完注册表中的所有模型并输出对比日志")
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

        # Ensure data is sorted by date before slicing to prevent Look-Ahead Bias
        train_df = train_df.sort_values(['date', 'ticker']).iloc[:config.subset_size]
        val_df = val_df.sort_values(['date', 'ticker']).iloc[:config.subset_size // 2]
        test_df = test_df.sort_values(['date', 'ticker']).iloc[:config.subset_size // 2]

    if args.run_all:
        logger.info("Running all models in the registry for comparison.")
        results = {}
        for name in MODEL_REGISTRY.keys():
            try:
                metrics = run_pipeline(name, config, config_dict, train_df, val_df, test_df, features, target_col)
                results[name] = metrics
            except Exception as e:
                logger.error(f"Failed running pipeline for {name}: {e}")
                results[name] = {"error": str(e)}

        # Output comparison table
        logger.info("=== Final Ablation Comparison ===")
        print(pd.DataFrame(results).T.to_string())

        # Save comparison log
        comp_log = f"logs/ablation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comp_log, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.success(f"Comparison log saved at: {comp_log}")

    else:
        run_pipeline(model_name, config, config_dict, train_df, val_df, test_df, features, target_col)

if __name__ == "__main__":
    main()
