import pandas as pd
import numpy as np
from loguru import logger
import argparse
import os


def calculate_metrics(pred_path, verbose=True):
    if not os.path.exists(pred_path):
        logger.error(f"找不到预测文件: {pred_path}")
        return None, None

    logger.info(f"正在读取预测数据: {pred_path}")
    df = pd.read_csv(pred_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1. 计算分月 IC / Rank IC
    def ic_group(group):
        ic = group["target"].corr(group["pred"])
        rank_ic = group["target"].corr(group["pred"], method="spearman")
        return pd.Series({"IC": ic, "RankIC": rank_ic})

    monthly_metrics = df.groupby("date").apply(ic_group)

    mean_ic = monthly_metrics["IC"].mean()
    mean_rank_ic = monthly_metrics["RankIC"].mean()
    ic_ir = mean_ic / monthly_metrics["IC"].std() if len(monthly_metrics) > 1 else 0

    if verbose:
        print("\n" + "=" * 60)
        print("      预测性能评估记录 (Metrics Report)")
        print("=" * 60)
        print(f"数据范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
        print(f"总样本数: {len(df):,}")
        print(f"分月均值 IC:     {mean_ic:.4f}")
        print(f"分月均值 Rank IC: {mean_rank_ic:.4f}")
        print(f"IC IR:           {ic_ir:.4f}")

    # 2. 简单多头组合回测 (模拟 Top 50 股票)
    # 假设每个月买入 pred 最高的 50 只，计算其次月 target (收益率) 的均值
    TOP_N = 50

    def top_n_ret(group):
        top_group = group.nlargest(TOP_N, "pred")
        return top_group["target"].mean()

    portfolio_ret = df.groupby("date").apply(top_n_ret)
    excess_ret = portfolio_ret - df.groupby("date")["target"].mean()  # 超额收益

    ann_ret = excess_ret.mean() * 12
    ann_vol = excess_ret.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (excess_ret.cumsum() - excess_ret.cumsum().cummax()).min()

    if verbose:
        print("-" * 60)
        print(f"多头组合 (Top {TOP_N}) 超额表现:")
        print(f"年化超额收益: {ann_ret:.2%}")
        print(f"年化超额波动: {ann_vol:.2%}")
        print(f"超额 Sharpe:  {sharpe:.4f}")
        print(f"超额 MaxDD:   {max_dd:.2%}")
        print("=" * 60 + "\n")

    return monthly_metrics, {"sharpe": sharpe, "max_dd": max_dd}


def main():
    parser = argparse.ArgumentParser(description="Ant-Transformer 评估系统")
    parser.add_argument(
        "--pred_path", type=str, required=True, help="预测结果 CSV 路径"
    )
    parser.add_argument(
        "--skip_monthly", action="store_true", help="跳过分月指标计算，只计算组合回测"
    )
    parser.add_argument(
        "--skip_combination", action="store_true", help="跳过组合回测，只计算分月指标"
    )
    parser.add_argument(
        "--top_n", type=int, default=50, help="组合回测时取前N个股票 (默认: 50)"
    )
    args = parser.parse_args()

    # 计算所有指标
    monthly_metrics = None
    combination_result = None

    if not args.skip_monthly:
        monthly_metrics, _ = calculate_metrics(args.pred_path, verbose=True)
    else:
        _, combination_result = calculate_metrics(args.pred_path, verbose=False)

    if not args.skip_combination:
        df = pd.read_csv(args.pred_path)
        df["date"] = pd.to_datetime(df["date"])

        TOP_N = args.top_n

        def top_n_ret(group):
            top_group = group.nlargest(TOP_N, "pred")
            return top_group["target"].mean()

        portfolio_ret = df.groupby("date").apply(top_n_ret)
        excess_ret = portfolio_ret - df.groupby("date")["target"].mean()

        ann_ret = excess_ret.mean() * 12
        ann_vol = excess_ret.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (excess_ret.cumsum() - excess_ret.cumsum().cummax()).min()

        if combination_result:
            combination_result["sharpe"] = sharpe
            combination_result["max_dd"] = max_dd
        else:
            combination_result = {"sharpe": sharpe, "max_dd": max_dd}

    if verbose and not args.skip_monthly and not args.skip_combination:
        logger.success("评估完成！")


if __name__ == "__main__":
    main()
