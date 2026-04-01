import pandas as pd
import numpy as np
from scipy import stats
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
        if len(group) < 2:
            return pd.Series({"IC": np.nan, "RankIC": np.nan})
        ic = group["target"].corr(group["pred"])
        rank_ic = group["target"].corr(group["pred"], method="spearman")
        return pd.Series({"IC": ic, "RankIC": rank_ic})

    if len(df) == 0:
        logger.warning("预测文件为空，无法计算指标。")
        return pd.DataFrame(columns=["IC", "RankIC"]), {"sharpe": 0, "max_dd": 0}

    monthly_metrics = df.groupby("date").apply(ic_group)

    if "IC" in monthly_metrics.columns:
        mean_ic = monthly_metrics["IC"].mean()
        mean_rank_ic = monthly_metrics["RankIC"].mean()
        ic_ir = mean_ic / monthly_metrics["IC"].std() if len(monthly_metrics) > 1 and monthly_metrics["IC"].std() > 0 else 0
    else:
        mean_ic = np.nan
        mean_rank_ic = np.nan
        ic_ir = np.nan

    if verbose:
        print("\n" + "=" * 60)
        print("      预测性能评估记录 (Metrics Report)")
        print("=" * 60)
        print(f"数据范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
        print(f"总样本数: {len(df):,}")
        print(f"分月均值 IC:     {mean_ic:.4f}")
        print(f"分月均值 Rank IC: {mean_rank_ic:.4f}")
        print(f"IC IR:           {ic_ir:.4f}")

        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_1samp(monthly_metrics["RankIC"].dropna(), 0)
        print(f"月度 RankIC 统计:")
        print(f"  正月比例:   {(monthly_metrics['RankIC'] > 0).mean():.2%}")
        print(f"  IC_IR:      {mean_ic / monthly_metrics['IC'].std():.4f}")
        print(f"  t统计量:    {t_stat:.4f}")
        print(f"  p值:        {p_value:.4f}")
        print(f"  {'✅ 显著 (p<0.05)' if p_value < 0.05 else '⚠️ 不显著，样本不足'}")

    # 2. 多空组合回测 (Long-Short Strategy)
    # 多头 (Long): 前 20% | 空头 (Short): 后 20%
    def ls_strategy_ret(group):
        n_20 = max(1, int(len(group) * 0.2))
        if n_20 == 0: return pd.Series({"long": 0, "short": 0, "ls": 0})
        long_group = group.nlargest(n_20, "pred")
        short_group = group.nsmallest(n_20, "pred")
        
        l_ret = long_group["target"].mean()
        s_ret = short_group["target"].mean()
        return pd.Series({"long": l_ret, "short": s_ret, "ls": l_ret - s_ret})

    strat_rets = df.groupby("date").apply(ls_strategy_ret)
    
    # 计算多空组合指标
    ls_ret = strat_rets["ls"]
    ann_ret = ls_ret.mean() * 12
    ann_vol = ls_ret.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # 最大回撤 (基于净值算)
    cum_val = (1 + ls_ret).cumprod()
    max_dd = (cum_val / cum_val.cummax() - 1).min()

    if verbose:
        print("-" * 60)
        print(f"多空对冲组合 (Top 20% vs Bottom 20%) 表现:")
        print(f"年化多空收益: {ann_ret:.2%}")
        print(f"年化多空波动: {ann_vol:.2%}")
        print(f"多空 Sharpe:  {sharpe:.4f}")
        print(f"多空 MaxDD:   {max_dd:.2%}")
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
    calculate_metrics(args.pred_path, verbose=True)

    if not args.skip_monthly and not args.skip_combination:
        logger.success("评估完成！")


if __name__ == "__main__":
    main()
