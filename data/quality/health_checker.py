import pandas as pd
from loguru import logger
from typing import Dict, Any

class DataHealthChecker:
    """
    通用数据健康检查器 - 改写自 Microsoft QLib
    
    检查项:
    1. check_ohlcv_columns() - OHLCV 列完整性
    2. check_missing_data() - 缺失值统计
    3. check_price_jumps() - 价格异常跳变(>50%)
    4. check_volume_jumps() - 成交量异常(>3倍)
    5. run_all_checks() - 运行所有检查并生成报告
    """
    
    def __init__(self, price_jump_threshold: float = 0.5, volume_jump_threshold: float = 3.0):
        """
        参数:
            price_jump_threshold: 价格跳变阈值(默认50%)
            volume_jump_threshold: 成交量跳变倍数阈值(默认3倍)
        """
        self.price_jump_threshold = price_jump_threshold
        self.volume_jump_threshold = volume_jump_threshold

    def check_ohlcv_columns(self, df: pd.DataFrame) -> bool:
        """检查 OHLCV 列完整性"""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"缺失必要的 OHLCV 列: {missing_cols}")
            return False
        return True

    def check_missing_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """统计每列的缺失值数量"""
        missing_counts = df.isnull().sum().to_dict()
        total_missing = sum(missing_counts.values())
        if total_missing > 0:
            logger.warning(f"发现缺失值, 总计: {total_missing} 处")
        else:
            logger.info("未发现缺失值")
        return missing_counts

    def check_price_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """检查价格异常跳变(相邻两天收盘价变化超过阈值)"""
        # 确保数据按股票和日期排序
        if "stock_code" in df.columns and "date" in df.columns:
            df = df.sort_values(by=["stock_code", "date"])
            
        jumps = []
        for stock, group in df.groupby("stock_code"):
            # 计算绝对收益率(不用pct_change以避免除零错误)
            returns = group["close"].pct_change().abs()
            jump_idx = returns > self.price_jump_threshold
            if jump_idx.any():
                abnormal = group[jump_idx].copy()
                abnormal["jump_ratio"] = returns[jump_idx]
                jumps.append(abnormal)
                
        if jumps:
            res = pd.concat(jumps)
            logger.warning(f"发现 {len(res)} 处价格异常跳变 (> {self.price_jump_threshold * 100}%)")
            return res
        return pd.DataFrame()

    def check_volume_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """检查成交量异常跳变(相邻两天成交量比值超过阈值)"""
        if "stock_code" in df.columns and "date" in df.columns:
            df = df.sort_values(by=["stock_code", "date"])
            
        jumps = []
        for stock, group in df.groupby("stock_code"):
            # 计算成交量倍数变化
            vol_ratio = group["volume"] / group["volume"].shift(1)
            # 同时也检查成交量突降 (如降为原来的1/3)
            jump_idx = (vol_ratio > self.volume_jump_threshold) | (vol_ratio < 1 / self.volume_jump_threshold)
            if jump_idx.any():
                abnormal = group[jump_idx].copy()
                abnormal["volume_ratio"] = vol_ratio[jump_idx]
                jumps.append(abnormal)
                
        if jumps:
            res = pd.concat(jumps)
            logger.warning(f"发现 {len(res)} 处成交量异常跳变 (倍数 > {self.volume_jump_threshold})")
            return res
        return pd.DataFrame()

    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """运行所有检查并生成报告"""
        logger.info("开始数据健康检查...")
        report = {}
        
        # 1. 检查列完整性
        is_complete = self.check_ohlcv_columns(df)
        report["has_all_ohlcv"] = is_complete
        
        if not is_complete:
            logger.error("列缺失，终止后续检查。")
            return report
            
        # 2. 缺失值检查
        report["missing_counts"] = self.check_missing_data(df)
        
        # 3. 价格跳变
        price_jumps_df = self.check_price_jumps(df)
        report["price_jumps_count"] = len(price_jumps_df)
        report["price_jumps_sample"] = price_jumps_df.head() if not price_jumps_df.empty else None
        
        # 4. 成交量跳变
        vol_jumps_df = self.check_volume_jumps(df)
        report["volume_jumps_count"] = len(vol_jumps_df)
        report["volume_jumps_sample"] = vol_jumps_df.head() if not vol_jumps_df.empty else None
        
        logger.info("数据健康检查完成。")
        return report
