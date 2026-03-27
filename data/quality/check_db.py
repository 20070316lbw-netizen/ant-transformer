import os
import sys
import pandas as pd
from loguru import logger

# 添加父目录到路径以便导入 quality 和 data 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db_manager import QuantDBManager
from data.quality.health_checker import DataHealthChecker

def main():
    logger.info("正在从数据库提取行情数据进行健康检查...")
    
    # 1. 连接数据库并提取因子数据
    db = QuantDBManager()
    query = "SELECT * FROM features_cn LIMIT 20000"
    
    with db.get_connection() as conn:
        df = conn.execute(query).df()
    
    if df.empty:
        logger.error("数据库 features_cn 表为空。")
        return

    # 2. 运行健康检查 (仅检查缺失值和基础统计)
    checker = DataHealthChecker()
    missing_report = checker.check_missing_data(df)
    
    # 3. 输出汇总
    print("\n" + "="*40)
    print("数据健康报告汇总")
    print("="*40)
    print(f"检查样本行数: {len(df)}")
    print(f"缺失值检查: {'通过' if sum(missing_report.values()) == 0 else '发现异常'}")
    if sum(missing_report.values()) > 0:
        for col, count in missing_report.items():
            if count > 0:
                print(f"  - {col}: {count} 个缺失")
    print("="*40)

if __name__ == "__main__":
    main()
