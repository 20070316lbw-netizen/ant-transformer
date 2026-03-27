from health_checker import DataHealthChecker
import pandas as pd
from loguru import logger

def main():
    #加载脏数据
    logger.info("加载示例脏数据...")
    df = pd.read_csv('example_dirty_data.csv')

    # 运行检查
    checker = DataHealthChecker(price_jump_threshold=0.5, volume_jump_threshold=3.0)
    report = checker.run_all_checks(df)

    # 打印报告
    print("\n" + "="*40)
    print("数据健康审查报告")
    print("="*40)
    print(f"1. OHLCV完整性: {'通过' if report.get('has_all_ohlcv') else '失败'}")
    
    print("\n2. 缺失值统计:")
    for col, count in report.get("missing_counts", {}).items():
        if count > 0:
            print(f"  - {col}: 缺失 {count} 行")
            
    print(f"\n3. 价格异常跳变 (超出50%): 共发现 {report.get('price_jumps_count', 0)} 处")
    if report.get("price_jumps_sample") is not None:
        print(report["price_jumps_sample"][["date", "stock_code", "close", "jump_ratio"]])
        
    print(f"\n4. 成交量异常跳变 (超出3倍): 共发现 {report.get('volume_jumps_count', 0)} 处")
    if report.get("volume_jumps_sample") is not None:
        print(report["volume_jumps_sample"][["date", "stock_code", "volume", "volume_ratio"]])
    print("="*40)

if __name__ == "__main__":
    main()
