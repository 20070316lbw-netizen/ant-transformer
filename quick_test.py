"""
快速测试脚本
用于验证代码是否能正常运行
"""

import torch
import sys
import os
from model.config import AntConfig
from model.ant import AntTransformer
from data.data_prep import prepare_data
from data.financial_dataset import FinancialDataset

# 添加项目根目录
sys.path.append(os.getcwd())


def test_imports():
    """测试导入是否正常"""
    print("1. 测试导入...")
    try:
        print("   [OK] 所有模块导入成功")
    except Exception as e:
        print(f"   [FAIL] 导入失败: {e}")
        return False
    return True


def test_config():
    """测试配置加载"""
    print("\n2. 测试配置加载...")
    try:
        config = AntConfig.load_from_yaml("config.yaml")
        config.validate()
        print(f"   [OK] 配置加载成功: {config.model_arch}")
    except Exception as e:
        print(f"   [FAIL] 配置加载失败: {e}")
        return False
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n3. 测试模型创建...")
    try:
        config = AntConfig.load_from_yaml("config.yaml")
        config.validate()
        model = AntTransformer(config)
        print(
            f"   [OK] 模型创建成功: {config.model_arch}, 参数量: {model.count_parameters():,}"
        )
    except Exception as e:
        print(f"   [FAIL] 模型创建失败: {e}")
        return False
    return True


def test_forward_pass():
    """测试前向传播"""
    print("\n4. 测试前向传播...")
    try:
        config = AntConfig.load_from_yaml("config.yaml")
        config.validate()
        model = AntTransformer(config)
        device = torch.device(
            "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)

        # 创建随机输入
        batch_size = 2
        seq_len = 6
        input_ids = torch.randn(batch_size, seq_len, config.input_dim).to(device)

        # 前向传播
        with torch.no_grad():
            logits, all_hiddens, all_gates = model(input_ids)

        print(f"   [OK] 前向传播成功:")
        print(f"      - 输入形状: {input_ids.shape}")
        print(f"      - 输出形状: {logits.shape}")
        print(f"      - 隐层数量: {len(all_hiddens)}")
        print(f"      - 门控数量: {len(all_gates)}")
    except Exception as e:
        print(f"   [FAIL] 前向传播失败: {e}")
        return False
    return True


def test_data_preparation():
    """测试数据准备"""
    print("\n5. 测试数据准备...")
    try:
        train_df, val_df, test_df, features, target_col = prepare_data(
            train_end="2023-12-31", val_end="2024-12-31"
        )
        print(f"   [OK] 数据加载成功:")
        print(f"      - Train: {len(train_df):,} 样本")
        print(f"      - Val: {len(val_df):,} 样本")
        print(f"      - Test: {len(test_df):,} 样本")
        print(f"      - Features: {features}")
    except Exception as e:
        print(f"   [FAIL] 数据准备失败: {e}")
        return False
    return True


def main():
    print("=" * 60)
    print("  DynaRouter 快速测试")
    print("=" * 60)
    print()

    results = []

    results.append(("导入测试", test_imports()))
    results.append(("配置测试", test_config()))
    results.append(("模型测试", test_model_creation()))
    results.append(("前向传播", test_forward_pass()))
    results.append(("数据测试", test_data_preparation()))

    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] 通过" if result else "[FAIL] 失败"
        print(f"  {name}: {status}")

    print()
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("\n[SUCCESS] 所有测试通过！可以开始训练模型了。")
        return 0
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
