# DynaRouter 项目重构完成总结

## ✅ 已完成的工作

### 1. 统一配置管理系统

#### 新增配置文件
- **config.yaml**: 主配置文件，包含所有可配置参数
- **configs/exp_full.yaml**: 完整4层模型配置
- **configs/exp_layer0_layer2.yaml**: 2层模型配置
- **configs/exp_layer0.yaml**: 1层模型配置
- **configs/exp_no_pruning.yaml**: 禁用门控裁剪配置
- **configs/exp_tuned_gate.yaml**: 调优门控权重配置

#### 配置文件包含的参数
```yaml
model:
  arch: "full"              # 模型架构
  d_model: 256              # 模型维度
  num_heads: 8              # 注意力头数
  d_ff: 1024                # FFN 维度
  cross_layer_heads: 4      # 跨层注意力头数
  gate_hidden_dim: 64       # 门控 MLP 隐层
  input_dim: 6              # 输入维度
  enable_layer_pruning: true # 是否启用层裁剪

training:
  epochs: 30                # 训练轮数
  batch_size: 1024          # 批次大小
  lr: 0.001                 # 学习率
  gate_lambda: 0.08         # 门控正则化权重
  max_grad_norm: 1.0        # 梯度裁剪
  dropout: 0.1              # Dropout 概率
  warmup_steps: 500         # Warmup 步数

experiment:
  use_subset: false         # 是否使用数据子集
  subset_size: null         # 子集大小
  output_prefix: "outputs"  # 输出路径

advanced:
  use_cuda: true            # 是否使用 CUDA
  num_workers: 0            # 数据加载线程数
  save_all_hiddens: false   # 是否保存所有层
```

### 2. 增强的训练脚本 (train.py)

#### 新增功能
- ✅ 支持从 YAML 配置文件加载参数
- ✅ 支持命令行参数覆盖配置文件
- ✅ 添加 `--no_pruning` 参数禁用层裁剪
- ✅ 更好的错误处理和日志输出
- ✅ 自动识别设备（CUDA/CPU）

#### 使用示例
```bash
# 使用默认配置
python train.py

# 使用特定配置文件
python train.py --config configs/exp_full.yaml

# 命令行覆盖
python train.py --model_arch layer0_layer2 --epochs 50

# 禁用层裁剪
python train.py --no_pruning

# 快速测试
python train.py --subset 10000
```

### 3. 改进的评估脚本 (evaluate.py)

#### 新增功能
- ✅ 更好的命令行参数解析
- ✅ 支持跳过分月指标或组合回测
- ✅ 可调整组合股票数量
- ✅ 更清晰的输出格式
- ✅ 函数化设计，便于扩展

#### 使用示例
```bash
# 基本评估
python evaluate.py --pred_path outputs/pred_full.csv

# 跳过分月指标
python evaluate.py --pred_path outputs/pred_full.csv --skip_monthly

# 跳过组合回测
python evaluate.py --pred_path outputs/pred_full.csv --skip_combination

# 调整组合股票数量
python evaluate.py --pred_path outputs/pred_full.csv --top_n 20
```

### 4. 层裁剪功能实现

#### 修改的文件
- **model/config.py**: 添加 `enable_layer_pruning` 字段
- **model/config.py**: 添加 `load_from_yaml()` 方法
- **model/layer.py**: 修改 `AntLayer.forward()` 支持裁剪开关
- **model/encoder.py**: 传递裁剪开关到层
- **model/ant.py**: 传递裁剪开关到编码器
- **train.py**: 传递裁剪开关到训练循环

#### 工作原理
```python
# 当 enable_layer_pruning=True:
# - 使用历史门控自适应决定是否跳过层
# - 门控值接近0表示跳过该层
# - 门控值接近1表示执行该层

# 当 enable_layer_pruning=False:
# - 所有层都完整执行
# - 门控机制被禁用
# - 适用于对比实验
```

### 5. 批量实验脚本

#### 脚本文件
- **run_all_experiments.sh**: Linux/Mac 批处理脚本
- **run_all_experiments.bat**: Windows 批处理脚本

#### 功能
- ✅ 自动运行所有实验配置
- ✅ 错误处理和状态报告
- ✅ 自动评估所有实验结果
- ✅ 进度可视化

### 6. 文档和工具

#### 使用文档
- **USAGE.md**: 完整的使用指南
  - 快速开始
  - 配置说明
  - 实验对比
  - 参数调优
  - 常见问题
  - 高级用法

#### 测试工具
- **quick_test.py**: 快速测试脚本
  - 测试模块导入
  - 测试配置加载
  - 测试模型创建
  - 测试前向传播
  - 测试数据准备

## 🎯 实验对比方案

### 实验一：完整模型
```bash
python train.py --config configs/exp_full.yaml
# 输出: outputs/pred_full.csv
```
- 4层 Transformer
- 启用门控裁剪
- 标准配置

### 实验二：减少层数
```bash
python train.py --config configs/exp_layer0_layer2.yaml
# 输出: outputs/pred_layer0_layer2.csv
```
- 2层 Transformer
- 第1层被裁剪
- 测试门控效果

### 实验三：只用一层
```bash
python train.py --config configs/exp_layer0.yaml
# 输出: outputs/pred_layer0.csv
```
- 1层 Transformer
- 第1、2、3层被裁剪
- 测试最小模型

### 实验四：禁用裁剪
```bash
python train.py --config configs/exp_no_pruning.yaml
# 输出: outputs/pred_full_no_pruning.csv
```
- 4层 Transformer
- 禁用门控裁剪
- 对比裁剪效果

### 实验五：调优门控
```bash
python train.py --config configs/exp_tuned_gate.yaml
# 输出: outputs/pred_tuned_gate.csv
```
- 4层 Transformer
- 更弱的门控约束
- 测试不同正则化强度

## 📊 评估指标

### 分月指标
- **IC (Information Coefficient)**: 预测值与真实值的相关系数
- **RankIC**: 预测排序与真实排序的相关系数（Spearman）
- **IC IR**: IC 的均值除以标准差

### 组合回测
- **年化超额收益**: 组合超越基准的年化收益
- **年化超额波动**: 超额收益的年化标准差
- **超额 Sharpe**: 超额收益的年化夏普比率
- **超额 MaxDD**: 最大回撤

## 🚀 使用流程

### 第一次使用
```bash
# 1. 运行快速测试
python quick_test.py

# 2. 运行一个实验
python train.py --config configs/exp_full.yaml

# 3. 评估结果
python evaluate.py --pred_path outputs/pred_full.csv
```

### 批量实验
```bash
# Linux/Mac
bash run_all_experiments.sh

# Windows
run_all_experiments.bat
```

### 自定义实验
```bash
# 1. 创建新配置文件
cp config.yaml configs/my_experiment.yaml
# 编辑 configs/my_experiment.yaml

# 2. 运行实验
python train.py --config configs/my_experiment.yaml

# 3. 评估
python evaluate.py --pred_path outputs/pred_my_experiment.csv
```

## 📁 项目结构

```
DynaRouter/
├── config.yaml                   # 主配置文件
├── train.py                      # 训练入口
├── evaluate.py                   # 评估入口
├── quick_test.py                 # 快速测试
├── USAGE.md                      # 使用指南
├── run_all_experiments.sh        # 批量实验脚本 (Linux/Mac)
├── run_all_experiments.bat       # 批量实验脚本 (Windows)
├── configs/                      # 实验配置目录
│   ├── exp_full.yaml             # 完整模型配置
│   ├── exp_layer0_layer2.yaml    # 减少层数配置
│   ├── exp_layer0.yaml           # 只用一层配置
│   ├── exp_no_pruning.yaml       # 禁用裁剪配置
│   └── exp_tuned_gate.yaml       # 调优门控配置
├── model/                        # 模型模块
│   ├── config.py                 # 配置类
│   ├── ant.py                    # 完整模型
│   ├── encoder.py                # 编码器
│   ├── layer.py                  # 单层（含裁剪）
│   ├── gate.py                   # 历史门控
│   └── attention.py              # 注意力机制
├── data/                         # 数据模块
│   ├── data_prep.py              # 数据准备
│   └── financial_dataset.py      # 金融数据集
├── outputs/                      # 预测结果输出
│   ├── pred_full.csv
│   ├── pred_layer0_layer2.csv
│   └── pred_layer0.csv
└── .venv/                        # 虚拟环境
```

## ✨ 主要改进点

1. **可配置性**: 所有超参数都可以通过配置文件调整，无需修改代码
2. **实验对比**: 预设了5个实验配置，一键运行并对比结果
3. **层裁剪功能**: 新增了 `enable_layer_pruning` 参数，可以灵活控制裁剪行为
4. **评估独立**: 训练和评估分离，可以单独评估任何预测结果
5. **批处理**: 提供了批量实验脚本，自动运行所有实验
6. **文档完善**: 提供了详细的使用指南和常见问题解答
7. **测试工具**: 提供了快速测试脚本，方便验证环境配置

## 🎉 完成！

现在你可以：
- ✅ 使用配置文件管理所有实验参数
- ✅ 轻松对比不同模型架构的效果
- ✅ 灵活调整门控正则化权重
- ✅ 控制是否启用层裁剪功能
- ✅ 批量运行所有实验
- ✅ 获得详细的评估报告

开始你的实验吧！
