# DynaRouter - 动态路由金融预测模型

这是一个基于 Ant Transformer 的金融预测模型，引入了**跨层注意力残差** 和**历史驱动门控** 机制，通过门控机制自适应地决定是否跳过某些层，实现模型结构的动态调整。

## 核心特性

1. **可配置架构**：通过配置文件灵活控制模型结构和训练参数
2. **动态路由**：历史门控机制自适应决定层是否执行
3. **实验对比**：预设多种实验配置，一键运行对比
4. **完整评估**：按月 IC/RankIC 计算 + 组合回测

## 快速开始

### 1. 环境准备
```bash
pip install torch datasets transformers tqdm loguru duckdb pandas numpy
```

### 2. 运行快速测试
```bash
python quick_test.py
```

### 3. 运行实验

#### 使用默认配置
```bash
python train.py
```

#### 运行多个实验对比
```bash
# Linux/Mac
bash run_all_experiments.sh

# Windows
run_all_experiments.bat
```

#### 自定义实验
```bash
# 使用特定配置文件
python train.py --config configs/exp_full.yaml

# 命令行参数覆盖
python train.py --model_arch layer0_layer2 --epochs 50 --gate_lambda 0.05
```

## 项目结构

```
DynaRouter/
├── config.yaml                  # 主配置文件
├── train.py                     # 训练入口
├── evaluate.py                  # 评估入口
├── quick_test.py                # 快速测试
├── USAGE.md                     # 详细使用指南
├── run_all_experiments.sh       # 批量实验脚本
├── run_all_experiments.bat      # 批量实验脚本
├── configs/                     # 实验配置
│   ├── exp_full.yaml            # 完整模型
│   ├── exp_layer0_layer2.yaml   # 减少层数
│   ├── exp_layer0.yaml          # 只用一层
│   ├── exp_no_pruning.yaml      # 禁用裁剪
│   └── exp_tuned_gate.yaml      # 调优门控
├── model/                       # 模型模块
│   ├── config.py                # 配置类
│   ├── ant.py                   # 完整模型
│   ├── encoder.py               # 编码器
│   ├── layer.py                 # 单层（含裁剪）
│   ├── gate.py                  # 历史门控
│   └── attention.py             # 注意力机制
├── data/                        # 数据模块
│   ├── data_prep.py             # 数据准备
│   └── financial_dataset.py     # 金融数据集
└── outputs/                     # 预测结果
```

## 配置说明

### 模型配置
```yaml
model:
  arch: "full"                  # full | layer0_layer2 | layer0
  d_model: 256
  num_heads: 8
  enable_layer_pruning: true    # 是否启用层裁剪
```

### 训练配置
```yaml
training:
  epochs: 30
  batch_size: 1024
  lr: 0.001
  gate_lambda: 0.08             # 门控正则化权重
```

## 实验对比

### 实验一：完整模型 (full)
4层 Transformer，包含所有层和门控机制

### 实验二：减少层数 (layer0_layer2)
2层 Transformer，第1层被门控裁剪

### 实验三：只用一层 (layer0)
1层 Transformer，其他层被完全裁剪

### 实验四：禁用裁剪 (no_pruning)
完整4层 Transformer，但禁用门控机制

### 实验五：调优门控 (tuned_gate)
调整门控权重，测试不同正则化强度

## 评估指标

### 分月指标
- **IC**: 预测值与真实值的相关系数
- **RankIC**: Spearman 相关系数
- **IC IR**: IC 的均值除以标准差

### 组合回测
- **年化超额收益**: 组合超越基准的年化收益
- **年化超额波动**: 超额收益的年化标准差
- **超额 Sharpe**: 超额收益的年化夏普比率
- **超额 MaxDD**: 最大回撤

## 常用命令

```bash
# 训练模型
python train.py --config configs/exp_full.yaml

# 禁用层裁剪
python train.py --no_pruning

# 使用数据子集进行快速测试
python train.py --subset 10000

# 评估预测结果
python evaluate.py --pred_path outputs/pred_full.csv

# 跳过分月指标，只看组合回测
python evaluate.py --pred_path outputs/pred_full.csv --skip_monthly

# 调整组合股票数量
python evaluate.py --pred_path outputs/pred_full.csv --top_n 20
```

## 详细文档

请参考 [USAGE.md](USAGE.md) 获取完整的使用指南，包括：
- 配置文件详细说明
- 实验对比方案
- 参数调优建议
- 常见问题解答
- 高级用法

## 核心理念

传统 Transformer 的层与层之间仅通过简单的残差连接。DynaRouter 允许每一层：
1. **回顾历史**：通过注意力机制直接访问之前所有层的隐藏状态
2. **动态路由**：利用历史门控机制自适应地决定当前层的信息权重

## 许可证

MIT License
