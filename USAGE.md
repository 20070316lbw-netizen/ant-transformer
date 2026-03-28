# DynaRouter 使用指南

## 快速开始

### 1. 基本训练命令

使用默认配置文件训练模型：

```bash
python train.py
```

### 2. 使用配置文件

可以创建不同的配置文件进行实验：

```bash
# 完整模型
python train.py --config configs/exp_full.yaml

# 减少层数
python train.py --config configs/exp_layer0_layer2.yaml

# 只用一层
python train.py --config configs/exp_layer0.yaml
```

### 3. 命令行参数覆盖

可以在运行时覆盖配置文件中的参数：

```bash
# 覆盖模型架构
python train.py --model_arch layer0_layer2 --epochs 50

# 禁用层裁剪
python train.py --no_pruning

# 调整学习率和门控权重
python train.py --lr 0.0005 --gate_lambda 0.05

# 使用数据子集进行快速测试
python train.py --subset 10000
```

### 4. 训练三种模型

创建实验配置文件（参考 config.yaml），然后：

```bash
# 实验一：完整模型
python train.py --config configs/exp_full.yaml

# 实验二：减少层数
python train.py --config configs/exp_layer0_layer2.yaml

# 实验三：只用一层
python train.py --config configs/exp_layer0.yaml
```

### 5. 评估预测结果

训练完成后，使用评估脚本：

```bash
# 基本评估
python evaluate.py --pred_path outputs/pred_full.csv

# 跳过分月指标，只看组合回测
python evaluate.py --pred_path outputs/pred_full.csv --skip_monthly

# 跳过组合回测，只看分月指标
python evaluate.py --pred_path outputs/pred_full.csv --skip_combination

# 调整组合股票数量
python evaluate.py --pred_path outputs/pred_full.csv --top_n 20
```

## 配置文件说明

### 核心配置项

#### model 配置
```yaml
model:
  arch: "full"              # 架构：full | layer0_layer2 | layer0
  d_model: 256              # 模型维度
  num_heads: 8              # 注意力头数
  d_ff: 1024                # FFN 内层维度
  cross_layer_heads: 4      # 跨层注意力头数
  gate_hidden_dim: 64       # 门控 MLP 隐层维度
  input_dim: 6              # 输入特征维度
  enable_layer_pruning: true # 是否启用层裁剪
```

#### training 配置
```yaml
training:
  epochs: 30                # 训练轮数
  batch_size: 1024          # 批次大小
  lr: 0.001                 # 学习率
  gate_lambda: 0.08         # 门控正则化权重
  max_grad_norm: 1.0        # 梯度裁剪
  dropout: 0.1              # Dropout 概率
  warmup_steps: 500         # Warmup 步数
```

#### experiment 配置
```yaml
experiment:
  use_subset: false         # 是否使用数据子集
  subset_size: null         # 子集大小（仅 use_subset=true 时有效）
  output_prefix: "outputs"  # 输出路径前缀
```

## 实验对比

### 实验一：完整模型 (full)

```bash
python train.py --config configs/exp_full.yaml
```

**模型结构**：4层 Transformer

**预期行为**：
- 所有层都参与计算
- 门控机制自适应裁剪层

**输出**：`outputs/pred_full.csv`

---

### 实验二：减少层数 (layer0_layer2)

```bash
python train.py --config configs/exp_layer0_layer2.yaml
```

**模型结构**：2层 Transformer

**预期行为**：
- 第0层和第2层参与计算
- 第1层被裁剪

**输出**：`outputs/pred_layer0_layer2.csv`

---

### 实验三：只用一层 (layer0)

```bash
python train.py --config configs/exp_layer0.yaml
```

**模型结构**：1层 Transformer

**预期行为**：
- 只有第0层参与计算
- 第1、2、3层被裁剪

**输出**：`outputs/pred_layer0.csv`

---

## 评估对比

### 对比三种模型的性能：

```bash
# 1. 完整模型评估
python evaluate.py --pred_path outputs/pred_full.csv

# 2. 减少层数评估
python evaluate.py --pred_path outputs/pred_layer0_layer2.csv

# 3. 只用一层评估
python evaluate.py --pred_path outputs/pred_layer0.csv
```

### 评估指标说明

#### 分月指标
- **IC (Information Coefficient)**：预测值与真实值的相关系数
- **RankIC**：预测值的排序与真实值的排序的相关系数（Spearman）
- **IC IR**：IC 的均值除以标准差

#### 组合回测
- **年化超额收益**：组合超越基准的年化收益
- **年化超额波动**：超额收益的年化标准差
- **超额 Sharpe**：超额收益的年化夏普比率
- **超额 MaxDD**：最大回撤

## 参数调优建议

### 1. 调整门控权重 (gate_lambda)

```bash
# 更强的门控约束（更稀疏）
python train.py --gate_lambda 0.15

# 更弱的门控约束（更多层参与）
python train.py --gate_lambda 0.01
```

### 2. 调整学习率

```bash
# 更大的学习率
python train.py --lr 0.003

# 更小的学习率
python train.py --lr 0.0005
```

### 3. 调整层数

```bash
# 更深的网络
python train.py --model_arch full --epochs 40

# 更浅的网络
python train.py --model_arch layer0
```

## 常见问题

### Q: 如何快速测试代码是否工作？

```bash
python train.py --subset 100 --epochs 1
```

### Q: 如何禁用层裁剪？

```bash
python train.py --no_pruning
```

### Q: 预测结果保存在哪里？

默认保存在 `outputs/` 目录下，文件名根据模型架构自动命名：
- `pred_full.csv`
- `pred_layer0_layer2.csv`
- `pred_layer0.csv`

### Q: 如何调整组合回测的股票数量？

```bash
python evaluate.py --pred_path outputs/pred_full.csv --top_n 100
```

## 性能优化建议

1. **GPU 加速**：确保 CUDA 可用，修改 `advanced.use_cuda` 为 `true`
2. **多线程加载**：设置 `advanced.num_workers` 为 `4` 或 `8`
3. **批次大小**：根据 GPU 内存调整 `batch_size`

## 高级用法

### 自定义配置文件

复制 `config.yaml` 并修改：

```bash
cp config.yaml configs/my_experiment.yaml
# 编辑 configs/my_experiment.yaml
python train.py --config configs/my_experiment.yaml
```

### 批量实验

创建脚本批量运行实验：

```bash
#!/bin/bash
# run_all_experiments.sh

echo "=== 运行实验一：完整模型 ==="
python train.py --config configs/exp_full.yaml

echo "=== 运行实验二：减少层数 ==="
python train.py --config configs/exp_layer0_layer2.yaml

echo "=== 运行实验三：只用一层 ==="
python train.py --config configs/exp_layer0.yaml

echo "=== 运行实验四：禁用裁剪 ==="
python train.py --no_pruning

echo "=== 运行实验五：调整门控权重 ==="
python train.py --gate_lambda 0.05

# 评估所有结果
for pred in outputs/pred_*.csv; do
    echo "评估: $pred"
    python evaluate.py --pred_path "$pred"
done
```

运行：
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

## 项目结构

```
DynaRouter/
├── config.yaml              # 默认配置文件
├── train.py                 # 训练入口
├── evaluate.py              # 评估入口
├── model/
│   ├── config.py           # 配置类
│   ├── ant.py              # 完整模型
│   ├── encoder.py          # 编码器
│   ├── layer.py            # 单层（含裁剪逻辑）
│   ├── gate.py             # 历史门控
│   └── attention.py        # 注意力机制
├── data/
│   ├── data_prep.py        # 数据准备
│   └── financial_dataset.py # 金融数据集
└── outputs/                # 预测结果输出目录
```
