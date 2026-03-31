# 🚀 Kaggle 快速入门 / Quick Start on Kaggle

本文档是在 Kaggle 环境中运行 **Ant-Transformer** 的完整命令参考手册。  
所有命令均可直接复制执行。

> **运行环境假设**
> - GPU：T4 × 2（双卡自动启用 `DataParallel`）
> - Python：3.11
> - 数据：`data/quant_lab.duckdb`（需提前挂载为 Kaggle Dataset）

---

## 一、环境安装

```bash
pip install -r requirements.txt
```

---

## 二、烟雾测试（< 5 分钟）

快速验证整条链路是否跑通，不拿来看指标。

```bash
# 步骤 1：训练（单层，2 轮，IC Loss）
python train.py --model_arch layer0 --epochs 2 --loss_type ic --seed 42

# 步骤 2：评估
python evaluate.py --pred_path outputs/pred_layer0.csv
```

预期输出：看到 IC / Rank IC / Sharpe 三行数字即为成功。

---

## 三、IC Loss 正式训练

### Layer0（单层，快，适合验证方向）

```bash
python train.py --model_arch layer0 --epochs 20 --loss_type ic --seed 42
python evaluate.py --pred_path outputs/pred_layer0.csv
```

### Layer0 + Layer2（两层裁剪版）

```bash
python train.py --model_arch layer0_layer2 --epochs 20 --loss_type ic --seed 42
python evaluate.py --pred_path outputs/pred_layer0_layer2.csv
```

### Full（完整四层模型）

```bash
python train.py --model_arch full --epochs 30 --loss_type ic --seed 42
python evaluate.py --pred_path outputs/pred_full.csv
```

---

## 四、MSE Loss 对照实验

```bash
python train.py --model_arch layer0 --epochs 20 --loss_type mse --seed 42
python evaluate.py --pred_path outputs/pred_layer0.csv
```

---

## 五、多 Seed 稳定性测试

```bash
# Seed 42
python train.py --model_arch layer0 --epochs 20 --loss_type ic --seed 42
python evaluate.py --pred_path outputs/pred_layer0.csv

# Seed 123
python train.py --model_arch layer0 --epochs 20 --loss_type ic --seed 123
python evaluate.py --pred_path outputs/pred_layer0.csv

# Seed 2026
python train.py --model_arch layer0 --epochs 20 --loss_type ic --seed 2026
python evaluate.py --pred_path outputs/pred_layer0.csv
```

🎯 目标：三个 Seed 的 Rank IC 均 > 0，Sharpe 均 > 0.7。

---

## 六、Ablation 实验（使用预定义配置）

```bash
# 基准（无分组频率注意力）
python train.py --config configs/ablation/ic_baseline.yaml

# 分组注意力 G=2
python train.py --config configs/ablation/ic_grouped_g2.yaml

# 分组注意力 G=4
python train.py --config configs/ablation/ic_grouped_g4.yaml
```

---

## 七、禁用层裁剪（Gate）

```bash
python train.py --model_arch full --epochs 20 --loss_type ic --no_pruning
python evaluate.py --pred_path outputs/pred_full.csv
```

---

## 八、常用参数速查

| 参数 | 可选值 | 说明 |
| :--- | :--- | :--- |
| `--model_arch` | `layer0` / `layer0_layer2` / `full` | 模型架构 |
| `--epochs` | 整数 | 训练轮数 |
| `--loss_type` | `mse` / `ic` | 损失函数 |
| `--seed` | 整数 | 随机种子 |
| `--no_pruning` | 开关 | 禁用门控剪枝 |
| `--batch_size` | 整数 | 批大小（默认 1024） |
| `--lr` | 浮点数 | 学习率（默认 0.001） |
| `--config` | YAML 路径 | 使用配置文件（覆盖默认值） |

---

## 九、GPU 说明

- 单 GPU：自动使用 `cuda:0`，无需额外设置。
- 双 T4（Kaggle P100 / T4 × 2）：代码自动检测并启用 `DataParallel`，训练日志会显示：
  ```
  INFO | 检测到 2 张 GPU，启用 DataParallel
  ```
- 若强制 CPU 运行，修改 `config.yaml` 中的 `use_cuda: false`。

---

## 十、输出文件说明

```
outputs/
├── pred_layer0.csv          # Layer0 验证集预测结果
├── pred_layer0_layer2.csv   # Layer0+2 验证集预测结果
├── pred_full.csv            # Full 模型验证集预测结果
└── ablation/
    ├── ic_baseline/
    ├── ic_grouped_g2/
    └── ic_grouped_g4/
```

---

*本文档由 Ant-Transformer 项目自动生成，与 USAGE.md、README.md 互为补充。*
