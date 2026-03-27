# Ant Transformer (小蚂蚁 Transformer)

这是一个实验性的 Transformer 变体，引入了**跨层注意力残差 (Cross-Layer Attention Residual)** 和**历史驱动门控 (History-driven Gating)** 机制。

## 核心理念
传统 Transformer 的层与层之间仅通过简单的残差连接。Ant Transformer 允许每一层：
1. **回顾历史**：通过注意力机制直接访问之前所有层的隐藏状态。
2. **动态路由**：利用门控机制自适应地决定当前层的信息权重。

## 快速开始

### 1. 环境准备
```bash
pip install torch datasets transformers tqdm
```

### 2. 运行测试 (使用虚拟数据)
```bash
python train.py --use_dummy_data
```

### 3. 在 SST-2 数据集上训练
```bash
python train.py
```

## 项目结构
- `model/`: 模型核心架构
- `data/`: 数据加载与预处理
- `config.py`: 超参数配置
- `train.py`: 训练入口
- `evaluate.py`: 评估脚本
