from dataclasses import dataclass

@dataclass
class AntConfig:
    # ── 模型维度 ────────────────────────────────────────────────
    vocab_size: int = 30522      # bert-base-uncased 词表大小
    max_seq_len: int = 128
    d_model: int = 256           # 隐层维度
    num_heads: int = 8           # 自注意力头数
    num_layers: int = 6          # Transformer 层数
    d_ff: int = 1024             # FFN 内层维度

    # ── 小蚂蚁专属超参 ──────────────────────────────────────────
    cross_layer_heads: int = 4   # 跨层 Attention 头数
    gate_hidden_dim: int = 64    # 历史门控 MLP 隐层

    # ── 正则化 ──────────────────────────────────────────────────
    dropout: float = 0.1

    # ── 任务 ────────────────────────────────────────────────────
    num_classes: int = 2         # SST-2: positive / negative

    # ── 训练 ────────────────────────────────────────────────────
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_steps: int = 500

    # ── 数据 ────────────────────────────────────────────────────
    tokenizer_name: str = "bert-base-uncased"
    checkpoint_path: str = "ant_best.pt"
    use_dummy_data: bool = False
