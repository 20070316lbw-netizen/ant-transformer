from dataclasses import dataclass

@dataclass
class AntConfig:
    # ── 基本架构 ────────────────────────────────────────────────
    model_type: str = "text"     # "text" | "financial"
    input_dim: int = 6           # 仅用于 financial 模式
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
    gate_lambda: float = 0.01    # 门控 L1 正则权重（鼓励稀疏/跳过层）

    # ── 数据 ────────────────────────────────────────────────────
    tokenizer_name: str = "bert-base-uncased"
    checkpoint_path: str = "ant_best.pt"
    use_dummy_data: bool = False
    subset_size: int = None     # 可选：限制数据集大小以加速实验

    def validate(self):
        """执行硬性配置校验，避免静默错误"""
        # 1. Attention 分头校验
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model({self.d_model}) 必须能被 num_heads({self.num_heads}) 整除")
        if self.d_model % self.cross_layer_heads != 0:
            raise ValueError(f"d_model({self.d_model}) 必须能被 cross_layer_heads({self.cross_layer_heads}) 整除")
        
        # 2. 模式校验
        if self.model_type not in ["text", "financial"]:
            raise ValueError(f"不支持的 model_type: {self.model_type}")
            
        # 3. 维度正数校验
        if any(v <= 0 for v in [self.d_model, self.num_layers, self.num_heads, self.max_seq_len]):
            raise ValueError("维度、层数、头数和序列长度必须为正数")

        print(">>> [Config] 校验通过：架构参数合法。")
