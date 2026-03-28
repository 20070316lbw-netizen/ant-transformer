"""
ant.py — AntTransformer（完整模型）

整体结构：
    input_ids [B, T]
        │
        ▼
    Embedding [B, T, D]
        │
        ▼
    PositionalEncoding
        │
        ▼
    LayerNorm + Dropout
        │
        ▼
    AntEncoder (N × AntLayer)
        ├── 标准自注意力
        ├── 跨层注意力（小蚂蚁核心）
        └── 历史驱动门控
        ▼
    [CLS] pooling → h_cls [B, D]
        │
        ▼
    Classifier (Linear → GELU → Linear)
        │
        ▼
    logits [B, num_classes]

位置编码使用经典 sinusoidal 形式：
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
"""

import math
import torch
import torch.nn as nn

from .config import AntConfig
from .encoder import AntEncoder


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码（固定，不可学习）"""

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 构造 PE 矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(
            1
        )  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)  # 不参与梯度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AntTransformer(nn.Module):
    """
    小蚂蚁 Transformer — 序列分类版本。

    核心创新（在 AntEncoder / AntLayer 中实现）：
    1. 跨层 Attention Residual：每层 attend 所有前层隐状态
    2. 历史驱动 Gate：根据跨层汇总自适应决定是否跳过当前层
    """

    def __init__(self, config: AntConfig):
        super().__init__()
        self.config = config

        # ── 输入端 ──────────────────────────────────────────────
        if config.model_type == "financial":
            self.input_proj = nn.Linear(config.input_dim, config.d_model)
        else:
            self.embedding = nn.Embedding(
                config.vocab_size, config.d_model, padding_idx=0
            )

        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.embed_norm = nn.LayerNorm(config.d_model)
        self.embed_drop = nn.Dropout(config.dropout)

        # ── 编码器 ──────────────────────────────────────────────
        self.encoder = AntEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            cross_layer_heads=config.cross_layer_heads,
            d_ff=config.d_ff,
            gate_hidden_dim=config.gate_hidden_dim,
            dropout=config.dropout,
        )

        # ── 分类头 ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化线性层，Embedding 正态初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 原代码的 isinstance 判断包含了 nn.Linear，导致条件永远不成立
                # 现在直接匹配 nn.Embedding，无论 text 还是 financial 模式都安全
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        enable_pruning: bool | None = None,
    ) -> tuple[torch.Tensor, list, list]:
        """
        Args:
            input_ids:      [B, T] token ids
            attention_mask: [B, T] 1 = real token, 0 = padding
        Returns:
            logits:      [B, num_classes]
            all_hiddens: List[[B,T,D]] 各层隐状态（供分析）
            all_gates:   List[[B,T,D]] 各层门控值（供分析）
        """
        # padding mask: True = 忽略该位置
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # [B, T]

        # ① Embedding / Projection + 位置编码
        if self.config.model_type == "financial":
            # input_ids 此时为 [B, T, D_in]
            x = self.input_proj(input_ids)
        else:
            x = self.embedding(input_ids)  # [B, T, D]
        x = self.pos_encoding(x)
        x = self.embed_norm(x)
        x = self.embed_drop(x)

        # ② 编码
        h_final, all_hiddens, all_gates = self.encoder(
            x,
            key_padding_mask=key_padding_mask,
            enable_pruning=(
                self.config.enable_layer_pruning
                if enable_pruning is None
                else enable_pruning
            ),
        )

        # ③ [CLS] 池化（取第 0 个 token）
        cls_repr = h_final[:, 0, :]  # [B, D]

        # ④ 分类
        logits = self.classifier(cls_repr)  # [B, num_classes]

        return logits, all_hiddens, all_gates

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
