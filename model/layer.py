"""
layer.py — 小蚂蚁单层（AntLayer）

一层完整的前向流程：

    输入 h_input [B, T, D]
        │
        ▼
    [1] StandardSelfAttention   → sa_out  [B, T, D]
        │
        ▼
    [2] CrossLayerAttention     → cross_out [B, T, D]
        │  （attend 到所有 prev_hiddens）
        ▼
    [3] Combined = LayerNorm(sa_out + cross_out)
        │
        ▼
    [4] FeedForward(Combined)   → ffn_out [B, T, D]
        │
        ▼
    [5] HistoryGate             → h_out [B, T, D]
        │  g = f(cross_out)
        │  h_out = g * ffn_out + (1-g) * h_input
        ▼
    输出 h_out, gate_val
"""

import torch
import torch.nn as nn

from .attention import StandardSelfAttention, CrossLayerAttention
from .gate import HistoryGate


class FeedForward(nn.Module):
    """带 Pre-LN 残差的 FFN（GELU 激活）"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class AntLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        cross_layer_heads: int,
        d_ff: int,
        gate_hidden_dim: int,
        dropout: float = 0.1,
        use_grouped_freq_attention: bool = False,
        num_head_groups: int = 4,
        group_mix_coeff: float = 0.1,
    ):
        super().__init__()
        self.self_attn = StandardSelfAttention(
            d_model,
            num_heads,
            dropout,
            use_grouped_freq_attention=use_grouped_freq_attention,
            num_head_groups=num_head_groups,
            group_mix_coeff=group_mix_coeff,
        )
        self.cross_attn = CrossLayerAttention(d_model, cross_layer_heads, dropout)
        self.combine_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.gate = HistoryGate(d_model, gate_hidden_dim)

    def forward(
        self,
        h_input: torch.Tensor,
        prev_hiddens: list,
        key_padding_mask: torch.Tensor = None,
        enable_pruning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_input:          [B, T, D]
            prev_hiddens:     List[[B, T, D]]  前所有层输出（可为空）
            key_padding_mask: [B, T]  True = padding
            enable_pruning:   bool  是否启用层裁剪功能
        Returns:
            h_out:    [B, T, D]
            gate_val: [B, T, D]
        """
        # ① 标准自注意力
        sa_out = self.self_attn(h_input, key_padding_mask=key_padding_mask)

        # 兼容消融实验：支持关闭跨层注意力
        use_cross_layer = getattr(self, "use_cross_layer", True)

        # ② 跨层注意力（attend 历史）
        if use_cross_layer:
            cross_out = self.cross_attn(sa_out, prev_hiddens)
            combined = self.combine_norm(sa_out + cross_out)
        else:
            cross_out = torch.zeros_like(sa_out)
            combined = self.combine_norm(sa_out)

        # ③ 融合后经 FFN
        ffn_out = self.ffn(combined)

        # ④ 历史驱动门控（如果启用）
        # 兼容消融实验：支持关闭历史门控（软门控）
        use_soft_gating = getattr(self, "use_soft_gating", True)
        if enable_pruning and use_soft_gating:
            h_out, gate_val = self.gate(cross_out, ffn_out, h_input)
        else:
            # 不启用裁剪或门控，直接使用 FFN 输出 + 残差
            h_out = ffn_out + h_input  # 补充传统 Transformer 的残差连接
            gate_val = torch.ones_like(h_input)  # 全开门控值

        return h_out, gate_val
