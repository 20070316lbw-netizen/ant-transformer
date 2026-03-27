"""
encoder.py — AntEncoder

将 N 个 AntLayer 堆叠，并在前向过程中动态维护
all_hiddens 列表，实现跨层注意力的"历史记忆"。

关键设计：
  - 第 0 层：prev_hiddens = []，CrossLayerAttention 返回全零
  - 第 l 层：prev_hiddens = [h_0, h_1, ..., h_{l-1}]
  - 每层完成后将输出 append 进列表，传递给下一层
"""

import torch
import torch.nn as nn

from .layer import AntLayer


class AntEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        cross_layer_heads: int,
        d_ff: int,
        gate_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AntLayer(d_model, num_heads, cross_layer_heads, d_ff, gate_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list, list]:
        """
        Args:
            x:                [B, T, D]  嵌入后的输入
            key_padding_mask: [B, T]  True = padding
        Returns:
            h_final:     [B, T, D]        最后一层输出
            all_hiddens: List[[B,T,D]]    所有层输出（供分析）
            all_gates:   List[[B,T,D]]    所有门控值（供分析）
        """
        all_hiddens: list[torch.Tensor] = []
        all_gates:   list[torch.Tensor] = []

        h = x
        for layer in self.layers:
            h, gate_val = layer(h, all_hiddens, key_padding_mask=key_padding_mask)
            all_hiddens.append(h)
            all_gates.append(gate_val)

        return h, all_hiddens, all_gates
