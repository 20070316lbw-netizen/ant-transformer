"""
attention.py — 小蚂蚁注意力模块

包含两种注意力：
  1. StandardSelfAttention  — 标准多头自注意力（与原版 Transformer 相同）
  2. CrossLayerAttention    — 跨层注意力（小蚂蚁核心）

跨层注意力数学形式：
  给定当前层隐状态 h_cur ∈ R^{B×T×D}，
  以及前 l 层的隐状态堆叠 H_prev ∈ R^{B×(l·T)×D}：

      Q = h_cur · W_Q              [B, T,   d_k]
      K = H_prev · W_K             [B, l·T, d_k]
      V = H_prev · W_V             [B, l·T, d_v]

      CrossAttn(Q,K,V) = softmax( QK^T / sqrt(d_k) ) · V

  最后加残差 + LayerNorm 输出。
"""

import torch
import torch.nn as nn


class StandardSelfAttention(nn.Module):
    """标准多头自注意力 + Pre-LN 残差"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, T, D]
            key_padding_mask: [B, T]  True = padding token（被忽略）
        Returns:
            out: [B, T, D]
        """
        residual = x
        # Pre-LN: 先 norm 再 attention（训练更稳定）
        x_norm = self.norm(x)
        out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return residual + self.dropout(out)


class CrossLayerAttention(nn.Module):
    """
    跨层注意力：当前层 attend 所有历史层的隐状态。

    当 prev_hiddens 为空（第 0 层）时，返回全零张量，
    此时历史门控 gate 会完全依赖 FFN 输出（g → 1）。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_current: torch.Tensor,
        prev_hiddens: list,
    ) -> torch.Tensor:
        """
        Args:
            h_current:    [B, T, D]         当前层自注意力输出
            prev_hiddens: List[[B, T, D]]   所有前层隐状态（可以为空）
        Returns:
            cross_out: [B, T, D]
        """
        if len(prev_hiddens) == 0:
            return torch.zeros_like(h_current)

        # 沿序列维度拼接：[B, L*T, D]，其中 L = 已完成的层数
        H_prev = torch.cat(prev_hiddens, dim=1)

        residual = h_current
        h_norm = self.norm(h_current)
        out, _ = self.attn(
            h_norm, H_prev, H_prev,
            need_weights=False,
        )
        return residual + self.dropout(out)
