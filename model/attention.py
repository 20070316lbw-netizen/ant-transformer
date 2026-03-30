"""
attention.py — 小蚂蚁注意力模块

包含两种注意力：
1. StandardSelfAttention — 标准多头自注意力（与原版 Transformer 相同）
2. CrossLayerAttention  — 跨层注意力（小蚂蚁核心）

跨层注意力数学形式：
    给定当前层隐状态 h_cur ∈ R^{B×T×D}，
    以及前 l 层的隐状态堆叠 H_prev ∈ R^{B×(l·T)×D}：
    Q = h_cur · W_Q  [B, T, d_k]
    K = H_prev · W_K [B, l·T, d_k]
    V = H_prev · W_V [B, l·T, d_v]
    CrossAttn(Q,K,V) = softmax( QK^T / sqrt(d_k) ) · V
最后加残差 + LayerNorm 输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedFreqAttention(nn.Module):
    """按 head 分组的频率感知注意力 + 轻量组间交互。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_groups: int = 4,
        dropout: float = 0.1,
        mix_coeff: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_groups != 0:
            raise ValueError("num_heads must be divisible by num_groups")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_head = d_model // num_heads
        self.mix_coeff = mix_coeff

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.group_q = nn.Linear(self.d_head, self.d_head)
        self.group_k = nn.Linear(self.d_head, self.d_head)
        self.group_v = nn.Linear(self.d_head, self.d_head)
        self.group_gate = nn.Linear(self.d_head * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            key_padding_mask: [B, T], True = ignore
        """
        B, T, D = x.shape
        H, G = self.num_heads, self.num_groups

        q = self.W_q(x).view(B, T, H, self.d_head).transpose(1, 2)  # [B,H,T,d]
        k = self.W_k(x).view(B, T, H, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,T,d]
        grouped = out.view(B, G, self.heads_per_group, T, self.d_head)

        rep = grouped.mean(dim=2)  # [B,G,T,d]
        rep_t = rep.permute(0, 2, 1, 3)  # [B,T,G,d]

        gq = self.group_q(rep_t)
        gk = self.group_k(rep_t)
        gv = self.group_v(rep_t)

        g_attn = torch.matmul(gq, gk.transpose(-2, -1)) * self.scale  # [B,T,G,G]
        g_attn = F.softmax(g_attn, dim=-1)
        cross = torch.matmul(g_attn, gv).permute(0, 2, 1, 3)  # [B,G,T,d]

        gate_input = torch.cat([rep, cross], dim=-1)
        gamma = torch.sigmoid(self.group_gate(gate_input))  # [B,G,T,1]
        rep_final = rep + gamma * cross

        rep_heads = rep_final.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        mixed = grouped + self.mix_coeff * rep_heads

        mixed = mixed.reshape(B, H, T, self.d_head).transpose(1, 2).reshape(B, T, D)
        return self.W_o(mixed)


class StandardSelfAttention(nn.Module):
    """标准多头自注意力 + Pre-LN 残差"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_grouped_freq_attention: bool = False,
        num_head_groups: int = 4,
        group_mix_coeff: float = 0.1,
    ):
        super().__init__()
        self.use_grouped_freq_attention = use_grouped_freq_attention
        if use_grouped_freq_attention:
            self.attn = GroupedFreqAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_groups=num_head_groups,
                dropout=dropout,
                mix_coeff=group_mix_coeff,
            )
        else:
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
            key_padding_mask: [B, T] True = padding token（被忽略）
        Returns:
            out: [B, T, D]
        """
        residual = x
        # Pre-LN: 先 norm 再 attention（训练更稳定）
        x_norm = self.norm(x)
        if self.use_grouped_freq_attention:
            out = self.attn(x_norm, key_padding_mask=key_padding_mask)
        else:
            out, _ = self.attn(
                x_norm,
                x_norm,
                x_norm,
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
            h_current:    [B, T, D] 当前层自注意力输出
            prev_hiddens: List[[B, T, D]] 所有前层隐状态（可以为空）
        Returns:
            cross_out:    [B, T, D]
        """
        if len(prev_hiddens) == 0:
            return torch.zeros_like(h_current)

        # 沿序列维度拼接：[B, L*T, D]，其中 L = 已完成的层数
        H_prev = torch.cat(prev_hiddens, dim=1)

        residual = h_current
        h_norm = self.norm(h_current)
        out, _ = self.attn(
            h_norm, H_prev, H_prev, need_weights=False
        )
        return residual + self.dropout(out)
