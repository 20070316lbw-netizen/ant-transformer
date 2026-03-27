"""
gate.py — 历史驱动门控（History-driven Gate）

数学形式：
    输入跨层注意力的汇总向量 cross_out ∈ R^{B×T×D}，
    计算每个 token 每个维度的门控值：
    g = sigmoid( W_2 · ReLU( W_1 · cross_out + b_1 ) + b_2 ) ∈ (0, 1)^{B×T×D}

最终输出融合 FFN 输出与跳接输入：
    h_out = g ⊙ layer_out + (1 - g) ⊙ h_input

直觉：
    - 如果历史层已经充分编码了当前 token 的信息（cross_out 丰富），
      g → 0，当前层被"跳过"，信息直接透传。
    - 如果历史层信息不足，g → 1，当前层 FFN 充分激活。
"""

import torch
import torch.nn as nn


class HistoryGate(nn.Module):

    def __init__(self, d_model: int, gate_hidden_dim: int = 64):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cross_out: torch.Tensor,
        layer_out: torch.Tensor,
        h_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cross_out: [B, T, D] 跨层注意力输出
            layer_out: [B, T, D] 当前层 FFN 输出
            h_input:   [B, T, D] 当前层输入（跳接来源）
        Returns:
            h_out:    [B, T, D] 门控融合后的隐状态
            gate_val: [B, T, D] 门控值（供可视化/分析）
        """
        gSource = self.gate_mlp(cross_out)  # [B, T, D] ∈ (0,1)
        h_out = gSource * layer_out + (1.0 - gSource) * h_input  # 软跳接
        return h_out, gSource
