import torch
import torch.nn as nn

class PearsonCorrLoss(nn.Module):
    """
    可微皮尔逊相关系数损失函数。
    通过最小化 1 - Cor 来实现 IC 最大化。
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测值 [B] 或 [B, 1]
            target: 真实标签 [B] 或 [B, 1]
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 中心化
        v_pred = pred - torch.mean(pred)
        v_target = target - torch.mean(target)
        
        # 皮尔逊系数公式: Cov(X,Y) / (Std(X) * Std(Y))
        num = torch.sum(v_pred * v_target)
        den = torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_target ** 2))
        
        corr = num / (den + self.eps)
        
        # 损失函数为 1 - corr (corr 越接近 1 则损失越小)
        return 1 - corr
