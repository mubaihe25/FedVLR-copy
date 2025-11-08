import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class JSDivergence(nn.Module):
    """Jensen-Shannon 散度实现
    
    Jensen-Shannon 散度是一种对称的散度度量，基于 KL 散度构建。
    它测量两个概率分布之间的相似性，值域为 [0, 1]，其中 0 表示分布相同。
    
    公式: JSD(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)]，其中 M = 0.5 * (P + Q)
    """

    def __init__(self, reduction: str = "batchmean", eps: float = 1e-8):
        """初始化 JS 散度模块
        
        Args:
            reduction: 归约方法，可选 "batchmean"、"sum" 或 "none"
            eps: 数值稳定性的小常数
        """
        super(JSDivergence, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, net_1_logits: torch.Tensor, net_2_logits: torch.Tensor) -> torch.Tensor:
        """计算两组 logits 之间的 JS 散度
        
        Args:
            net_1_logits: 第一个网络的 logits，形状为 [batch_size, num_classes]
            net_2_logits: 第二个网络的 logits，形状为 [batch_size, num_classes]
            
        Returns:
            两个分布之间的 JS 散度
        """
        # 检查输入维度是否匹配
        if net_1_logits.shape != net_2_logits.shape:
            raise ValueError(f"输入形状不匹配: {net_1_logits.shape} vs {net_2_logits.shape}")
            
        # 计算概率分布
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)
        
        # 计算混合分布 M = 0.5 * (P + Q)
        total_m = 0.5 * (net_1_probs + net_2_probs)
        
        # 为数值稳定性添加小常数
        total_m = torch.clamp(total_m, min=self.eps)
        
        # 计算 KL(P||M)
        kl_1 = F.kl_div(
            F.log_softmax(net_1_logits, dim=1),  # 使用 log_softmax 提高数值稳定性
            total_m,
            reduction=self.reduction
        )
        
        # 计算 KL(Q||M)
        kl_2 = F.kl_div(
            F.log_softmax(net_2_logits, dim=1),
            total_m,
            reduction=self.reduction
        )
        
        # JS 散度 = 0.5 * [KL(P||M) + KL(Q||M)]
        js_div = 0.5 * (kl_1 + kl_2)
        
        return js_div


class SymmetrizedKLDivergence(nn.Module):
    """对称化的 KL 散度实现
    
    标准 KL 散度是不对称的，这个类实现了对称版本：
    SymKL(P||Q) = 0.5 * [KL(P||Q) + KL(Q||P)]
    """
    
    def __init__(self, reduction: str = "batchmean", eps: float = 1e-8):
        """初始化对称化 KL 散度模块
        
        Args:
            reduction: 归约方法，可选 "batchmean"、"sum" 或 "none"
            eps: 数值稳定性的小常数
        """
        super(SymmetrizedKLDivergence, self).__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, net_1_logits: torch.Tensor, net_2_logits: torch.Tensor) -> torch.Tensor:
        """计算两组 logits 之间的对称化 KL 散度
        
        Args:
            net_1_logits: 第一个网络的 logits，形状为 [batch_size, num_classes]
            net_2_logits: 第二个网络的 logits，形状为 [batch_size, num_classes]
            
        Returns:
            两个分布之间的对称化 KL 散度
        """
        # 计算概率分布
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)
        
        # 为数值稳定性添加小常数
        net_1_probs = torch.clamp(net_1_probs, min=self.eps)
        net_2_probs = torch.clamp(net_2_probs, min=self.eps)
        
        # 计算 KL(P||Q)
        kl_1_2 = F.kl_div(
            torch.log(net_1_probs),
            net_2_probs,
            reduction=self.reduction
        )
        
        # 计算 KL(Q||P)
        kl_2_1 = F.kl_div(
            torch.log(net_2_probs),
            net_1_probs,
            reduction=self.reduction
        )
        
        # 对称化 KL 散度 = 0.5 * [KL(P||Q) + KL(Q||P)]
        sym_kl = 0.5 * (kl_1_2 + kl_2_1)
        
        return sym_kl


def compute_divergence(
    p_logits: torch.Tensor, 
    q_logits: torch.Tensor, 
    method: str = "js"
) -> torch.Tensor:
    """计算两个分布之间的散度
    
    Args:
        p_logits: 第一个分布的 logits
        q_logits: 第二个分布的 logits
        method: 散度方法，可选 "js"（Jensen-Shannon）或 "kl"（对称化 KL）
        
    Returns:
        计算的散度值
    """
    if method.lower() == "js":
        divergence_fn = JSDivergence()
    elif method.lower() == "kl":
        divergence_fn = SymmetrizedKLDivergence()
    else:
        raise ValueError(f"不支持的散度方法: {method}，请使用 'js' 或 'kl'")
    
    return divergence_fn(p_logits, q_logits)
