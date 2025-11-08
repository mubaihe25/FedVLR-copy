import copy
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.experts import SumExpert, MLPExpert, MultiHeadAttentionExpert, GateExpert, get_expert


class GatingNetwork(nn.Module):
    """门控网络：为多专家系统提供权重分配
    
    根据输入特征计算每个专家的权重，实现动态路由机制。
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, latent_dim: int = 128):
        """初始化门控网络
        
        Args:
            in_dim: 输入特征维度
            out_dim: 输出维度（专家数量）
            dropout: Dropout 概率
            latent_dim: 隐藏层维度
        """
        super(GatingNetwork, self).__init__()

        # 两层前馈网络
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算专家权重
        
        Args:
            x: 输入特征，形状为 [batch_size, in_dim]
            
        Returns:
            专家权重，形状为 [1, out_dim]
        """
        # 前向传播
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # 使用 softmax 确保权重和为 1
        weights = F.softmax(out, dim=1)

        # 如果是批处理输入，计算平均权重
        if len(weights.size()) > 1 and weights.size(0) > 1:
            weights = torch.mean(weights, dim=0)
            
        # 确保输出形状为 [1, out_dim]
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
            
        return weights


class TopKModule(torch.autograd.Function):
    """自定义 TopK 操作，支持反向传播
    
    在前向传播中选择 top-k 个元素，在反向传播中正确传递梯度。
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """选择 top-k 个分数和对应的索引
        
        Args:
            ctx: 上下文对象，用于存储反向传播所需的张量
            scores: 输入分数，形状为 [batch_size, num_items]
            topk: 要选择的元素数量
            
        Returns:
            top-k 分数和对应的索引
        """
        # 选择 top-k 个分数和索引
        top_n_scores, top_n_indices = torch.topk(scores, topk, dim=1)
        
        # 保存反向传播所需的张量
        ctx.save_for_backward(scores, top_n_scores, top_n_indices)
        ctx.topk = topk
        
        return top_n_scores, top_n_indices

    @staticmethod
    def backward(ctx, grad_top_n_scores: torch.Tensor, grad_top_n_indices: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """反向传播梯度
        
        Args:
            ctx: 上下文对象
            grad_top_n_scores: top-k 分数的梯度
            grad_top_n_indices: top-k 索引的梯度（通常不使用）
            
        Returns:
            原始分数的梯度和 None（因为 topk 是整数参数）
        """
        scores, top_n_scores, top_n_indices = ctx.saved_tensors
        
        # 创建与原始分数相同形状的零张量
        grad_scores = torch.zeros_like(scores)
        
        # 将 top-k 分数的梯度分配到对应位置
        grad_scores.scatter_(1, top_n_indices, grad_top_n_scores)
        
        return grad_scores, None


class SwitchingFusionModule(nn.Module):
    """切换融合模块：动态选择最适合的专家
    
    结合多个专家模型，通过门控网络动态分配权重，实现混合专家系统。
    """

    def __init__(self, in_dim: int, embed_dim: int, dropout: float = 0.1, latent_dim: int = 128):
        """初始化切换融合模块
        
        Args:
            in_dim: 输入特征维度
            embed_dim: 嵌入向量维度
            dropout: Dropout 概率
            latent_dim: 隐藏层维度
        """
        super(SwitchingFusionModule, self).__init__()

        # 门控网络，用于分配专家权重
        self.router = GatingNetwork(embed_dim * 3, 3, dropout, latent_dim)

        # 专家模块列表
        self.experts = nn.ModuleList([
            SumExpert(),                         # 求和专家
            MLPExpert(embed_dim),                # MLP 专家
            GateExpert(embed_dim, embed_dim)     # 门控专家
            # 可以添加更多专家，如 MultiHeadAttentionExpert(embed_dim, 8)
        ])
        
        # 记录专家数量
        self.num_experts = len(self.experts)
        
        # 用于调试的索引
        self.idx = -1

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """融合三个特征向量
        
        Args:
            x: 第一个特征向量，形状为 [batch_size, embed_dim]
            y: 第二个特征向量，形状为 [batch_size, embed_dim]
            z: 第三个特征向量，形状为 [batch_size, embed_dim]
            
        Returns:
            融合后的特征向量，形状为 [batch_size, embed_dim]
        """
        # 将输入送入每个专家模块
        expert_outputs = [expert(x, y, z) for expert in self.experts]
        
        # 拼接所有专家输出
        combined_output = torch.cat(expert_outputs, dim=1)  # [batch_size, embed_dim*3]
        
        # 使用路由器计算专家权重
        weights = self.router(combined_output)  # [1, num_experts]
        
        # 加权组合专家输出
        output = torch.zeros_like(x)
        for i, expert_output in enumerate(expert_outputs):
            output += weights[0, i] * expert_output
            
        return output


class FusionLayer(nn.Module):
    """融合层：将多模态特征融合为统一表示
    
    首先将各模态特征映射到相同的潜在空间，然后使用指定的融合模块进行融合。
    """

    def __init__(self, in_dim: int, fusion_module: str = 'moe', latent_dim: int = 128):
        """初始化融合层
        
        Args:
            in_dim: 输入特征维度
            fusion_module: 融合模块类型，可选 'moe', 'sum', 'mlp', 'attention', 'gate', 'cross'
            latent_dim: 潜在空间维度
        """
        super(FusionLayer, self).__init__()

        # 特征映射层，将各模态特征映射到相同维度
        self.id_affine = nn.Linear(in_dim, latent_dim)
        self.txt_affine = nn.Linear(in_dim, latent_dim)
        self.vis_affine = nn.Linear(in_dim, latent_dim)
        
        # 特征归一化层
        self.id_norm = nn.LayerNorm(latent_dim)
        self.txt_norm = nn.LayerNorm(latent_dim)
        self.vis_norm = nn.LayerNorm(latent_dim)

        # 根据指定类型选择融合模块
        if fusion_module == 'moe':
            self.fusion = SwitchingFusionModule(latent_dim, latent_dim, dropout=0.1, latent_dim=latent_dim)
        elif fusion_module in ['sum', 'mlp', 'attention', 'gate', 'cross']:
            self.fusion = get_expert(fusion_module, latent_dim)
        else:
            raise ValueError(f'Invalid fusion module: {fusion_module}, currently support: '
                            f'moe, sum, mlp, attention, gate, cross')
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for name, p in self.named_parameters():
            if 'weight' in name:
                if p.dim() >= 2:  # 只对维度大于等于2的张量应用xavier初始化
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.01)  # 对1维张量使用正态分布初始化
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, id_feat: torch.Tensor, txt_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """融合多模态特征
        
        Args:
            id_feat: ID特征，形状为 [batch_size, in_dim]
            txt_feat: 文本特征，形状为 [batch_size, in_dim]
            vis_feat: 视觉特征，形状为 [batch_size, in_dim]
            
        Returns:
            融合后的特征，形状为 [batch_size, latent_dim]
        """
        # 特征映射和归一化
        id_feat = self.id_norm(self.id_affine(id_feat))
        txt_feat = self.txt_norm(self.txt_affine(txt_feat))
        vis_feat = self.vis_norm(self.vis_affine(vis_feat))

        # 特征融合
        return self.fusion(id_feat, txt_feat, vis_feat)


class MR(GeneralRecommender):
    """多模态推荐模型
    
    结合 ID、文本和视觉特征进行推荐，使用可配置的融合模块。
    """

    def __init__(self, config, dataloader):
        """初始化多模态推荐模型
        
        Args:
            config: 配置字典
            dataloader: 数据加载器
        """
        super(MR, self).__init__(config, dataloader)
        self.config = config

        # 项目 ID 嵌入
        self.id_embed = nn.Embedding(self.n_items, config['embedding_size'])
        
        # 融合层
        self.fusion = FusionLayer(
            in_dim=config['embedding_size'], 
            fusion_module=config['fusion_module'],
            latent_dim=config['latent_size']
        )

        # 预测层
        self.predictor = nn.Linear(config['latent_size'], 1)
        self.logistic = nn.Sigmoid()
        
        # 缓存所有项目的 ID
        self.register_buffer('item_pool', torch.arange(self.n_items))

        # 应用初始化
        self.apply(xavier_normal_initialization)
        
    def set_id_embed(self, id_embed: nn.Embedding) -> None:
        """设置 ID 嵌入层
        
        Args:
            id_embed: 预训练的 ID 嵌入层
        """
        self.id_embed.weight.data.copy_(id_embed.weight.data)

    def forward(self, item_indices: torch.Tensor, txt_embed: torch.Tensor, vision_embed: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            item_indices: 项目索引，形状为 [batch_size]
            txt_embed: 所有项目的文本嵌入，形状为 [n_items, embedding_size]
            vision_embed: 所有项目的视觉嵌入，形状为 [n_items, embedding_size]
            
        Returns:
            预测分数，形状为 [batch_size, 1]
        """
        # 获取项目的各模态特征
        id_embed = self.id_embed(item_indices)
        txt = txt_embed[item_indices]
        vision = vision_embed[item_indices]

        # 特征融合
        out = self.fusion(id_embed, txt, vision)
        
        # 预测分数
        out = self.predictor(out)
        out = self.logistic(out)

        return out

    def full_sort_predict(self, interaction: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """全排序预测
        
        Args:
            interaction: 交互张量
            *args: 额外参数，包括文本嵌入和视觉嵌入
            
        Returns:
            所有项目的预测分数，形状为 [n_items]
        """
        txt_embed, vis_embed = args[0], args[1]

        # 确保项目索引在正确的设备上
        items = torch.arange(self.n_items).to(self.device)

        # 计算所有项目的分数
        scores = self.forward(items, txt_embed, vis_embed)

        return scores.view(-1)
