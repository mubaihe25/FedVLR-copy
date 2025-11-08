import torch
import torch.nn as nn
from typing import Tuple, Optional


class SumExpert(nn.Module):
    """求和专家：将输入的三个嵌入向量简单相加
    
    这是最简单的融合方法，直接将三个特征向量相加，不引入额外参数。
    适用于特征向量具有相同的语义空间和相似的重要性时。
    """

    def __init__(self):
        """初始化求和专家"""
        super(SumExpert, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """将三个嵌入向量相加
        
        Args:
            x: 第一个嵌入向量，形状为 [batch_size, embed_size]
            y: 第二个嵌入向量，形状为 [batch_size, embed_size]
            z: 第三个嵌入向量，形状为 [batch_size, embed_size]
            
        Returns:
            融合后的嵌入向量，形状为 [batch_size, embed_size]
        """
        # 检查输入维度是否匹配
        if not (x.shape == y.shape == z.shape):
            raise ValueError(f"输入形状不匹配: {x.shape}, {y.shape}, {z.shape}")
            
        # 直接对输入的三个嵌入向量求和
        return x + y + z


class MLPExpert(nn.Module):
    """MLP 专家：使用多层感知机将三个嵌入融合
    
    通过拼接三个特征向量，然后使用多层感知机进行非线性变换，
    可以学习更复杂的特征交互关系。
    """

    def __init__(self, embed_size: int, hidden_size: Optional[int] = None, dropout: float = 0.1):
        """初始化 MLP 专家
        
        Args:
            embed_size: 输入嵌入向量的维度
            hidden_size: 隐藏层的维度，默认为 embed_size * 2
            dropout: Dropout 概率，用于防止过拟合
        """
        super(MLPExpert, self).__init__()
        
        if hidden_size is None:
            hidden_size = embed_size * 2
            
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size)  # 添加层归一化提高稳定性
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """使用 MLP 融合三个嵌入向量
        
        Args:
            x: 第一个嵌入向量，形状为 [batch_size, embed_size]
            y: 第二个嵌入向量，形状为 [batch_size, embed_size]
            z: 第三个嵌入向量，形状为 [batch_size, embed_size]
            
        Returns:
            融合后的嵌入向量，形状为 [batch_size, embed_size]
        """
        # 将三个嵌入向量拼接并输入到 MLP 中
        concat_features = torch.cat([x, y, z], dim=-1)
        fused_features = self.mlp(concat_features)

        return fused_features


class MultiHeadAttentionExpert(nn.Module):
    """多头注意力专家：使用多头注意力机制将三个嵌入融合
    
    利用注意力机制捕捉三个特征向量之间的相互关系，
    可以根据上下文自适应地调整不同特征的重要性。
    """

    def __init__(self, embed_size: int, num_heads: int = 4, dropout: float = 0.1):
        """初始化多头注意力专家
        
        Args:
            embed_size: 输入嵌入向量的维度
            num_heads: 注意力头的数量
            dropout: Dropout 概率
        """
        super(MultiHeadAttentionExpert, self).__init__()

        # 确保 embed_size 能被 num_heads 整除
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除"
        
        self.embed_size = embed_size
        self.combined_size = embed_size * 3
        
        # 投影层，将拼接后的特征映射到适合注意力机制的维度
        self.projection = nn.Linear(self.combined_size, self.combined_size)
        
        # 多头注意力层
        self.attn = nn.MultiheadAttention(
            embed_dim=self.combined_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 设置为 True 使输入形状为 [batch_size, seq_len, embed_dim]
        )
        
        # 输出层，将注意力输出映射回原始嵌入维度
        self.fc_out = nn.Sequential(
            nn.Linear(self.combined_size, embed_size),
            nn.LayerNorm(embed_size)
        )
        
        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """使用多头注意力融合三个嵌入向量
        
        Args:
            x: 第一个嵌入向量，形状为 [batch_size, embed_size]
            y: 第二个嵌入向量，形状为 [batch_size, embed_size]
            z: 第三个嵌入向量，形状为 [batch_size, embed_size]
            
        Returns:
            融合后的嵌入向量，形状为 [batch_size, embed_size]
        """
        batch_size = x.size(0)
        
        # 将三个嵌入向量拼接
        concat_features = torch.cat([x, y, z], dim=-1)  # [batch_size, embed_size*3]
        
        # 投影
        projected = self.projection(concat_features)
        
        # 重塑为序列形式，序列长度为1
        query = key = value = projected.unsqueeze(1)  # [batch_size, 1, embed_size*3]
        
        # 应用多头注意力
        attn_output, _ = self.attn(query, key, value)  # [batch_size, 1, embed_size*3]
        
        # 去除序列维度
        attn_output = attn_output.squeeze(1)  # [batch_size, embed_size*3]
        
        # 通过输出层
        output = self.fc_out(attn_output)  # [batch_size, embed_size]
        
        return output


class GateExpert(nn.Module):
    """门控专家：使用门控机制融合三个嵌入向量
    
    为每个特征向量分配一个门控值，控制其在最终融合中的贡献度。
    门控机制允许模型动态调整不同特征的重要性。
    """

    def __init__(self, embed_size: int, hidden_size: Optional[int] = None, dropout: float = 0.1):
        """初始化门控专家
        
        Args:
            embed_size: 输入嵌入向量的维度
            hidden_size: 隐藏层和输出的维度，默认与 embed_size 相同
            dropout: Dropout 概率
        """
        super(GateExpert, self).__init__()
        
        if hidden_size is None:
            hidden_size = embed_size
            
        # ID特征的门控网络
        self.id_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        # 文本特征的门控网络
        self.txt_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        # 视觉特征的门控网络
        self.vis_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for name, p in self.named_parameters():
            if 'weight' in name:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, id_feat: torch.Tensor, txt_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """使用门控机制融合三个嵌入向量
        
        Args:
            id_feat: ID特征向量，形状为 [batch_size, embed_size]
            txt_feat: 文本特征向量，形状为 [batch_size, embed_size]
            vis_feat: 视觉特征向量，形状为 [batch_size, embed_size]
            
        Returns:
            融合后的嵌入向量，形状为 [batch_size, hidden_size]
        """
        # 应用门控网络获取各特征的门控值
        id_values = self.id_gate(id_feat)    # [batch_size, hidden_size]
        txt_values = self.txt_gate(txt_feat)  # [batch_size, hidden_size]
        vis_values = self.vis_gate(vis_feat)  # [batch_size, hidden_size]

        # 拼接门控后的特征
        gated_features = torch.cat([id_values, txt_values, vis_values], dim=1)  # [batch_size, hidden_size*3]
        
        # 融合
        output = self.fusion(gated_features)  # [batch_size, hidden_size]

        return output


class CrossAttentionExpert(nn.Module):
    """交叉注意力专家：使用交叉注意力机制融合三个嵌入向量
    
    通过让每个特征向量作为查询，其他特征向量作为键和值，
    实现特征之间的交叉注意力，捕捉更复杂的特征交互。
    """
    
    def __init__(self, embed_size: int, num_heads: int = 4, dropout: float = 0.1):
        """初始化交叉注意力专家
        
        Args:
            embed_size: 输入嵌入向量的维度
            num_heads: 注意力头的数量
            dropout: Dropout 概率
        """
        super(CrossAttentionExpert, self).__init__()
        
        # 确保 embed_size 能被 num_heads 整除
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除"
        
        # 三个交叉注意力层，每个特征作为查询时使用
        self.cross_attn_id = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_txt = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_vis = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.LayerNorm(embed_size)
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, id_feat: torch.Tensor, txt_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """使用交叉注意力融合三个嵌入向量
        
        Args:
            id_feat: ID特征向量，形状为 [batch_size, embed_size]
            txt_feat: 文本特征向量，形状为 [batch_size, embed_size]
            vis_feat: 视觉特征向量，形状为 [batch_size, embed_size]
            
        Returns:
            融合后的嵌入向量，形状为 [batch_size, embed_size]
        """
        batch_size = id_feat.size(0)
        
        # 重塑为序列形式，序列长度为1
        id_seq = id_feat.unsqueeze(1)    # [batch_size, 1, embed_size]
        txt_seq = txt_feat.unsqueeze(1)  # [batch_size, 1, embed_size]
        vis_seq = vis_feat.unsqueeze(1)  # [batch_size, 1, embed_size]
        
        # 将其他两个特征拼接为键值对
        kv_for_id = torch.cat([txt_seq, vis_seq], dim=1)   # [batch_size, 2, embed_size]
        kv_for_txt = torch.cat([id_seq, vis_seq], dim=1)   # [batch_size, 2, embed_size]
        kv_for_vis = torch.cat([id_seq, txt_seq], dim=1)   # [batch_size, 2, embed_size]
        
        # 应用交叉注意力
        id_attn_out, _ = self.cross_attn_id(id_seq, kv_for_id, kv_for_id)   # [batch_size, 1, embed_size]
        txt_attn_out, _ = self.cross_attn_txt(txt_seq, kv_for_txt, kv_for_txt)  # [batch_size, 1, embed_size]
        vis_attn_out, _ = self.cross_attn_vis(vis_seq, kv_for_vis, kv_for_vis)  # [batch_size, 1, embed_size]
        
        # 去除序列维度
        id_attn_out = id_attn_out.squeeze(1)   # [batch_size, embed_size]
        txt_attn_out = txt_attn_out.squeeze(1)  # [batch_size, embed_size]
        vis_attn_out = vis_attn_out.squeeze(1)  # [batch_size, embed_size]
        
        # 拼接注意力输出
        concat_features = torch.cat([id_attn_out, txt_attn_out, vis_attn_out], dim=1)  # [batch_size, embed_size*3]
        
        # 融合
        output = self.fusion(concat_features)  # [batch_size, embed_size]
        
        return output


def get_expert(expert_type: str, embed_size: int, **kwargs) -> nn.Module:
    """获取指定类型的专家模型
    
    Args:
        expert_type: 专家类型，可选 'sum', 'mlp', 'attention', 'gate', 'cross'
        embed_size: 嵌入向量的维度
        **kwargs: 传递给专家模型的额外参数
        
    Returns:
        专家模型实例
    """
    expert_map = {
        'sum': SumExpert,
        'mlp': MLPExpert,
        'attention': MultiHeadAttentionExpert,
        'gate': GateExpert,
        'cross': CrossAttentionExpert
    }
    
    if expert_type not in expert_map:
        raise ValueError(f"不支持的专家类型: {expert_type}，可选: {list(expert_map.keys())}")
    
    expert_class = expert_map[expert_type]
    
    if expert_type == 'sum':
        return expert_class()
    else:
        return expert_class(embed_size, **kwargs)
