import copy
import logging
from typing import Dict, Tuple, Any, List, Optional

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class FedAvg(GeneralRecommender):
    """联邦平均推荐模型
    
    实现基于联邦学习的推荐系统，使用项目共性嵌入进行预测
    """

    def __init__(self, config, dataloader):
        super(FedAvg, self).__init__(config, dataloader)

        self.embed_size = config['latent_size']

        # 初始化项目共性嵌入
        self.item_commonality = torch.nn.Embedding(
            num_embeddings=self.n_items, 
            embedding_dim=self.embed_size
        )

        # 输出层
        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # 应用初始化
        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality: nn.Embedding) -> None:
        """设置项目共性嵌入
        
        Args:
            item_commonality: 项目共性嵌入层
        """
        # 使用load_state_dict而不是直接复制，更加稳定
        self.item_commonality.load_state_dict(item_commonality.state_dict())

    def forward(self, item_indices: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            item_indices: 项目索引张量
            
        Returns:
            预测评分
        """
        # 获取项目嵌入
        item_commonality = self.item_commonality(item_indices)

        # 计算预测
        pred = self.affine_output(item_commonality)
        rating = self.logistic(pred)

        return rating

    def full_sort_predict(self, interaction: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """全排序预测
        
        Args:
            interaction: 交互张量
            
        Returns:
            所有项目的预测分数
        """
        users = interaction[0]
        items = torch.arange(self.n_items).to(self.device)
        scores = self.forward(items)

        return scores.view(-1)


class FedAvgTrainer(FederatedTrainer):
    """联邦平均训练器
    
    实现联邦学习中的模型训练和参数聚合
    """

    def __init__(self, config, model, mg=False):
        super(FedAvgTrainer, self).__init__(config, model, mg)

        self.lr_network = self.config['lr']
        self.lr_args = self.config['lr']

        # 复制项目共性嵌入
        self.item_commonality = copy.deepcopy(model.item_commonality)
        
        # 初始化权重字典和客户端模型字典
        self.weights = {}
        self.client_models = {}
        
        # 损失函数
        self.crit = nn.BCELoss()
        self.independency = nn.MSELoss()
        self.reg = nn.L1Loss()
        
        # 获取logger
        self.logger = logging.getLogger(__name__)

    def _set_optimizer(self, model) -> torch.optim.Optimizer:
        """初始化优化器
        
        Args:
            model: 模型实例
            
        Returns:
            优化器实例
        """
        # 参数分组，可以为不同参数设置不同学习率
        param_list = [
            {'params': model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': model.item_commonality.parameters(), 'lr': self.lr_args},
        ]

        # 根据配置选择优化器
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop
        }
        
        optimizer_class = optimizer_map.get(self.learner.lower())
        if optimizer_class:
            optimizer = optimizer_class(param_list, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(model.parameters(), lr=self.lr_network)
            
        return optimizer

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)
        
        # 设置项目共性嵌入
        client_model.set_item_commonality(self.item_commonality)

        # 如果不是第一次迭代且用户模型存在，加载用户特定参数
        if iteration > 0 and user in self.client_models:
            try:
                # 加载用户特定参数
                state_dict = client_model.state_dict()
                for key, value in self.client_models[user].items():
                    if key in state_dict:
                        state_dict[key] = copy.deepcopy(value)
                client_model.load_state_dict(state_dict)
            except Exception as e:
                self.logger.error(f"Error loading client model for user {user}: {e}")
        
        # 将模型移至指定设备并创建优化器
        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        batch_data = batch.to(self.device)
        user, poss, negs = batch_data[0], batch_data[1], batch_data[2]

        # 更新用户权重
        user_id = user[0].item()
        self.weights[user_id] = self.weights.get(user_id, 0) + len(poss) + len(negs)

        # 构建评分
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        ratings[:poss.size(0)] = 1.0

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(items)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        user, client_model = args

        # 获取模型状态字典
        tmp_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        
        # 存储非公共项参数
        self.client_models[user] = {
            k: copy.deepcopy(v) for k, v in tmp_dict.items() 
            if 'item_commonality' not in k
        }
        
        # 只上传公共项参数
        upload_params = {
            k: copy.deepcopy(v) for k, v in tmp_dict.items()
            if 'item_commonality' in k
        }

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]
        
        # 检查是否有参与者
        if not participant_params:
            self.logger.warning("No participants for aggregation")
            return
        
        try:
            # 计算总权重
            total_weights = sum(self.weights.get(user, 0) for user in participant_params)
            if total_weights <= 0:
                self.logger.warning("Total weights is zero, skipping aggregation")
                return
            
            # 获取设备
            device = self.item_commonality.weight.device
            
            # 聚合项目共性嵌入
            if all('item_commonality.weight' in params for params in participant_params.values()):
                # 使用向量化操作进行加权平均
                weights = torch.tensor([self.weights.get(user, 0) / total_weights for user in participant_params], 
                                      device=device)
                
                # 将所有参数堆叠并移至正确的设备
                param_stack = torch.stack([
                    params['item_commonality.weight'].to(device) 
                    for params in participant_params.values()
                ])
                
                # 计算加权平均
                self.item_commonality.weight.data = torch.sum(
                    weights.view(-1, 1, 1) * param_stack, dim=0
                )
            else:
                # 逐个聚合参数
                aggregated_weight = None
                for user, params in participant_params.items():
                    if 'item_commonality.weight' not in params:
                        continue
                        
                    weight = self.weights.get(user, 0) / total_weights
                    param = params['item_commonality.weight'].to(device)
                    
                    if aggregated_weight is None:
                        aggregated_weight = weight * param
                    else:
                        aggregated_weight += weight * param
                
                if aggregated_weight is not None:
                    self.item_commonality.weight.data = aggregated_weight
        except Exception as e:
            self.logger.error(f"Error in parameter aggregation: {e}")

    def _update_hyperparams(self, *args, **kwargs) -> None:
        """更新超参数
        
        可以在此实现学习率调度等
        """
        pass

    def calculate_loss(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            pred: 预测值
            truth: 真实值
            
        Returns:
            损失值
        """
        return self.crit(pred, truth)
