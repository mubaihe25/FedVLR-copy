import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class FCF(GeneralRecommender):
    """
    联邦协同过滤模型(Federated Collaborative Filtering)
    实现了一个基于物品共性特征的推荐系统
    """

    def __init__(self, config, dataloader):
        super(FCF, self).__init__(config, dataloader)

        # 嵌入维度大小
        self.embed_size = config['latent_size']

        # 物品共性特征嵌入层
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)

        # 输出层：将嵌入转换为评分
        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        # Sigmoid函数将输出转换为0-1之间的值
        self.logistic = torch.nn.Sigmoid()

        # 初始化模型参数
        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality):
        self.item_commonality.load_state_dict(item_commonality.state_dict())
        # self.item_commonality.freeze = True

    def forward(self, item_indices):
        """前向传播，预测物品评分"""
        # 获取物品共性特征嵌入
        item_commonality = self.item_commonality(item_indices)
        
        # 通过线性层和sigmoid激活函数得到预测评分
        pred = self.affine_output(item_commonality)
        rating = self.logistic(pred)

        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        """为所有物品预测评分"""
        # 注意：这里没有使用用户信息，因为模型依赖的是物品共性
        users = interaction[0]  # 此参数在当前实现中未使用
        items = torch.arange(self.n_items).to(self.device)
        scores = self.forward(items)

        return scores.view(-1)


class FCFTrainer(FederatedTrainer):
    """联邦协同过滤模型的训练器，实现联邦学习算法"""

    def __init__(self, config, model, mg=False):
        super(FCFTrainer, self).__init__(config, model, mg)

        # 网络参数和物品参数的学习率
        self.lr_network = self.config['lr']
        self.lr_args = self.config['lr'] * model.n_items

        # 存储物品共性特征的全局副本
        self.item_commonality = copy.deepcopy(model.item_commonality)

        # 损失函数
        self.crit = nn.BCELoss()  # 二分类交叉熵损失
        self.independency = nn.MSELoss()  # 均方误差损失（未使用）
        self.reg = nn.L1Loss()  # L1正则化损失（未使用）

    def _set_optimizer(self, model):
        """初始化优化器，为不同参数设置不同学习率"""
        param_list = [
            {'params': model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': model.item_commonality.parameters(), 'lr': self.lr_args},
        ]

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(param_list, weight_decay=self.weight_decay)
        else:
            self.logger.warning('未识别的优化器，使用默认的Adam优化器')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _set_client(self, *args, **kwargs):
        """为特定用户设置客户端模型"""
        user, iteration = args

        # 创建模型副本
        client_model = copy.deepcopy(self.model)
        client_model.set_item_commonality(self.item_commonality)

        # 如果不是第一次迭代且有之前保存的客户端模型，则加载之前的参数
        if iteration != 0 and user in self.client_models.keys():
            for key in self.client_models[user].keys():
                client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])
            client_model.set_item_commonality(self.item_commonality)

        # 将模型移至指定设备并设置优化器
        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        """训练一个批次的数据"""
        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        
        # 构建正负样本的评分标签
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[:poss.size(0)] = 1  # 正样本评分为1，负样本为0
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        # 训练步骤
        optimizer.zero_grad()
        pred = model(items)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """存储客户端模型并准备上传参数"""
        user, client_model = args

        # 复制客户端模型参数
        tmp_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        
        # 保存客户端模型（除物品共性特征外的部分）
        self.client_models[user] = copy.deepcopy(tmp_dict)
        for key in tmp_dict.keys():
            if 'item_commonality' in key:
                del self.client_models[user][key]

        # 准备上传参数（只包含物品共性特征）
        upload_params = copy.deepcopy(tmp_dict)
        for key in list(upload_params.keys()):
            if 'item_commonality' not in key:
                del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """聚合多个客户端的参数"""
        participant_params = args[0]
        
        # 检查是否有参与者
        if not participant_params:
            return
            
        # 使用更简洁的聚合方式
        users = list(participant_params.keys())
        
        # 初始化聚合张量为零
        aggregated_weights = torch.zeros_like(self.item_commonality.weight.data)
        
        # 累加所有用户的参数，确保设备一致性
        for user in users:
            # 将客户端参数移动到服务器模型所在的设备
            client_weights = participant_params[user]['item_commonality.weight'].data.to(self.item_commonality.weight.device)
            aggregated_weights += client_weights
            
        # 计算平均值
        self.item_commonality.weight.data = aggregated_weights / len(users)

    def _update_hyperparams(self, *args, **kwargs):
        """更新超参数（当前未实现）"""
        pass

    def calculate_loss(self, interaction, *args, **kwargs):
        """计算损失函数"""
        pred, truth = interaction, args[0]
        
        # 使用二分类交叉熵损失
        loss = self.crit(pred, truth)

        return loss
