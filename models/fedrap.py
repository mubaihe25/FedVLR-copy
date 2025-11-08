import copy
import math

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class FedRAP(GeneralRecommender):
    """
    联邦推荐与个性化模型 (Federated Recommendation with Personalization)
    
    该模型将物品表示分为共性部分和个性化部分，通过联邦学习方式进行训练
    """

    def __init__(self, config, dataloader):
        """
        初始化FedRAP模型
        
        Args:
            config: 模型配置参数
            dataloader: 数据加载器
        """
        super(FedRAP, self).__init__(config, dataloader)

        self.embed_size = config['latent_size']

        # 物品个性化嵌入
        self.item_personality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)
        # 物品共性嵌入
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)

        # 输出层
        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # 初始化参数
        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality):
        """
        设置物品共性嵌入
        
        Args:
            item_commonality: 物品共性嵌入层
        """
        self.item_commonality.load_state_dict(item_commonality.state_dict())
        # self.item_commonality.freeze = True

    def forward(self, item_indices):
        """
        前向传播
        
        Args:
            item_indices: 物品索引
            
        Returns:
            rating: 预测评分
            item_personality: 物品个性化嵌入
            item_commonality: 物品共性嵌入
        """
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        pred = self.affine_output(item_personality + item_commonality)
        rating = self.logistic(pred)

        return rating, item_personality, item_commonality

    def full_sort_predict(self, interaction, *args, **kwargs):
        """
        全排序预测
        
        Args:
            interaction: 交互信息
            
        Returns:
            scores: 所有物品的预测分数
        """
        users = interaction[0]
        items = torch.arange(self.n_items).to(self.device)
        scores, _, _ = self.forward(items)

        return scores.view(-1)


class FedRAPTrainer(FederatedTrainer):
    """
    FedRAP模型的联邦训练器
    """

    def __init__(self, config, model, mg=False):
        """
        初始化FedRAP训练器
        
        Args:
            config: 训练配置参数
            model: FedRAP模型
            mg: 是否使用模型梯度
        """
        super(FedRAPTrainer, self).__init__(config, model, mg)

        # 设置学习率
        self.lr_network = self.config['lr']
        self.lr_args = self.config['lr'] * model.n_items

        # 复制共性嵌入
        self.item_commonality = copy.deepcopy(model.item_commonality)

        # 损失函数参数
        self.alpha = config['alpha']  # 独立性损失权重
        self.beta = config['beta']    # 正则化损失权重
        self.crit, self.independency, self.reg = nn.BCELoss(), nn.MSELoss(), nn.L1Loss()

    def _set_optimizer(self, model):
        """
        初始化优化器
        
        Args:
            model: 模型实例
            
        Returns:
            optimizer: 优化器实例
        """
        param_list = [
            {'params': model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': model.item_personality.parameters(), 'lr': self.lr_args},
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
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _set_client(self, *args, **kwargs):
        """
        设置客户端模型
        
        Args:
            user: 用户ID
            iteration: 当前迭代次数
            
        Returns:
            client_model: 客户端模型
            client_optimizer: 客户端优化器
        """
        user, iteration = args

        client_model = copy.deepcopy(self.model)
        client_model.set_item_commonality(self.item_commonality)

        # 如果不是第一次迭代且用户模型存在，则加载之前的模型参数
        if iteration != 0 and hasattr(self, 'client_models') and user in self.client_models.keys():
            for key in self.client_models[user].keys():
                client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])
            client_model.set_item_commonality(self.item_commonality)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        """
        训练一个批次
        
        Args:
            batch: 批次数据
            model: 客户端模型
            optimizer: 客户端优化器
            
        Returns:
            model: 更新后的模型
            optimizer: 优化器
            loss: 损失值
        """
        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        # 根据正负样本构建评分标签
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[:poss.size(0)] = 1
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred, item_personality, item_commonality = model(items)
        loss = self.calculate_loss(pred.view(-1), ratings, item_personality, item_commonality)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """
        存储客户端模型参数
        
        Args:
            user: 用户ID
            client_model: 客户端模型
            
        Returns:
            upload_params: 上传的参数（仅包含共性嵌入）
        """
        user, client_model = args

        # 确保client_models属性存在
        if not hasattr(self, 'client_models'):
            self.client_models = {}
            
        # 复制客户端模型到CPU并存储
        tmp_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        self.client_models[user] = copy.deepcopy(tmp_dict)
        # 从客户端模型中移除共性嵌入
        for key in tmp_dict.keys():
            if 'item_commonality' in key:
                del self.client_models[user][key]

        # 准备上传参数（仅包含共性嵌入）
        upload_params = copy.deepcopy(tmp_dict)
        for key in list(upload_params.keys()):
            if 'item_commonality' not in key:
                del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """
        聚合客户端参数
        
        Args:
            participant_params: 参与者的参数
        """
        participant_params = args[0]

        # 聚合所有参与者的共性嵌入参数
        i = 0
        for user in participant_params.keys():
            if i == 0:
                self.item_commonality.weight.data = participant_params[user]['item_commonality.weight'].data
            else:
                self.item_commonality.weight.data += participant_params[user]['item_commonality.weight'].data
            i += 1

        # 取平均值
        self.item_commonality.weight.data /= len(participant_params)

    def _update_hyperparams(self, *args, **kwargs):
        """
        更新超参数
        
        Args:
            iteration: 当前迭代次数
        """
        iteration = args[0]

        # 学习率衰减
        self.lr_args *= self.config['decay_rate']
        self.lr_network *= self.config['decay_rate']

        # 根据迭代次数更新损失函数权重
        self.alpha = math.tanh(iteration / 10) * self.alpha
        self.beta = math.tanh(iteration / 10) * self.beta

    def calculate_loss(self, interaction, *args, **kwargs):
        """
        计算损失函数
        
        Args:
            interaction: 预测值
            truth: 真实值
            item_personality: 物品个性化嵌入
            item_commonality: 物品共性嵌入
            
        Returns:
            loss: 总损失
        """
        pred, truth, item_personality, item_commonality = interaction, args[0], args[1], args[2]

        # 创建零张量作为正则化目标
        dummy_target = torch.zeros_like(item_commonality).to(self.device)

        # 计算总损失：预测损失 - 独立性损失 + 正则化损失
        loss = self.crit(pred, truth) \
               - self.alpha * self.independency(item_personality, item_commonality) \
               + self.beta * self.reg(item_commonality, dummy_target)

        return loss

    # @torch.no_grad()
    # def evaluate(self, eval_data, is_test=False, idx=0):
    #
    #     metrics = None
    #     for user, loader in eval_data.loaders.items():
    #         client_model, client_optimizer = self._set_client(user, 1)
    #         client_model.eval()
    #
    #         client_metrics = None
    #         batch_scores = []
    #         for batch_idx, batch in enumerate(loader):
    #             batch = batch[1].to(self.device)
    #             scores = client_model.full_sort_predict(batch)
    #             mask = batch[1]
    #             scores[mask] = -float('inf')
    #             _, indices = torch.topk(scores, k=max(self.config['topk']))
    #             batch_scores.append(indices)
    #
    #         client_metrics = self.evaluator.evaluate(batch_scores, loader, is_test=is_test, idx=idx)
    #
    #         if metrics == None:
    #             metrics = client_metrics
    #         else:
    #             for key in client_metrics.keys():
    #                 metrics[key] += client_metrics[key]
    #
    #     for key in metrics.keys():
    #         metrics[key] = metrics[key] / len(eval_data.loaders)
    #
    #     return metrics
