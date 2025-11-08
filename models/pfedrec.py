import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class PFedRec(GeneralRecommender):
    """个性化联邦推荐系统模型
    
    该模型实现了基于联邦学习的个性化推荐系统，保持用户隐私的同时实现协作学习。
    全局共享物品嵌入，本地保留用户特定的模型参数。
    """

    def __init__(self, config, dataloader):
        super(PFedRec, self).__init__(config, dataloader)

        self.embed_size = config['embedding_size']
        # 创建物品嵌入矩阵
        self.item_embed = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)
        # 全连接输出层，将嵌入映射为评分
        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        # Sigmoid激活函数确保输出在[0,1]范围内
        self.logistic = torch.nn.Sigmoid()

        # 应用Xavier初始化
        self.apply(xavier_normal_initialization)

    def forward(self, item_indices):
        """前向传播计算物品评分
        
        Args:
            item_indices: 物品索引张量
            
        Returns:
            物品的预测评分
        """
        item_embed = self.item_embed(item_indices)
        pred = self.affine_output(item_embed)
        rating = self.logistic(pred)
        
        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        """为所有物品生成评分预测
        
        用于推荐时对所有物品进行排序
        
        Args:
            interaction: 交互数据
            
        Returns:
            所有物品的预测评分
        """
        items = torch.arange(self.n_items).to(self.device)
        scores = self.forward(items)
        
        return scores.view(-1)


class PFedRecTrainer(FederatedTrainer):
    """PFedRec模型的联邦学习训练器
    
    实现了联邦学习训练逻辑，包括客户端模型设置、训练、参数聚合等
    """

    def __init__(self, config, model, mg=False):
        super(PFedRecTrainer, self).__init__(config, model, mg)

        # 设置不同组件的学习率
        self.lr_network = self.config['lr']  # 网络层的学习率
        self.lr_args = self.config['lr'] * model.n_items  # 物品嵌入的学习率

        # 二元交叉熵损失函数
        self.crit = nn.BCELoss()

    def _set_optimizer(self, model):
        """初始化优化器
        
        对不同参数组设置不同学习率
        
        Args:
            model: 待优化的模型
            
        Returns:
            torch.optim: 配置好的优化器
        """
        # 为不同参数组设置不同学习率
        param_list = [
            {'params': model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': model.item_embed.parameters(), 'lr': self.lr_args},
        ]

        # 根据配置选择优化器
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(param_list, weight_decay=self.weight_decay)
        else:
            self.logger.warning('收到未识别的优化器类型，默认使用Adam优化器')
            optimizer = optim.Adam(param_list, lr=self.learning_rate)
        return optimizer

    def _set_client(self, *args, **kwargs):
        """设置特定用户的客户端模型
        
        根据用户和迭代轮次初始化客户端模型，加载已有的客户端参数和全局参数
        
        Args:
            user: 用户ID
            iteration: 当前迭代轮次
            
        Returns:
            client_model: 初始化好的客户端模型
            client_optimizer: 对应的优化器
        """
        user, iteration = args

        # 创建客户端模型副本
        client_model = copy.deepcopy(self.model)
        
        # 非首轮迭代时，加载既有参数
        if iteration != 0:
            # 加载用户特定的客户端模型参数（如果存在）
            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])
            
            # 加载全局共享参数
            for key in self.global_model.keys():
                client_model.state_dict()[key] = copy.deepcopy(self.global_model[key])

        # 将模型迁移到指定设备
        client_model = client_model.to(self.device)
        # 设置客户端优化器
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        """训练一个批次的数据
        
        处理一批次数据并更新模型参数
        
        Args:
            batch: 当前批次数据
            args: 包含模型和优化器的其他参数
            
        Returns:
            更新后的模型、优化器和计算的损失
        """
        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        
        # 构建正负样本的评分标签
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[:poss.size(0)] = 1  # 正样本标记为1
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        # 执行优化步骤
        optimizer.zero_grad()
        pred = model(items)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """存储客户端模型并准备上传参数
        
        存储用户特定的参数（仅affine_output层），并准备上传共享参数（item_embed）
        
        Args:
            user: 用户ID
            client_model: 训练后的客户端模型
            
        Returns:
            upload_params: 准备上传到服务器的参数
        """
        user, client_model = args

        # 复制客户端模型参数到CPU
        tmp_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        
        # 存储客户端模型，但只保留affine_output层（用户特定参数）
        self.client_models[user] = copy.deepcopy(tmp_dict)
        for key in tmp_dict.keys():
            if 'affine_output' not in key:
                del self.client_models[user][key]

        # 准备上传参数，只包含非affine_output层（共享参数）
        upload_params = copy.deepcopy(tmp_dict)
        for key in self.client_models[user].keys():
            if 'affine_output' in key:
                del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """聚合多个客户端上传的参数
        
        使用平均策略聚合多个客户端的参数更新
        
        Args:
            participant_params: 参与者上传的参数字典
        """
        participant_params = args[0]

        # 聚合多个客户端的参数
        i = 0
        for user in participant_params.keys():
            if i == 0:
                # 第一个参与者的参数作为初始值
                self.global_model = copy.deepcopy(participant_params[user])
            else:
                # 累加其他参与者的参数
                for key in participant_params[user].keys():
                    self.global_model[key].data += participant_params[user][key].data
            i += 1

        # 计算参数平均值
        for key in self.global_model.keys():
            self.global_model[key].data /= len(participant_params)

    def calculate_loss(self, interaction, *args, **kwargs):
        """计算模型损失
        
        使用二元交叉熵损失函数计算预测与真实标签之间的损失
        
        Args:
            interaction: 模型预测值
            args[0]: 真实标签
            
        Returns:
            计算的损失值
        """
        pred, truth = interaction, args[0]
        loss = self.crit(pred, truth)
        
        return loss
