import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class GMF(nn.Module):
    def __init__(self, n_users, n_items, latent_dim):
        super(GMF, self).__init__()
        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating


class MLP(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, layers):
        super(MLP, self).__init__()

        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


class FedNCF(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(FedNCF, self).__init__(config, dataloader)

        self.embed_size = config['latent_size']
        self.latent_dim_mf = self.embed_size
        self.latent_dim_mlp = self.embed_size

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim_mf)

        layers = [2 * self.latent_dim_mlp, self.latent_dim_mlp, self.latent_dim_mlp // 2, self.latent_dim_mlp // 4]

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1] + self.latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def forward(self, user_indices, item_indices):
        """
        模型前向传播
        Args:
            user_indices: 用户索引
            item_indices: 物品索引
        Returns:
            rating: 预测评分/概率
        """
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        """
        为特定用户生成所有物品的预测评分
        Args:
            interaction: 包含用户ID的交互信息
        Returns:
            scores: 所有物品的预测评分
        """
        user = interaction[0][0]
        users = torch.tensor([user], device=self.device).expand(self.n_items)
        items = torch.arange(self.n_items, device=self.device)
        scores = self.forward(users, items)

        return scores.view(-1)


class FedNCFTrainer(FederatedTrainer):
    def __init__(self, config, model, mg=False):
        super(FedNCFTrainer, self).__init__(config, model, mg)
        self.lr = self.config['lr']
        self.crit, self.mae = nn.BCELoss(), nn.L1Loss()
        self.client_data_sizes = {}

    def _set_optimizer(self, model):
        """
        初始化优化器
        Args:
            model: 需要优化的模型
        Returns:
            optimizer: 优化器实例
        """
        param_list = [
            {'params': model.parameters(), 'lr': self.lr},
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
            self.logger.warning('未识别的优化器类型，使用默认的Adam优化器')
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
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
        
        if iteration != 0:
            client_model.load_state_dict(copy.deepcopy(self.global_model))
            if user in self.client_models:
                client_model.load_state_dict(copy.deepcopy(self.client_models[user]))

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        """
        训练一个批次
        Args:
            batch: 训练数据批次
            args: 包含模型和优化器的参数
        Returns:
            model: 更新后的模型
            optimizer: 优化器
            loss: 训练损失
        """
        batch_data = batch.to(self.device)
        user, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        ratings[:poss.size(0)] = 1
        users = torch.full_like(items, user[0], dtype=torch.long, device=self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(users, items)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        if user[0].item() not in self.client_data_sizes:
            self.client_data_sizes[user[0].item()] = 0
        self.client_data_sizes[user[0].item()] += len(batch_data[0])

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """
        存储客户端模型参数
        Args:
            user: 用户ID
            client_model: 客户端模型
        Returns:
            upload_params: 上传到服务器的参数
        """
        user, client_model = args

        tmp_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        self.client_models[user] = copy.deepcopy(tmp_dict)
        upload_params = copy.deepcopy(tmp_dict)

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """
        聚合客户端参数
        Args:
            participant_params: 所有参与客户端的参数
        """
        participant_params = args[0]
        
        total_samples = sum(self.client_data_sizes.get(user, 1) for user in participant_params.keys())
        
        if not hasattr(self, 'global_model') or self.global_model is None:
            self.global_model = copy.deepcopy(participant_params[list(participant_params.keys())[0]])
            for key in self.global_model:
                self.global_model[key].zero_()
        
        for user in participant_params:
            # weight = self.client_data_sizes.get(user, 1) / total_samples
            weight = 1 / len(self.client_models)
            for key in participant_params[user]:
                self.global_model[key].data += participant_params[user][key].data * weight

    def calculate_loss(self, interaction, *args, **kwargs):
        """
        计算损失
        Args:
            interaction: 预测结果
            args: 包含真实标签的参数
        Returns:
            loss: 计算的损失值
        """
        pred, truth = interaction, args[0]
        loss = self.crit(pred, truth)
        return loss
