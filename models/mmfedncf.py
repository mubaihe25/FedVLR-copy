import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from utils.federated.trainer import FederatedTrainer
from utils.utils import modal_ablation


class GMF(nn.Module):
    """
    通用矩阵分解(General Matrix Factorization)模型
    将用户和物品映射到隐藏空间，通过元素乘积预测评分
    """

    def __init__(self, n_users, n_items, latent_dim):
        super(GMF, self).__init__()
        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)  # 元素乘积
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating


class MLP(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, layers):
        super(MLP, self).__init__()

        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat(
            [user_embedding, item_embedding], dim=-1
        )  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


class MMFedNCF(GeneralRecommender):
    """
    多模态联邦神经协同过滤模型
    结合了MLP和GMF的混合模型，加入了多模态特征融合功能
    """

    def __init__(self, config, dataloader):
        super(MMFedNCF, self).__init__(config, dataloader)

        self.embed_size = config["embedding_size"]
        self.latent_size = config["latent_size"]

        self.latent_dim_mf = self.latent_size
        self.latent_dim_mlp = self.latent_size

        self.item_commonality = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )

        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_mf
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim_mf
        )

        self.fusion = FusionLayer(
            self.embed_size,
            fusion_module=config["fusion_module"],
            latent_dim=config["latent_size"],
        )

        layers = [
            2 * self.latent_dim_mlp,
            self.latent_dim_mlp,
            self.latent_dim_mlp // 2,
            self.latent_dim_mlp // 4,
        ]

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=layers[-1] + self.latent_dim_mf, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality):
        """
        设置物品共性嵌入向量

        Args:
            item_commonality: 物品共性嵌入层
        """
        self.item_commonality.load_state_dict(item_commonality.state_dict())
        # self.item_commonality.freeze = True

    # 兼容性别名，保持向后兼容
    setItemCommonality = set_item_commonality

    def forward(self, user_indices, item_indices, txt_embed, vision_embed):
        """
        前向传播函数

        Args:
            user_indices: 用户索引
            item_indices: 物品索引
            txt_embed: 文本特征嵌入
            vision_embed: 视觉特征嵌入

        Returns:
            logits: 预测的logits（未经过sigmoid）
        """
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # 确保特征不参与计算图的构建
        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()
        item_commonality = self.item_commonality(item_indices)

        # 进行多模态消融测试
        item_commonality, txt, vision = modal_ablation(
            item_commonality,
            txt,
            vision,
            txt_mode=self.config["txt_mode"],
            vis_mode=self.config["vis_mode"],
            id_mode=self.config["id_mode"],
        )

        out = self.fusion(item_commonality, txt, vision)

        # 融合特征与物品嵌入向量
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp + out], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf + out)

        # MLP部分
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # 连接MLP输出和GMF输出
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)

        return logits

    def full_sort_predict(self, interaction, txt_feat=None, vis_feat=None):
        """全排序预测方法

        Args:
            interaction: 交互数据
            txt_feat: 文本特征
            vis_feat: 视觉特征

        Returns:
            scores: 预测分数
        """
        # 检查输入类型并适当处理
        if isinstance(interaction, list):
            # 联邦学习环境下的处理
            user = interaction[0]
            if isinstance(user, torch.Tensor):
                user = user[0]  # 获取用户ID
        else:
            # 传统环境下的处理
            user = interaction[0, 0]

        # 获取用户嵌入
        user_mlp = self.embedding_user_mlp(user)
        user_mf = self.embedding_user_mf(user)

        # 获取所有物品嵌入
        all_item_mlp = self.embedding_item_mlp.weight
        all_item_mf = self.embedding_item_mf.weight

        # 特征融合处理
        item_feats, txt_feats, vis_feats = self._process_features(
            all_item_mlp, txt_feat, vis_feat
        )
        fused_item = self.fusion(item_feats, txt_feats, vis_feats)

        # 将用户向量扩展到与物品数量相同的维度
        user_mlp = user_mlp.expand(all_item_mlp.shape[0], -1)
        user_mf = user_mf.expand(all_item_mf.shape[0], -1)

        # 计算MLP部分的输入
        mlp_vector = torch.cat([user_mlp, all_item_mlp + fused_item], dim=-1)
        # 计算MF部分的输入
        mf_vector = torch.mul(user_mf, all_item_mf + fused_item)

        # MLP部分前向传播
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # 连接MLP和MF结果
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        # 计算最终得分
        logits = self.affine_output(vector)
        scores = self.logistic(logits)

        return scores.view(-1)

    def _process_features(self, item_feats, txt_feat=None, vis_feat=None):
        """处理特征数据，应用模态消融测试等操作

        Args:
            item_feats: 物品ID特征
            txt_feat: 文本特征
            vis_feat: 视觉特征

        Returns:
            tuple: 处理后的(物品特征, 文本特征, 视觉特征)
        """
        # 获取物品共性嵌入
        item_commonality = self.item_commonality.weight

        # 确保特征存在，若不存在则创建零tensor
        if txt_feat is None:
            txt_feat = torch.zeros_like(item_commonality)

        if vis_feat is None:
            vis_feat = torch.zeros_like(item_commonality)

        # 应用模态消融测试
        item_commonality, txt_feat, vis_feat = modal_ablation(
            item_commonality,
            txt_feat,
            vis_feat,
            txt_mode=self.config["txt_mode"],
            vis_mode=self.config["vis_mode"],
            id_mode=self.config["id_mode"],
        )

        return item_commonality, txt_feat, vis_feat


class MMFedNCFTrainer(FederatedTrainer):
    """
    多模态联邦神经协同过滤模型的训练器
    实现联邦学习范式下的模型训练和聚合
    """

    def __init__(self, config, model, mg=False):
        super(MMFedNCFTrainer, self).__init__(config, model, mg)

        self.lr = self.config["lr"]

        self.item_commonality = copy.deepcopy(model.item_commonality)

        self.fusion = copy.deepcopy(model.fusion)
        self.optimizer = optim.Adam(self.fusion.parameters(), lr=self.lr)

        self.crit, self.mae = nn.BCEWithLogitsLoss(), nn.L1Loss()

    def _set_optimizer(self, model):
        """
        初始化优化器

        Args:
            model: 需要优化的模型

        Returns:
            optimizer: 优化器实例
        """
        param_list = [
            {"params": model.parameters(), "lr": self.lr},
        ]

        if self.learner.lower() == "adam":
            optimizer = optim.Adam(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == "sgd":
            optimizer = optim.SGD(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == "adagrad":
            optimizer = optim.Adagrad(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(param_list, weight_decay=self.weight_decay)
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
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
        if iteration != 0:
            # 加载客户端模型参数
            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key] = copy.deepcopy(
                        self.client_models[user][key]
                    )

            # 加载全局融合层参数
            client_model.fusion.load_state_dict(self.fusion.state_dict())

        # 设置物品共性嵌入
        client_model.set_item_commonality(self.item_commonality)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        """
        训练一个批次的数据

        Args:
            batch: 包含用户和物品数据的批次
            args: 包含模型和优化器的参数

        Returns:
            model: 更新后的模型
            optimizer: 更新后的优化器
            loss: 训练损失
        """
        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        batch_data = batch.to(self.device)
        user, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        # construct ratings according to the interaction of positive and negative items
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[: poss.size(0)] = 1
        ratings = ratings.to(self.device)
        users = torch.full_like(ratings, user[0], dtype=torch.long).to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(users, items, txt_features, vis_features)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """
        存储客户端模型并准备上传参数

        Args:
            user: 用户ID
            client_model: 客户端模型

        Returns:
            upload_params: 需要上传到服务器的参数
        """
        user, client_model = args

        user_dict = copy.deepcopy(client_model.to("cpu").state_dict())
        self.client_models[user] = copy.deepcopy(user_dict)

        upload_params = {
            name: param.grad.clone()
            for name, param in client_model.fusion.named_parameters()
            if param.grad is not None
        }
        upload_params["item_commonality.weight"] = user_dict[
            "item_commonality.weight"
        ].clone()

        for key in user_dict.keys():
            if not any(sub in key for sub in ["router"]):
                del self.client_models[user][key]
            else:
                if key in upload_params.keys():
                    del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """
        聚合客户端参数

        Args:
            participant_params: 参与者参数字典
        """
        participant_params = args[0]

        num_participants = len(participant_params)
        if num_participants == 0:
            self.logger.warning("没有参与者提供参数，跳过聚合")
            return  # 如果没有参与者，直接返回

        self.fusion.train()
        self.optimizer.zero_grad()

        grad_accumulator = {}
        id_embed_weight_sum = None

        for user, param_dict in participant_params.items():
            # 检查权重是否存在
            if user not in self.weights:
                continue

            w = self.weights[user] / self.model.n_items

            for name, param in param_dict.items():
                if name == "item_commonality.weight":
                    id_embed_weight_sum = (
                        param
                        if id_embed_weight_sum is None
                        else id_embed_weight_sum + param
                    )
                else:
                    if name in grad_accumulator:
                        grad_accumulator[name] += param
                    else:
                        grad_accumulator[name] = param

        # 更新融合层梯度
        for name, param in self.fusion.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name].to(param.device) / num_participants

        self.optimizer.step()

        # 更新物品共性嵌入权重
        if id_embed_weight_sum is not None:
            self.item_commonality.weight.data = id_embed_weight_sum / num_participants

    def calculate_loss(self, prediction, target, *args, **kwargs):
        """
        计算损失函数

        Args:
            prediction: 模型预测结果
            target: 真实标签

        Returns:
            loss: 计算的损失值
        """
        loss = self.crit(prediction, target)
        return loss
