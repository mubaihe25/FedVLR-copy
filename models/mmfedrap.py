import copy
import math

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from utils.federated.trainer import FederatedTrainer
from utils.utils import modal_ablation


class MMFedRAP(GeneralRecommender):
    """
    多模态联邦推荐模型，结合个性化和共性化的项目表示
    """

    def __init__(self, config, dataloader):
        super(MMFedRAP, self).__init__(config, dataloader)
        self.config = config

        # 嵌入维度和潜在空间维度
        self.embed_size = config["embedding_size"]
        self.latent_size = config["latent_size"]

        # 项目的个性化和共性化嵌入
        self.item_personality = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )
        self.item_commonality = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )

        # 多模态融合层
        self.fusion = FusionLayer(
            self.embed_size,
            fusion_module=config["fusion_module"],
            latent_dim=self.latent_size,
        )

        # 输出层
        self.affine_output = torch.nn.Linear(
            in_features=self.latent_size, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

        # 初始化参数
        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality):
        """设置项目共性嵌入"""
        self.item_commonality.load_state_dict(item_commonality.state_dict())

    def forward(self, item_indices, txt_embed, vision_embed):
        """
        前向传播
        Args:
            item_indices: 项目索引
            txt_embed: 文本嵌入
            vision_embed: 视觉嵌入
        Returns:
            logits: 预测logits值
            item_personality: 项目个性化嵌入
            item_commonality: 项目共性化嵌入
        """
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        # 确保特征不参与计算图的构建
        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()

        # 进行多模态消融测试
        processed_id, processed_txt, processed_vision = modal_ablation(
            item_commonality,  # 只对共性化嵌入进行消融测试
            txt,
            vision,
            txt_mode=self.config["txt_mode"],
            vis_mode=self.config["vis_mode"],
            id_mode=self.config["id_mode"],
        )

        # 将个性化嵌入与处理后的共性化嵌入结合
        combined_id = item_personality + processed_id

        # 使用处理后的嵌入进行融合
        out = self.fusion(combined_id, processed_txt, processed_vision)

        logits = self.affine_output(out)
        # 注释掉sigmoid激活函数，因为BCEWithLogitsLoss会内部处理
        # rating = self.logistic(logits)

        return logits, item_personality, item_commonality

    def full_sort_predict(self, interaction, *args, **kwargs):
        """
        全排序预测
        Args:
            interaction: 交互信息
            args: 额外参数，包含文本和视觉嵌入
        Returns:
            scores: 所有项目的预测分数
        """
        txt_embed, vis_embed = args[0], args[1]
        items = torch.arange(self.n_items).to(self.device)
        logits, _, _ = self.forward(items, txt_embed, vis_embed)
        # 在预测时应用sigmoid获取概率
        return torch.sigmoid(logits).view(-1)


class MMFedRAPTrainer(FederatedTrainer):
    """
    MMFedRAP模型的联邦训练器
    """

    def __init__(self, config, model, mg=False):
        super(MMFedRAPTrainer, self).__init__(config, model, mg)

        self.lr = self.config["lr"]

        # 复制模型的共性化嵌入和融合层
        self.item_commonality = copy.deepcopy(model.item_commonality)
        self.fusion = copy.deepcopy(model.fusion)
        self.optimizers = torch.optim.Adam(self.fusion.parameters(), lr=self.lr)

        # 损失函数的超参数
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.crit = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss替代BCELoss
        self.independency = nn.MSELoss()  # 均方误差损失，用于独立性约束
        self.reg = nn.L1Loss()  # L1正则化损失

    def _set_optimizer(self, model):
        """
        初始化优化器
        Args:
            model: 模型
        Returns:
            optimizer: 优化器
        """
        param_list = [
            {"params": model.fusion.parameters(), "lr": self.lr},
            {"params": model.affine_output.parameters(), "lr": self.lr},
            {
                "params": model.item_personality.parameters(),
                "lr": self.lr * self.model.n_items,
            },
            {
                "params": model.item_commonality.parameters(),
                "lr": self.lr * self.model.n_items,
            },
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
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        return optimizer

    def _set_client(self, *args, **kwargs):
        """
        设置客户端模型
        Args:
            args: 包含用户ID和迭代次数
        Returns:
            client_model: 客户端模型
            client_optimizer: 客户端优化器
        """
        user, iteration = args

        # 复制全局模型
        client_model = copy.deepcopy(self.model)

        if iteration != 0:
            # 如果不是第一次迭代，加载之前保存的客户端模型
            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key] = copy.deepcopy(
                        self.client_models[user][key]
                    )

            # 加载全局融合层
            client_model.fusion.load_state_dict(self.fusion.state_dict())

        # 设置共性化嵌入
        client_model.set_item_commonality(self.item_commonality)

        # 将模型移至设备并初始化优化器
        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        """
        训练一个批次
        Args:
            batch: 批次数据
            args: 包含模型和优化器
        Returns:
            model: 更新后的模型
            optimizer: 优化器
            loss: 损失值
        """
        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]

        # 构建正负样本的评分
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[: poss.size(0)] = 1  # 正样本评分为1，负样本为0
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        # 前向传播和反向传播
        optimizer.zero_grad()
        pred, item_personality, item_commonality = model(
            items, txt_features, vis_features
        )
        loss = self.calculate_loss(
            pred.view(-1), ratings, item_personality, item_commonality
        )
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """
        存储客户端模型并准备上传参数
        Args:
            args: 包含用户ID和客户端模型
        Returns:
            upload_params: 需要上传的参数
        """
        user, client_model = args

        # 复制客户端模型参数
        user_dict = copy.deepcopy(client_model.to("cpu").state_dict())
        self.client_models[user] = copy.deepcopy(user_dict)

        # 准备上传参数
        upload_params = {
            name: param.grad.clone()
            for name, param in client_model.fusion.named_parameters()
            if param.grad is not None
        }
        upload_params["item_commonality.weight"] = user_dict[
            "item_commonality.weight"
        ].clone()

        # 从客户端模型中移除共性化和融合层参数
        for key in user_dict.keys():
            if any(
                sub in key for sub in ["item_commonality", "mlp", "attention", "gate"]
            ):
                del self.client_models[user][key]
            else:
                if key in upload_params.keys():
                    del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """
        聚合客户端参数
        Args:
            args: 包含参与者参数
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

        # 累加所有客户端的梯度和嵌入权重
        for user, params in participant_params.items():
            for name, param in params.items():
                if "item_commonality" in name:
                    id_embed_weight_sum = (
                        param
                        if id_embed_weight_sum is None
                        else id_embed_weight_sum + param
                    )
                else:
                    if name not in grad_accumulator:
                        grad_accumulator[name] = param
                    else:
                        grad_accumulator[name] += param

        # 计算平均梯度并更新融合层
        for name, param in self.fusion.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name].to(param.device) / num_participants

        self.optimizer.step()

        # 更新共性化嵌入权重
        if id_embed_weight_sum is not None:
            self.item_commonality.weight.data = id_embed_weight_sum / num_participants

    def _update_hyperparams(self, *args, **kwargs):
        """
        更新超参数
        Args:
            args: 包含迭代次数
        """
        iteration = args[0]

        # 学习率衰减
        self.lr *= self.config["decay_rate"]

        # 根据迭代次数动态调整alpha和beta
        self.alpha = math.tanh(iteration / 10) * self.alpha
        self.beta = math.tanh(iteration / 10) * self.beta

    def calculate_loss(self, interaction, *args, **kwargs):
        """
        计算损失
        Args:
            interaction: 预测值
            args: 包含真实值、个性化嵌入和共性化嵌入
        Returns:
            loss: 总损失
        """
        pred, truth, item_personality, item_commonality = (
            interaction,
            args[0],
            args[1],
            args[2],
        )

        # 用于正则化的零目标
        dummy_target = torch.zeros_like(item_commonality).to(self.device)

        # 总损失 = 预测损失 - alpha * 独立性损失 + beta * 正则化损失
        loss = (
            self.crit(pred, truth)
            - self.alpha * self.independency(item_personality, item_commonality)
            + self.beta * self.reg(item_commonality, dummy_target)
        )

        return loss
