import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from utils.federated.trainer import FederatedTrainer
from utils.utils import modal_ablation


class MMPFedRec(GeneralRecommender):
    """多模态个性化联邦推荐系统模型

    结合文本和视觉特征的联邦推荐系统，保持用户隐私的同时利用多模态信息
    进行个性化推荐。
    """

    def __init__(self, config, dataloader):
        super(MMPFedRec, self).__init__(config, dataloader)

        # 配置嵌入维度和潜在特征维度
        self.embed_size = config["embedding_size"]
        self.latent_size = config["latent_size"]

        # 物品嵌入层
        self.item_embed = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )

        # 多模态特征融合层
        self.fusion = FusionLayer(
            self.embed_size,
            fusion_module=config["fusion_module"],
            latent_dim=self.latent_size,
        )

        # 输出层和激活函数
        self.affine_output = torch.nn.Linear(
            in_features=self.latent_size, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

        # 初始化模型参数
        self.apply(xavier_normal_initialization)

    def forward(self, item_indices, txt_embed, vision_embed):
        """前向传播函数

        结合物品ID嵌入和多模态特征计算预测评分

        Args:
            item_indices: 物品索引
            txt_embed: 文本特征嵌入
            vision_embed: 视觉特征嵌入

        Returns:
            物品的预测logits
        """
        # 获取物品ID嵌入
        item_embed = self.item_embed(item_indices)

        # 确保特征不参与计算图的构建
        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()

        # 进行多模态消融测试
        item_embed, txt, vision = modal_ablation(
            item_embed,
            txt,
            vision,
            txt_mode=self.config["txt_mode"],
            vis_mode=self.config["vis_mode"],
            id_mode=self.config["id_mode"],
        )

        # 融合ID嵌入和文本特征
        out = self.fusion(item_embed, txt, vision)

        # 生成预测分数
        logits = self.affine_output(out)
        # 注释掉sigmoid激活函数，因为BCEWithLogitsLoss会内部处理
        # rating = self.logistic(logits)

        return logits

    def full_sort_predict(self, interaction, *args, **kwargs):
        """为所有物品生成评分预测

        Args:
            interaction: 交互数据
            args: 包含文本和视觉特征的额外参数

        Returns:
            所有物品的预测评分
        """
        txt_embed, vis_embed = args[0], args[1]

        # 生成所有物品的索引
        items = torch.arange(self.n_items).to(self.device)

        # 计算所有物品的预测logits
        logits = self.forward(items, txt_embed, vis_embed)

        # 在预测时应用sigmoid获取概率
        return torch.sigmoid(logits).view(-1)


class MMPFedRecTrainer(FederatedTrainer):
    """MMPFedRec模型的联邦学习训练器

    实现多模态联邦推荐系统的训练逻辑
    """

    def __init__(self, config, model, mg=False):
        super(MMPFedRecTrainer, self).__init__(config, model, mg)

        # 设置学习率
        self.lr_network = self.config["lr"]
        self.lr_args = self.config["lr"] * model.n_items

        # 复制并初始化融合层
        self.fusion = copy.deepcopy(model.fusion)
        # 为融合层创建单独的优化器
        self.optimizer = torch.optim.Adam(self.fusion.parameters(), lr=self.lr_network)

        # 定义损失函数
        self.crit = nn.BCEWithLogitsLoss()

        # 初始化特征存储（应在训练前设置这些特征）
        # self.t_feat = None  # 文本特征
        # self.v_feat = None  # 视觉特征

    def _set_optimizer(self, model):
        """初始化模型优化器

        为不同参数组设置不同学习率

        Args:
            model: 待优化的模型

        Returns:
            torch.optim: 配置好的优化器
        """
        # 为不同参数组设置不同学习率
        param_list = [
            {"params": model.affine_output.parameters(), "lr": self.lr_network},
            {"params": model.item_embed.parameters(), "lr": self.lr_args},
        ]

        # 根据配置选择优化器
        if self.learner.lower() == "adam":
            optimizer = optim.Adam(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == "sgd":
            optimizer = optim.SGD(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == "adagrad":
            optimizer = optim.Adagrad(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(param_list, weight_decay=self.weight_decay)
        else:
            self.logger.warning("收到未识别的优化器类型，默认使用Adam优化器")
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return optimizer

    def _set_client(self, *args, **kwargs):
        """设置特定用户的客户端模型

        初始化客户端模型并加载相应参数

        Args:
            args: 用户ID和迭代轮次

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
                    client_model.state_dict()[key] = copy.deepcopy(
                        self.client_models[user][key]
                    )

            # 加载全局共享参数
            for key in self.global_model.keys():
                client_model.state_dict()[key] = copy.deepcopy(self.global_model[key])

            # 加载全局融合层
            client_model.fusion = copy.deepcopy(self.fusion)

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
        # 确保特征已加载到当前设备
        if self.t_feat is None or self.v_feat is None:
            raise ValueError("文本或视觉特征未设置，请在训练前设置t_feat和v_feat")

        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        # 准备批次数据
        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]

        # 构建正负样本的评分标签
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[: poss.size(0)] = 1  # 正样本标记为1
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        # 执行优化步骤
        optimizer.zero_grad()
        pred = model(items, txt_features, vis_features)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """存储客户端模型并准备上传参数

        分离本地和共享参数，准备服务器聚合

        Args:
            args: 用户ID和客户端模型

        Returns:
            upload_params: 需要上传到服务器的梯度参数
        """
        user, client_model = args

        # 复制客户端模型参数到CPU
        user_dict = copy.deepcopy(client_model.to("cpu").state_dict())

        # 存储客户端模型，但只保留affine_output层（用户特定参数）
        self.client_models[user] = copy.deepcopy(user_dict)
        for key in user_dict.keys():
            if "affine_output" not in key:
                del self.client_models[user][key]

        # 准备要上传的融合层梯度
        upload_params = {
            name: param.grad.clone()
            for name, param in client_model.fusion.named_parameters()
            if param.grad is not None
        }

        # 从客户端模型中删除融合层相关参数
        for key in list(self.client_models[user].keys()):
            if any(sub in key for sub in ["mlp", "attention", "gate"]):
                del self.client_models[user][key]
            else:
                if key in upload_params.keys():
                    del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """聚合多个客户端上传的参数

        聚合多个客户端的梯度更新

        Args:
            args: 包含参与者上传参数的字典
        """
        participant_params = args[0]

        # 检查是否有参与者，如果没有则直接返回
        if not participant_params:
            self.logger.warning("没有参与者提供参数，跳过聚合")
            return

        # 设置融合层为训练模式
        self.fusion.train()
        self.optimizer.zero_grad()

        # 梯度累加器
        grad_accumulator = {}

        # 处理物品嵌入参数
        i = 0
        for user, params in participant_params.items():
            if i == 0:
                self.global_model = copy.deepcopy(participant_params[user])
            else:
                for key in participant_params[user].keys():
                    self.global_model[key].data += participant_params[user][key].data
            i += 1

            # 累加梯度
            for name, param in self.fusion.named_parameters():
                if name in params and params[name] is not None:
                    if name not in grad_accumulator:
                        grad_accumulator[name] = params[name].clone()
                    else:
                        grad_accumulator[name] += params[name].clone()

        # 使用平均梯度更新融合层
        num_participants = len(participant_params)
        for name, param in self.fusion.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name].to(param.device) / num_participants

        # 更新融合层参数
        self.optimizer.step()

        # 物品嵌入参数平均化
        for key in self.global_model.keys():
            self.global_model[key].data /= num_participants

    def calculate_loss(self, interaction, *args, **kwargs):
        """计算模型损失

        计算预测与真实标签之间的损失

        Args:
            interaction: 模型预测值
            args[0]: 真实标签

        Returns:
            计算的损失值
        """
        pred, truth = interaction, args[0]
        loss = self.crit(pred, truth)

        return loss
