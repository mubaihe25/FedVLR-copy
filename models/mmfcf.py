import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from utils.federated.trainer import FederatedTrainer
from utils.utils import modal_ablation


class MMFCF(GeneralRecommender):

    def __init__(self, config, dataloader):
        super(MMFCF, self).__init__(config, dataloader)

        self.embed_size = config["embedding_size"]
        self.latent_size = config["latent_size"]

        self.item_commonality = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )

        self.fusion = FusionLayer(
            self.embed_size,
            fusion_module=config["fusion_module"],
            latent_dim=config["latent_size"],
        )

        self.affine_output = torch.nn.Linear(
            in_features=self.latent_size, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality):
        """设置物品共性特征嵌入层

        Args:
            item_commonality: 物品共性特征嵌入层
        """
        self.item_commonality.load_state_dict(item_commonality.state_dict())
        # self.item_commonality.freeze = True

    def forward(self, item_indices, txt_embed, vision_embed):
        item_commonality = self.item_commonality(item_indices)

        # 确保特征不参与计算图的构建
        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()

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

        pred = self.affine_output(out)
        # 注释掉sigmoid激活函数，因为BCEWithLogitsLoss会内部处理
        # rating = self.logistic(pred)

        return pred  # 直接返回logits

    def full_sort_predict(self, interaction, *args, **kwargs):
        txt_embed, vis_embed = args[0], args[1]

        users = interaction[0]
        items = torch.arange(self.n_items).to(self.device)
        logits = self.forward(items, txt_embed, vis_embed)

        # 在预测时应用sigmoid获取概率
        return torch.sigmoid(logits).view(-1)


class MMFCFTrainer(FederatedTrainer):

    def __init__(self, config, model, mg=False):
        super(MMFCFTrainer, self).__init__(config, model, mg)

        self.lr_network = self.config["lr"]
        self.lr_args = self.config["lr"] * model.n_items

        self.item_commonality = copy.deepcopy(model.item_commonality)

        self.fusion = copy.deepcopy(model.fusion)
        self.optimizer = optim.Adam(self.fusion.parameters(), lr=self.lr_network)

        self.crit = nn.BCEWithLogitsLoss()

    def _set_optimizer(self, model):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        param_list = [
            {"params": model.fusion.parameters(), "lr": self.lr_network},
            {"params": model.affine_output.parameters(), "lr": self.lr_network},
            {"params": model.item_commonality.parameters(), "lr": self.lr_args},
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
        user, iteration = args

        client_model = copy.deepcopy(self.model)

        if iteration != 0 and user in self.client_models.keys():
            for key in self.client_models[user].keys():
                client_model.state_dict()[key] = copy.deepcopy(
                    self.client_models[user][key]
                )

            client_model.fusion.load_state_dict(self.fusion.state_dict())

        client_model.set_item_commonality(self.item_commonality)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        # construct ratings according to the interaction of positive and negative items
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[: poss.size(0)] = 1
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(items, txt_features, vis_features)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        """存储客户端模型和准备上传参数

        Args:
            args: 位置参数，包含用户ID和客户端模型
            kwargs: 关键字参数

        Returns:
            dict: 需要上传到服务器的参数
        """
        user, client_model = args

        # 将模型移至CPU并复制状态字典
        user_dict = copy.deepcopy(client_model.to("cpu").state_dict())
        self.client_models[user] = copy.deepcopy(user_dict)

        # 准备上传参数：融合层的梯度和物品共性特征权重
        upload_params = {}

        # 收集融合层的梯度
        for name, param in client_model.fusion.named_parameters():
            if param.grad is not None:
                upload_params[name] = param.grad.clone()

        # 添加物品共性特征权重
        if "item_commonality.weight" in user_dict:
            upload_params["item_commonality.weight"] = user_dict[
                "item_commonality.weight"
            ].clone()

        # 从客户端模型状态字典中移除特定参数
        # 这些参数将在服务器端更新
        for key in list(self.client_models[user].keys()):
            if any(
                sub in key for sub in ["item_commonality", "mlp", "attention", "gate"]
            ):
                del self.client_models[user][key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        """聚合多个客户端的参数

        Args:
            args: 位置参数，第一个参数是参与者参数字典
            kwargs: 关键字参数
        """
        participant_params = args[0]

        # 检查是否有参与者，如果没有则直接返回
        if not participant_params:
            self.logger.warning("没有参与者提供参数，跳过聚合")
            return

        num_participants = len(participant_params)

        # 初始化梯度累加器和物品嵌入权重累加
        grad_accumulator = {}
        id_embed_weight_sum = None

        # 聚合参与者的参数
        for user, param_dict in participant_params.items():
            for name, param in param_dict.items():
                if name == "item_commonality.weight":
                    # 累加物品共性特征的权重
                    id_embed_weight_sum = (
                        param
                        if id_embed_weight_sum is None
                        else id_embed_weight_sum + param
                    )
                else:
                    # 累加其他参数的梯度
                    if name in grad_accumulator:
                        grad_accumulator[name] += param
                    else:
                        grad_accumulator[name] = param

        # 设置融合层参数的梯度为所有客户端平均值
        for name, param in self.fusion.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name].to(param.device) / num_participants
            else:
                # 如果某参数没有收到任何梯度更新，记录警告
                self.logger.warning(f"警告: 参数 {name} 没有收到任何梯度更新")

        # 应用梯度更新
        self.optimizer.step()

        # 更新物品共性特征权重为平均值
        if id_embed_weight_sum is not None:
            self.item_commonality.weight.data = id_embed_weight_sum / num_participants
        else:
            self.logger.warning("警告: 没有收到任何物品特征更新")

    def _update_hyperparams(self, *args, **kwargs):
        pass

    def calculate_loss(self, interaction, *args, **kwargs):
        pred, truth = interaction, args[0]

        loss = self.crit(pred, truth)

        return loss
