import copy
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class FedVBPR(GeneralRecommender):
    """联邦多模态 VBPR 模型

    该模型在经典 VBPR 的基础上，将项目 ID 嵌入视为联邦学习中的公共参数，
    其余用户相关参数保留在客户端本地。
    """

    def __init__(self, config, dataloader):
        super(FedVBPR, self).__init__(config, dataloader)

        # 基础超参数
        self.embed_size = config["embedding_size"]  # 最终统一的特征维度
        self.u_embedding_size = self.i_embedding_size = self.embed_size
        self.reg_weight = config["reg_weight"]

        # 用户 & 项目嵌入
        # 用户嵌入维度为 2 * embed_size，与 VBPR 拼接逻辑保持一致
        self.u_embedding = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.n_users, self.embed_size * 2)
            )
        )
        # 项目 ID 嵌入，作为联邦公共参数，维度为 embed_size
        self.item_commonality = nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )

        # 由 trainer 在训练/推理时动态提供 item_raw_features
        self.item_raw_features = None  # type: Optional[torch.Tensor]

        # item_linear 延迟初始化，待获得特征维度后再构造
        self.item_linear: nn.Linear | None = None

        # 损失函数
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # 参数初始化
        self.apply(xavier_normal_initialization)

    # ---------------------------------------------------------------------
    # 接口函数
    # ---------------------------------------------------------------------
    def set_item_commonality(self, item_commonality: nn.Embedding):
        """同步服务器端的公共项目嵌入"""
        self.item_commonality.load_state_dict(item_commonality.state_dict())

    def get_user_embedding(self, user: torch.LongTensor) -> torch.Tensor:
        return self.u_embedding[user, :]

    def forward(self, dropout: float = 0.0):
        """前向传播，依赖外部已设置好的 self.item_raw_features"""
        assert (
            self.item_raw_features is not None
        ), "item_raw_features 未设置，请在调用 forward 前赋值"

        # 若尚未初始化映射层，则根据特征维度初始化
        if self.item_linear is None:
            in_dim = self.item_raw_features.shape[1]
            self.item_linear = nn.Linear(in_dim, self.embed_size).to(self.device)
            xavier_normal_initialization(self.item_linear)

        item_feat_embed = self.item_linear(self.item_raw_features.to(self.device))
        item_embed = torch.cat((self.item_commonality.weight, item_feat_embed), dim=-1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embed, dropout)
        return user_e, item_e

    # ---------------------------------------------------------------------
    # 训练 & 预测
    # ---------------------------------------------------------------------
    def calculate_loss(self, interaction):
        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]

        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_score, neg_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction, *args, **kwargs):
        """全排序预测

        Args:
            interaction: 交互信息 (user, ...)
            *args: 可选 (txt_embed, vis_embed)
        """
        # 处理外部传入的特征
        txt_embed = args[0] if len(args) > 0 else None
        vis_embed = args[1] if len(args) > 1 else None

        if txt_embed is not None and vis_embed is not None:
            self.item_raw_features = torch.cat((txt_embed, vis_embed), dim=-1)
        elif txt_embed is not None:
            self.item_raw_features = txt_embed
        elif vis_embed is not None:
            self.item_raw_features = vis_embed
        else:
            raise ValueError("Both txt_embed and vis_embed are None in full_sort_predict")

        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        scores = torch.matmul(user_e, item_embeddings.transpose(0, 1)).view(-1)
        return scores


class FedVBPRTrainer(FederatedTrainer):
    """FedVBPR 训练器，实现类似 MMFedAvg 的联邦参数聚合逻辑"""

    def __init__(self, config, model: FedVBPR, mg: bool = False):
        super(FedVBPRTrainer, self).__init__(config, model, mg)

        # 学习率设置
        self.lr_network = self.config["lr"]
        self.lr_args = self.config["lr"]

        # 服务器端保存的公共参数
        self.item_commonality = copy.deepcopy(model.item_commonality)

        # -------------------------------------------------------------
        # 对齐 txt / vis 特征行数，使其等于 n_items，防止索引越界
        # -------------------------------------------------------------
        n_items = model.n_items

        def _align_feat(feat: torch.Tensor | None, target_rows: int):
            """若 feat 行数不足 target_rows 则补零，若超过则保留（不裁剪）"""
            if feat is None:
                return None
            if feat.size(0) < target_rows:
                pad = torch.zeros(
                    (target_rows - feat.size(0), feat.size(1)),
                    dtype=feat.dtype,
                    device=feat.device,
                )
                return torch.cat([feat, pad], dim=0)
            else:
                return feat  # 保留原始行数，即便大于 target_rows

        if hasattr(self, "t_feat"):
            self.t_feat = _align_feat(self.t_feat, n_items)
        if hasattr(self, "v_feat"):
            self.v_feat = _align_feat(self.v_feat, n_items)

        # 重新计算全量项目数（可能 > 训练集）
        full_n_items = max(
            self.t_feat.size(0) if hasattr(self, "t_feat") and self.t_feat is not None else 0,
            self.v_feat.size(0) if hasattr(self, "v_feat") and self.v_feat is not None else 0,
            model.item_commonality.num_embeddings,
        )

        if full_n_items > model.n_items:
            # 扩展 item_commonality
            delta = full_n_items - model.n_items
            extra_weight = torch.empty(delta, model.embed_size, device=model.item_commonality.weight.device)
            nn.init.xavier_uniform_(extra_weight)
            model.item_commonality.weight.data = torch.cat(
                [model.item_commonality.weight.data, extra_weight], dim=0
            )
            model.item_commonality.num_embeddings = full_n_items  # type: ignore[attr-defined]
            model.n_items = full_n_items

            # 同步 server 侧副本
            self.item_commonality = copy.deepcopy(model.item_commonality)

        # 根据对齐后的多模态特征确定输入维度，并初始化映射层
        feature_dim = 0
        if hasattr(self, "t_feat") and self.t_feat is not None:
            feature_dim += self.t_feat.shape[1]
        if hasattr(self, "v_feat") and self.v_feat is not None:
            feature_dim += self.v_feat.shape[1]

        if feature_dim == 0:
            # 如果没有任何特征，抛出错误
            raise ValueError("无法确定多模态特征维度，t_feat 与 v_feat 均为 None")

        # 若模型仍未初始化 item_linear，则在此初始化
        if model.item_linear is None:
            model.item_linear = nn.Linear(feature_dim, model.embed_size).to(model.device)
            xavier_normal_initialization(model.item_linear)

        self.item_linear = copy.deepcopy(model.item_linear)

        # 类型检查
        assert self.item_linear is not None

        # 客户端权重累积，用于加权平均
        self.weights = defaultdict(int)
        self.client_models = {}

        # 仅优化 item_linear（全局共享）
        if self.item_linear is not None:
            param_list_for_opt = self.item_linear.parameters()  # type: ignore[arg-type]
        else:
            param_list_for_opt = []

        self.optimizer = optim.Adam(
            param_list_for_opt,
            lr=self.lr_network,
            weight_decay=self.weight_decay,
        )

        # 损失函数直接使用模型内部实现

    # ------------------------------------------------------------------
    # Optimizer & Client Set-up
    # ------------------------------------------------------------------
    def _set_optimizer(self, model: FedVBPR):
        param_list = []

        if model.item_linear is not None:
            param_list.append({"params": model.item_linear.parameters(), "lr": self.lr_network})

        param_list.append({"params": model.item_commonality.parameters(), "lr": self.lr_args})

        optimizer_map = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "adagrad": optim.Adagrad,
            "rmsprop": optim.RMSprop,
        }
        opt_class = optimizer_map.get(self.learner.lower(), optim.Adam)
        return opt_class(param_list, weight_decay=self.weight_decay)

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)

        # 加载用户本地私有参数（若存在）
        if iteration != 0 and user in self.client_models:
            for key in self.client_models[user].keys():
                client_model.state_dict()[key] = copy.deepcopy(
                    self.client_models[user][key]
                )

            # 同步全局 item_linear
            client_model.item_linear.load_state_dict(self.item_linear.state_dict())

        # 同步全局公共项目嵌入
        client_model.set_item_commonality(self.item_commonality)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)
        return client_model, client_optimizer

    # ------------------------------------------------------------------
    # Training on Client
    # ------------------------------------------------------------------
    def _train_one_batch(self, batch, *args, **kwargs):
        # 根据侧信息构造 item_raw_features 并设置到模型
        txt_features = self.t_feat.to(self.device) if hasattr(self, "t_feat") and self.t_feat is not None else None
        vis_features = self.v_feat.to(self.device) if hasattr(self, "v_feat") and self.v_feat is not None else None

        if txt_features is not None and vis_features is not None:
            item_raw_features = torch.cat((txt_features, vis_features), dim=-1)
        elif txt_features is not None:
            item_raw_features = txt_features
        elif vis_features is not None:
            item_raw_features = vis_features
        else:
            raise ValueError("Both txt_features and vis_features are None!")

        batch_data = batch.to(self.device)
        user, poss, negs = batch_data[0], batch_data[1], batch_data[2]

        # 更新交互权重
        self.weights[user[0].item()] += len(poss) + len(negs)

        model, optimizer = args[0], args[1]

        # 注入原始特征到模型
        model.item_raw_features = item_raw_features

        optimizer.zero_grad()
        loss = model.calculate_loss((user, poss, negs))
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def _store_client_model(self, *args, **kwargs):
        user, client_model = args

        # 保存用户完整模型以便下轮继续训练
        user_state = copy.deepcopy(client_model.to("cpu").state_dict())
        self.client_models[user] = copy.deepcopy(user_state)

        # 仅上传公共参数及 item_linear 的梯度
        upload_params = {
            name: param.grad.clone()
            for name, param in client_model.item_linear.named_parameters()
            if param.grad is not None
        }
        upload_params["item_commonality.weight"] = user_state[
            "item_commonality.weight"
        ].clone()

        # 移除本地无需全局同步的参数以节省内存
        for key in list(self.client_models[user].keys()):
            if "item_commonality" in key or "item_linear" in key:
                del self.client_models[user][key]
            elif key in upload_params:
                del upload_params[key]

        return upload_params

    # ------------------------------------------------------------------
    # Aggregate on Server
    # ------------------------------------------------------------------
    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]
        if not participant_params:
            return

        self.item_linear.train()
        self.optimizer.zero_grad()

        grad_accumulator = {}
        id_embed_weight_sum = None

        for user, param_dict in participant_params.items():
            weight = self.weights[user] / self.model.n_items
            for name, param in param_dict.items():
                if name == "item_commonality.weight":
                    id_embed_weight_sum = (
                        weight * param
                        if id_embed_weight_sum is None
                        else id_embed_weight_sum + weight * param
                    )
                else:
                    grad_accumulator[name] = grad_accumulator.get(name, 0.0) + weight * param

        # 设置 item_linear 的梯度
        for name, param in self.item_linear.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name].to(param.device)

        # 可选梯度裁剪
        if hasattr(self, "clip_grad_norm") and self.clip_grad_norm:
            clip_grad_norm_(self.item_linear.parameters(), self.clip_grad_norm)

        # 更新全局 item_linear
        self.optimizer.step()

        # 将更新后的全局参数同步到服务器主模型，便于下轮复制
        self.model.item_linear.load_state_dict(self.item_linear.state_dict())

        # 更新全局公共项目嵌入
        if id_embed_weight_sum is not None:
            self.item_commonality.weight.data = id_embed_weight_sum.to(
                self.item_commonality.weight.device
            )

        # 同步公共嵌入到主模型
        self.model.item_commonality.weight.data = (
            self.item_commonality.weight.data.clone()
        )

    # ------------------------------------------------------------------
    def _update_hyperparams(self, *args, **kwargs):
        """可在此处实现学习率调度等"""
        pass 