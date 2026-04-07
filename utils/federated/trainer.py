import os

import numpy as np
import torch

from common.trainer import Trainer
from utils.utils import sampleClients


class FederatedTrainer(Trainer):
    """联邦学习训练器

    继承自基础训练器，实现了联邦学习的训练逻辑。
    """

    def __init__(self, config, model, mg=False):
        super(FederatedTrainer, self).__init__(config, model, mg)

        self.global_model = None
        self.client_models = {}
        self.optimizers = {}

        self.last_participants = None
        self.weights = None
        self.user_metrics = {}

        if config["is_multimodal_model"]:
            dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
            v_feat_file_path = os.path.join(dataset_path, config["vision_feature_file"])
            t_feat_file_path = os.path.join(dataset_path, config["text_feature_file"])

            self.v_feat = self._load_feature_file(v_feat_file_path)
            self.t_feat = self._load_feature_file(t_feat_file_path)

            assert (
                self.v_feat is not None or self.t_feat is not None
            ), "Features all NONE"

            if self.v_feat is not None:
                self.v_feat.requires_grad_(False)
            if self.t_feat is not None:
                self.t_feat.requires_grad_(False)

    def _load_feature_file(self, file_path):
        """
        加载特征文件

        Args:
            file_path: 文件路径

        Returns:
            特征张量，若加载失败则返回None
        """
        if os.path.isfile(file_path):
            try:
                feature = torch.from_numpy(np.load(file_path, allow_pickle=True)).type(
                    torch.FloatTensor
                )
                feature.requires_grad_(False)
                return feature
            except Exception as e:
                self.logger.error(f"Failed to load feature file {file_path}: {e}")
        return None

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        """训练一个轮次

        Args:
            train_data: 训练数据
            epoch_idx: 轮次索引
            loss_func: 损失函数

        Returns:
            平均损失和用户损失列表
        """
        # Check before training
        assert hasattr(self, "model"), "Please specify the model"
        if not self.req_training:
            return 0.0, []

        # Randomly select a subset of clients
        sampled_clients = sampleClients(
            list(train_data.user_set),
            self.config["clients_sample_strategy"],
            self.config["clients_sample_ratio"],
            self.last_participants,
        )
        round_index = epoch_idx + 1
        self.experiment_hooks.start_round(round_index, sampled_clients)

        # Store the selected clients for the next round
        self.last_participants = sampled_clients
        self.weights = {user: 0 for user in range(self.model.n_users)}

        participant_params = {}
        total_loss, user_losses = 0, []
        for user in sampled_clients:
            client_loader = train_data.loaders[user]
            client_model, client_optimizer = self._set_client(user, epoch_idx)

            client_losses = self._train_client(
                client_model, client_optimizer, client_loader, user, epoch_idx
            )
            if client_losses is None:
                # 修复返回类型不一致问题，将返回值改为相同类型
                return float("nan"), []

            total_loss += client_losses[-1]
            user_losses.append(client_losses[-1])

            client_update = self._store_client_model(user, client_model)
            participant_params[user] = self.experiment_hooks.after_local_train(
                round_index=round_index,
                client_id=user,
                client_update=client_update,
            )

        # Aggregate the client model parameters in the server side
        participant_params = self.experiment_hooks.before_aggregation(
            round_index=round_index,
            participant_params=participant_params,
        )
        self._aggregate_params(participant_params)

        # Update the model hyperparameters
        self._update_hyperparams(epoch_idx)

        round_loss = total_loss / len(sampled_clients)
        self.experiment_hooks.finish_train_round(
            round_index=round_index,
            train_loss=round_loss,
            participant_count=len(sampled_clients),
        )

        return round_loss, user_losses

    def _train_client(
        self, client_model, client_optimizer, client_loader, user=None, epoch_idx=None
    ):
        """
        训练单个客户端模型

        Args:
            client_model: 客户端模型
            client_optimizer: 客户端优化器
            client_loader: 客户端数据加载器
            user: 用户ID
            epoch_idx: 外部循环索引

        Returns:
            客户端损失列表，如果训练过程中出现NaN则返回None
        """
        client_losses = []
        client_model.train()
        for epoch in range(self.config["local_epochs"]):
            client_loss = 0
            for batch_idx, batch in enumerate(client_loader):
                # 确保数据在正确的设备上
                if isinstance(batch, (tuple, list)):
                    batch = [
                        b.to(self.device) if torch.is_tensor(b) else b for b in batch
                    ]
                elif torch.is_tensor(batch):
                    batch = batch.to(self.device)

                client_model, client_optimizer, loss = self._train_one_batch(
                    batch, client_model, client_optimizer
                )

                if self._check_nan(loss):
                    self.logger.info(
                        "NaN Loss exists at the [Batch:{} of {}-th Inner Epoch at {}-th user of {}-th outer loop]".format(
                            batch_idx,
                            epoch,
                            user if user is not None else "unknown",
                            epoch_idx if epoch_idx is not None else "unknown",
                        )
                    )
                    return None

                client_loss += loss.item()

            client_losses.append(client_loss / len(client_loader))

            if (
                epoch > 0
                and abs(client_losses[-1] - client_losses[-2])
                / (client_losses[-1] + 1e-6)
                < self.config["tol"]
            ):
                break

        if user is not None and epoch_idx is not None:
            self.experiment_hooks.record_client_train(
                round_index=epoch_idx + 1,
                client_id=user,
                client_losses=client_losses,
            )

        return client_losses

    def _set_client(self, user_id, epoch_idx):
        """设置客户端模型和优化器

        Args:
            user_id: 用户ID
            epoch_idx: 轮次索引

        Returns:
            客户端模型和优化器

        Note:
            此方法必须在子类中实现
        """
        raise NotImplementedError("_set_client method must be implemented by subclass")

    def _train_one_batch(self, batch, client_model, client_optimizer):
        """训练一个批次

        Args:
            batch: 数据批次
            client_model: 客户端模型
            client_optimizer: 客户端优化器

        Returns:
            更新后的客户端模型、优化器和损失

        Note:
            此方法必须在子类中实现
        """
        raise NotImplementedError(
            "_train_one_batch method must be implemented by subclass"
        )

    def _aggregate_params(self, participant_params):
        """聚合参与者的模型参数

        Args:
            participant_params: 参与者的模型参数

        Note:
            此方法必须在子类中实现
        """
        raise NotImplementedError(
            "_aggregate_params method must be implemented by subclass"
        )

    def _store_client_model(self, user_id, client_model):
        """存储客户端模型

        Args:
            user_id: 用户ID
            client_model: 客户端模型

        Returns:
            存储的客户端模型参数

        Note:
            此方法必须在子类中实现
        """
        raise NotImplementedError(
            "_store_client_model method must be implemented by subclass"
        )

    def _update_hyperparams(self, epoch_idx):
        """更新超参数

        Args:
            epoch_idx: 轮次索引

        Note:
            此方法必须在子类中实现
        """
        pass

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        """评估模型

        Args:
            eval_data: 评估数据
            is_test: 是否为测试评估
            idx: 评估索引

        Returns:
            评估指标
        """
        t_feat, v_feat = None, None
        if self.config["is_multimodal_model"]:
            t_feat = self.t_feat.to(self.device)
            v_feat = self.v_feat.to(self.device)

        metrics = None
        for user, loader in eval_data.loaders.items():
            client_model, _ = self._set_client(user, 1)
            client_model.eval()

            batch_scores = []
            for batch_idx, batch in enumerate(loader):
                # 确保数据在正确的设备上
                if isinstance(batch, (tuple, list)):
                    batch = [
                        b.to(self.device) if torch.is_tensor(b) else b for b in batch
                    ]
                elif torch.is_tensor(batch):
                    batch = batch.to(self.device)
                else:
                    batch = batch[1].to(self.device)

                scores = client_model.full_sort_predict(batch, t_feat, v_feat)
                mask = batch[1]
                scores[mask] = -float("inf")
                _, indices = torch.topk(scores, k=max(self.config["topk"]))
                batch_scores.append(indices)

            client_metrics = self.evaluator.evaluate(
                batch_scores, loader, is_test=is_test, idx=idx
            )

            self.user_metrics[user] = client_metrics

            if metrics is None:
                metrics = client_metrics
            else:
                for key in client_metrics.keys():
                    metrics[key] += client_metrics[key]

        for key in metrics.keys():
            metrics[key] = metrics[key] / len(eval_data.loaders)

        return metrics
