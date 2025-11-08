import pandas

import utils.dataloader as loader_utils
from utils.dataset import RecDataset


class FederatedDataLoader(object):
    """联邦学习数据加载器

    Args:
        config (Config): 配置对象
        dataset (Dataset): 数据集对象
        batch_size (int, optional): 批次大小，默认为1
        neg_sampling (bool, optional): 是否进行负采样，默认为False
        shuffle (bool, optional): 是否打乱数据，默认为False
        stage (str, optional): 阶段，可选值为'train', 'valid', 'test'，默认为'train'
        additional_dataset (Dataset, optional): 额外的数据集，用于评估阶段，默认为None
    """

    def __init__(
        self,
        config,
        dataset,
        batch_size=1,
        neg_sampling=False,
        shuffle=False,
        stage="train",
        additional_dataset=None,
    ):
        self.config = config
        self.dataset = dataset

        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling

        self.additional_dataset = additional_dataset
        self.stage = stage

        # 初始化数据加载器
        self.data_loader = self._get_federated_loader()

        # 用户ID列表，用于迭代
        self.user_ids = list(self.data_loader.keys())
        self.user_idx = 0

    def _get_federated_loader(self):
        """获取联邦数据加载器

        Returns:
            dict: 用户ID到数据加载器的映射
        """
        if self.stage == "train":
            return self._get_train_loader()
        elif self.stage in ["valid", "test", "eval"]:
            return self._get_eval_loader()
        else:
            raise ValueError(f"无效的阶段: {self.stage}")

    def _get_user_datasets(self, filter_condition=None):
        """获取用户的数据集

        Args:
            filter_condition (callable, optional): 过滤用户的函数

        Returns:
            dict: 用户ID到数据集的映射
        """
        user_datasets = {}
        for user_id in self.dataset.df[self.dataset.uid_field].unique():
            if filter_condition and not filter_condition(user_id):
                continue

            # 获取用户数据
            user_df = self.dataset.df[
                self.dataset.df[self.dataset.uid_field] == user_id
            ]

            # 创建新的数据集对象
            user_dataset = RecDataset(self.config, user_df)

            # 设置必要的属性
            user_dataset.inter_num = len(user_df)
            user_dataset.user_num = 1  # 一个用户
            user_dataset.item_num = self.dataset.item_num  # 保持项目数与原始数据集一致

            user_datasets[user_id] = user_dataset

        return user_datasets

    def _get_train_loader(self):
        """获取训练数据加载器

        Returns:
            dict: 用户ID到训练数据加载器的映射
        """
        user_datasets = self._get_user_datasets()
        user_loader = {}

        for user_id, user_dataset in user_datasets.items():
            user_loader[user_id] = loader_utils.TrainDataLoader(
                self.config,
                user_dataset,
                batch_size=self.config["train_batch_size"],
                shuffle=self.shuffle,
            )

        return user_loader

    def _get_eval_loader(self):
        """获取评估数据加载器

        Returns:
            dict: 用户ID到评估数据加载器的映射
        """
        assert (
            self.additional_dataset is not None
        ), "additional_dataset should not be None in eval dataloader"

        # 过滤条件：用户必须存在于额外数据集中
        def filter_condition(user_id):
            return (
                user_id
                in self.additional_dataset.df[self.additional_dataset.uid_field].values
            )

        user_datasets = self._get_user_datasets(filter_condition)
        user_loader = {}

        for user_id, user_dataset in user_datasets.items():
            # 获取额外数据集中该用户的数据
            user_additional_df = self.additional_dataset.df[
                self.additional_dataset.df[self.additional_dataset.uid_field] == user_id
            ]

            # 创建额外数据集对象
            user_additional_dataset = RecDataset(self.config, user_additional_df)
            user_additional_dataset.inter_num = len(user_additional_df)
            user_additional_dataset.user_num = 1
            user_additional_dataset.item_num = self.additional_dataset.item_num

            user_loader[user_id] = loader_utils.EvalDataLoader(
                self.config,
                user_dataset,
                batch_size=self.config["eval_batch_size"],
                additional_dataset=user_additional_dataset,
            )

        return user_loader

    def __iter__(self):
        """迭代器方法

        Returns:
            self: 返回自身
        """
        self.user_idx = 0
        if self.shuffle:
            import random

            random.shuffle(self.user_ids)
        return self

    def __next__(self):
        """获取下一个用户的数据加载器

        Returns:
            tuple: (用户ID, 数据加载器)

        Raises:
            StopIteration: 当所有用户都已迭代完毕时
        """
        if self.user_idx >= len(self.user_ids):
            raise StopIteration

        user_id = self.user_ids[self.user_idx]
        loader = self.data_loader[user_id]
        self.user_idx += 1

        return user_id, loader

    def __len__(self):
        """获取用户数量

        Returns:
            int: 用户数量
        """
        return len(self.user_ids)

    def pretrain_setup(self):
        """预训练设置，重置随机状态"""
        # 重置迭代器
        self.user_idx = 0
        # 如果需要打乱，则重新打乱用户ID
        if self.shuffle:
            import random

            random.shuffle(self.user_ids)

        # 对每个用户的数据加载器也进行设置
        for user_id, loader in self.data_loader.items():
            if hasattr(loader, "pretrain_setup"):
                loader.pretrain_setup()

    @property
    def loaders(self):
        """获取所有数据加载器

        Returns:
            dict: 用户ID到数据加载器的映射
        """
        return self.data_loader

    @property
    def user_set(self):
        """获取用户集合

        Returns:
            list: 用户ID列表
        """
        return self.user_ids
