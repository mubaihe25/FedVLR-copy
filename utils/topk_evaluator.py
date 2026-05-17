# coding: utf-8
# @email: enoche.chow@gmail.com
"""
################################
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from utils.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence


# These metrics are typical in topk recommendations
topk_metrics = {
    metric.lower(): metric
    for metric in ["Recall", "Recall2", "Precision", "NDCG", "MAP"]
}


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.metrics = config["metrics"]
        self.topk = config["topk"]
        self.save_recom_result = config["save_recommended_topk"]
        self._recommend_export_counter = 0
        self._check_args()

    @staticmethod
    def _safe_filename_part(value):
        text = str(value)
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)

    @staticmethod
    def _user_ids_for_export(eval_data):
        try:
            user_ids = eval_data.get_eval_users()
        except Exception:
            return []
        if hasattr(user_ids, "detach"):
            user_ids = user_ids.detach().cpu().numpy()
        if hasattr(user_ids, "tolist"):
            user_ids = user_ids.tolist()
        if not isinstance(user_ids, list):
            user_ids = [user_ids]
        return [str(int(value)) if isinstance(value, (int, np.integer)) else str(value) for value in user_ids]

    def _topk_export_path(self, dir_name, model_name, dataset_name, idx, max_k, user_ids):
        self._recommend_export_counter += 1
        if len(user_ids) == 1:
            user_part = "user{}".format(self._safe_filename_part(user_ids[0]))
        elif user_ids:
            user_part = "users{}_first{}_last{}".format(
                len(user_ids),
                self._safe_filename_part(user_ids[0]),
                self._safe_filename_part(user_ids[-1]),
            )
        else:
            user_part = "users0"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = "{}-{}-idx{}-{}-top{}-{}-{:06d}.csv".format(
            self._safe_filename_part(model_name),
            self._safe_filename_part(dataset_name),
            self._safe_filename_part(idx),
            user_part,
            max_k,
            timestamp,
            self._recommend_export_counter,
        )
        return os.path.join(dir_name, filename)

    @staticmethod
    def _update_recommend_topk_manifest(dir_name, file_path, user_ids, max_k, idx):
        manifest_path = os.path.join(dir_name, "recommend_topk_manifest.json")
        try:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8-sig") as handle:
                    manifest = json.load(handle)
            else:
                manifest = {
                    "manifest_type": "recommend_topk_manifest",
                    "topk_files": [],
                    "user_ids": [],
                    "file_count": 0,
                    "warning": None,
                }
            basename = os.path.basename(file_path)
            entry = {
                "file": basename,
                "user_ids": list(user_ids),
                "topk": int(max_k),
                "idx": str(idx),
            }
            manifest.setdefault("topk_files", []).append(entry)
            existing_users = [str(value) for value in manifest.get("user_ids", [])]
            for user_id in user_ids:
                if str(user_id) not in existing_users:
                    existing_users.append(str(user_id))
            manifest["user_ids"] = existing_users
            manifest["file_count"] = len(manifest.get("topk_files", []))
            manifest["generated_at"] = datetime.now().isoformat(timespec="seconds")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
        except Exception:
            return

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(
                scores_list, batch_first=True, padding_value=-np.inf
            )  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        if not batch_matrix_list:
            print("Warning: Empty batch matrix list in evaluation")
            empty_result = {}
            for metric in self.metrics:
                for k in self.topk:
                    key = "{}@{}".format(metric, k)
                    empty_result[key] = 0.0
            return empty_result

        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        if topk_index.ndim == 1:
            topk_index = np.expand_dims(topk_index, axis=0)

        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config["dataset"]
            model_name = self.config["model"]
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config["recommend_topk"])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            user_ids = self._user_ids_for_export(eval_data)
            file_path = self._topk_export_path(
                dir_name, model_name, dataset_name, idx, max_k, user_ids
            )
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, "id", eval_data.get_eval_users())
            x_df.columns = ["id"] + ["top_" + str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep="\t", index=False)
            self._update_recommend_topk_manifest(dir_name, file_path, user_ids, max_k, idx)
        assert len(pos_len_list) == len(topk_index)
        # if recom right?
        bool_rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = "{}@{}".format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)
        return metric_dict

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError("metrics must be str or list")

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError(
                    "There is no user grouped topk metric named {}!".format(m)
                )
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        "topk must be a positive integer or a list of positive integers, but get `{}`".format(
                            topk
                        )
                    )
        else:
            raise TypeError("The topk must be a integer, list")

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def __str__(self):
        mesg = (
            "The TopK Evaluator Info:\n"
            + "\tMetrics:["
            + ", ".join([topk_metrics[metric.lower()] for metric in self.metrics])
            + "], TopK:["
            + ", ".join(map(str, self.topk))
            + "]"
        )
        return mesg
