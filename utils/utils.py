# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import datetime
import importlib
import random

import numpy as np
import pandas as pd
import torch


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y-%H-%M-%S")

    return cur


def get_model(model_name):
    """根据模型名称自动选择模型类

    Args:
        model_name (str): 模型名称

    Returns:
        Recommender: 模型类

    Raises:
        ImportError: 当模块未找到时
        AttributeError: 当模型类未找到时
    """
    try:
        model_file_name = model_name.lower()
        module_path = ".".join(["models", model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            model_class = getattr(model_module, model_name)
            return model_class
        raise ImportError(f"Module {module_path} not found")
    except (ImportError, AttributeError) as e:
        raise type(e)(f"Failed to load model {model_name}: {str(e)}")


def get_trainer(alias=None, is_federated=False):
    """获取训练器类

    Args:
        alias (str, optional): 模型别名
        is_federated (bool, optional): 是否使用联邦学习

    Returns:
        Trainer: 训练器类

    Raises:
        ValueError: 当联邦学习需要但未提供别名时
        ImportError: 当模块未找到时
        AttributeError: 当训练器类未找到时
    """
    try:
        if is_federated:
            if not alias:
                raise ValueError("Alias must be provided for federated training")

            model_name = alias.lower()
            module_path = ".".join(["models", model_name])
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)
                return getattr(model_module, f"{alias}Trainer")
            raise ImportError(f"Module {module_path} not found")
        else:
            return getattr(importlib.import_module("common.trainer"), "Trainer")
    except (ImportError, AttributeError, ValueError) as e:
        raise type(e)(f"Failed to load trainer for {alias}: {str(e)}")


def init_seed(seed):
    """初始化随机种子以确保结果可重现

    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False

    # 统一判断是否更好的逻辑
    is_better = (value > best) if bigger else (value < best)

    if is_better:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True

    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ""
    for metric, value in result_dict.items():
        result_str += str(metric) + ": " + "%.4f" % value + ", "
    return result_str


############ LATTICE Utilities #########


def build_knn_neighbourhood(adj, topk):
    """构建KNN邻域

    Args:
        adj (torch.Tensor): 邻接矩阵
        topk (int): 保留的最近邻数量

    Returns:
        torch.Tensor: 加权邻接矩阵
    """
    # 确保topk不超过矩阵维度
    topk = min(topk, adj.shape[1])
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    """构建相似度矩阵

    Args:
        context (torch.Tensor): 上下文张量

    Returns:
        torch.Tensor: 相似度矩阵
    """
    # 使用F.normalize优化归一化操作
    import torch.nn.functional as F

    context_norm = F.normalize(context, p=2, dim=-1)
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization="none"):
    try:
        from torch_scatter import scatter_add
    except ImportError:
        # 如果torch_scatter不可用，使用替代实现
        def scatter_add(src, index, dim=0, dim_size=None):
            """手动实现scatter_add功能"""
            if dim_size is None:
                dim_size = index.max().item() + 1
            result = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
            result.index_add_(dim, index, src)
            return result

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization="none"):
    if normalization == "sym":
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == "rw":
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == "none":
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    """构建KNN归一化图

    Args:
        adj (torch.Tensor): 邻接矩阵
        topk (int): 保留的最近邻数量
        is_sparse (bool): 是否使用稀疏表示
        norm_type (str): 归一化类型

    Returns:
        torch.Tensor: 归一化图
    """
    device = adj.device
    # 确保topk不超过矩阵维度
    topk = min(topk, adj.shape[1])
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)

    if is_sparse:
        # 使用列表推导式优化
        rows, cols = [], []
        for row_idx, cols_idx in enumerate(knn_ind):
            rows.extend([row_idx] * len(cols_idx))
            cols.extend(cols_idx.tolist())

        i = torch.LongTensor([rows, cols]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(
            i, v, normalization=norm_type, num_nodes=adj.shape[0]
        )
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def sampleClients(
    client_list, sample_strategy="random", sample_ratio=0.1, last_clients=None
):
    """从客户端列表中采样

    Args:
        client_list (list): 客户端ID列表
        sample_strategy (str, optional): 采样策略. 默认为'random'.
        sample_ratio (float, optional): 采样比例. 默认为0.1.
        last_clients (list, optional): 上一轮采样的客户端. 默认为None.

    Returns:
        list: 采样的客户端ID列表

    Raises:
        ValueError: 如果采样策略无效
    """
    # 使用所有客户端
    if sample_ratio >= 1:
        return client_list.copy()  # 返回副本以避免修改原始列表

    # 计算要采样的客户端数量
    sample_num = max(1, int(len(client_list) * sample_ratio))

    # 移除上一轮采样的客户端
    available_clients = client_list
    if last_clients:
        # 使用集合操作更高效
        available_clients = list(set(client_list) - set(last_clients))
        # 如果可用客户端不足，则使用所有客户端
        if len(available_clients) < sample_num:
            available_clients = client_list

    if sample_strategy == "random":
        # 随机采样客户端
        return random.sample(available_clients, min(sample_num, len(available_clients)))
    else:
        raise ValueError(f"Invalid sample strategy: {sample_strategy}")


def get_combinations(config, result_file):
    """获取超参数的所有组合

    Args:
        config (dict): 包含所有配置的字典
        result_file (str): 保存结果的CSV文件

    Returns:
        tuple: 超参数组合和总循环次数
    """
    from itertools import product
    import os

    # 尝试读取现有的CSV文件，获取已经运行过的参数组合
    existing_combinations = []
    if os.path.exists(result_file):
        try:
            df = pd.read_csv(result_file)
            if all(param in df.columns for param in config["hyper_parameters"]):
                existing_combinations = df[config["hyper_parameters"]].to_dict(
                    "records"
                )
        except Exception as e:
            print(f"Error reading existing combinations: {e}")

    # 超参数
    hyper_ls = [config.get(param, []) or [None] for param in config["hyper_parameters"]]

    # 组合
    combinators = list(product(*hyper_ls))

    # 将所有组合转为字典格式以便比较
    combination_dicts = [
        dict(zip(config["hyper_parameters"], comb)) for comb in combinators
    ]

    # 使用集合操作优化过滤
    existing_set = {tuple(sorted(d.items())) for d in existing_combinations}
    remaining_combinations = [
        comb
        for comb in combination_dicts
        if tuple(sorted(comb.items())) not in existing_set
    ]

    comb_tuple = [list(d.values()) for d in remaining_combinations]
    total_loops = len(remaining_combinations)

    return comb_tuple, total_loops


def _csv_safe_value(value):
    """Convert nested values into stable CSV-friendly cells."""
    import json

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _normalize_metric_key(key):
    return str(key).lower().replace("@", "").replace("_", "").replace("-", "")


def _read_metric_value(source, metric_name, cutoff):
    """Read metric values from dicts using keys like recall@50 or recall50."""
    if not isinstance(source, dict):
        return None

    target = "{}{}".format(metric_name, cutoff)
    for key, value in source.items():
        if _normalize_metric_key(key) == target:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _nested_count_sum(metrics, field_names):
    if not isinstance(metrics, dict):
        return 0

    total = 0
    for value in metrics.values():
        if not isinstance(value, dict):
            continue
        for field_name in field_names:
            raw_value = value.get(field_name)
            if raw_value is None:
                continue
            try:
                total += int(raw_value)
                break
            except (TypeError, ValueError):
                continue
    return total


def _round_metric_payload(round_metric):
    if not isinstance(round_metric, dict):
        return {}
    extra = round_metric.get("extra") if isinstance(round_metric.get("extra"), dict) else {}
    valid_result = (
        extra.get("valid_result")
        if isinstance(extra.get("valid_result"), dict)
        else round_metric.get("valid_result")
    )
    test_result = (
        extra.get("test_result")
        if isinstance(extra.get("test_result"), dict)
        else round_metric.get("test_result")
    )
    attack_metrics = extra.get("attack_metrics") if isinstance(extra.get("attack_metrics"), dict) else {}
    defense_metrics = extra.get("defense_metrics") if isinstance(extra.get("defense_metrics"), dict) else {}

    return {
        "round_index": round_metric.get("round_index") or round_metric.get("round_id"),
        "round_id": round_metric.get("round_id") or round_metric.get("round_index"),
        "train_loss": (
            round_metric.get("train_loss")
            if round_metric.get("train_loss") is not None
            else round_metric.get("avg_train_loss")
        ),
        "valid_score": round_metric.get("valid_score"),
        "test_score": round_metric.get("test_score"),
        "valid_recall50": _read_metric_value(valid_result, "recall", 50),
        "valid_ndcg50": _read_metric_value(valid_result, "ndcg", 50),
        "test_recall50": _read_metric_value(test_result, "recall", 50),
        "test_ndcg50": _read_metric_value(test_result, "ndcg", 50),
        "participant_count": (
            round_metric.get("participant_count")
            if round_metric.get("participant_count") is not None
            else round_metric.get("num_participants")
        ),
        "malicious_client_count": round_metric.get("malicious_client_count"),
        "attacked_client_count": _nested_count_sum(
            attack_metrics, ("attacked_client_count", "poisoned_client_count")
        ),
        "clipped_client_count": _nested_count_sum(
            defense_metrics, ("clipped_client_count", "total_clipped_clients")
        ),
        "filtered_client_count": _nested_count_sum(
            defense_metrics, ("filtered_client_count", "total_filtered_clients")
        ),
    }


def _best_metric(rows, field_name):
    candidates = []
    for row in rows:
        value = row.get(field_name)
        if value is None or value == "":
            continue
        try:
            candidates.append((float(value), row.get("round_index")))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def _build_round_csv_rows(param_dict, round_metrics):
    rows = []
    base_params = {key: _csv_safe_value(value) for key, value in param_dict.items()}
    for round_metric in round_metrics or []:
        payload = _round_metric_payload(round_metric)
        rows.append(
            {
                **base_params,
                "row_type": "round",
                "best_source": "",
                "best_recall50": "",
                "best_ndcg50": "",
                "best_round_for_recall50": "",
                "best_round_for_ndcg50": "",
                **payload,
            }
        )

    if not rows:
        return []

    test_recall, test_recall_round = _best_metric(rows, "test_recall50")
    test_ndcg, test_ndcg_round = _best_metric(rows, "test_ndcg50")
    valid_recall, valid_recall_round = _best_metric(rows, "valid_recall50")
    valid_ndcg, valid_ndcg_round = _best_metric(rows, "valid_ndcg50")

    has_test_metrics = test_recall is not None or test_ndcg is not None
    best_source = "test" if has_test_metrics else "valid"
    rows.append(
        {
            **base_params,
            "row_type": "best_summary",
            "round_index": "",
            "round_id": "",
            "train_loss": "",
            "valid_score": "",
            "test_score": "",
            "valid_recall50": "",
            "valid_ndcg50": "",
            "test_recall50": "",
            "test_ndcg50": "",
            "participant_count": "",
            "malicious_client_count": "",
            "attacked_client_count": "",
            "clipped_client_count": "",
            "filtered_client_count": "",
            "best_source": best_source,
            "best_recall50": test_recall if has_test_metrics else valid_recall,
            "best_ndcg50": test_ndcg if has_test_metrics else valid_ndcg,
            "best_round_for_recall50": test_recall_round if has_test_metrics else valid_recall_round,
            "best_round_for_ndcg50": test_ndcg_round if has_test_metrics else valid_ndcg_round,
        }
    )
    return rows


def save_experiment_results(
    param_dict,
    result_dict,
    csv_filename="experiment_results.csv",
    round_metrics=None,
    experiment_result_dict=None,
):
    """保存实验结果到CSV文件

    Args:
        param_dict (dict): 参数字典
        result_dict (dict): 结果字典
        csv_filename (str, optional): CSV文件名. 默认为'experiment_results.csv'.
    """
    import os
    import pandas as pd

    if round_metrics is None and isinstance(experiment_result_dict, dict):
        round_metrics = experiment_result_dict.get("round_metrics")

    round_rows = _build_round_csv_rows(param_dict, round_metrics)
    if round_rows:
        os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
        new_df = pd.DataFrame(round_rows)
        try:
            existing_df = pd.read_csv(csv_filename)
        except FileNotFoundError:
            existing_df = pd.DataFrame()
        df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if not existing_df.empty
            else new_df
        )
        df.to_csv(csv_filename, index=False)
        return

    # 合并参数和实验结果为一条记录
    record = {
        **{key: _csv_safe_value(value) for key, value in param_dict.items()},
        **{key: _csv_safe_value(value) for key, value in result_dict.items()},
    }

    # 尝试读取现有的CSV文件，如果文件不存在，则创建一个新的DataFrame
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    # 将参数字典转为DataFrame行
    new_row = pd.DataFrame([record])

    # 检查是否存在相同的参数组合
    if not df.empty and all(col in df.columns for col in param_dict):
        # 使用更健壮的方式找到匹配行
        match_mask = pd.Series(True, index=df.index)
        for col, val in param_dict.items():
            # 确保列存在且值类型匹配
            if col in df.columns:
                # 将值转换为字符串进行比较，避免类型问题
                match_mask = match_mask & (df[col].astype(str) == str(val))

        # 找到匹配的行
        match = df[match_mask]

        if not match.empty:
            # 如果找到了相同的参数组合，获取行索引
            index = match.index[0]
            # 更新行
            for col, val in record.items():
                if col in df.columns:
                    df.at[index, col] = val
                else:
                    df[col] = None  # 添加新列
                    df.at[index, col] = val
        else:
            # 如果没有找到相同的参数组合，添加新行
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # 如果文件是新的或没有匹配列，直接追加数据
        df = pd.concat([df, new_row], ignore_index=True)

    # 确保目录存在
    os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)

    # 将结果写入CSV文件
    df.to_csv(csv_filename, index=False)


def save_experiment_json_outputs(
    config,
    experiment_result_dict,
    experiment_summary_dict,
):
    """Save detailed and summary experiment outputs as JSON files."""
    import json
    import os

    result_file_name = config.get("result_file_name")
    if not result_file_name:
        raise ValueError("config['result_file_name'] is required for JSON export")

    output_dir = os.path.dirname(result_file_name) or "."
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(result_file_name))[0]
    detail_path = os.path.join(output_dir, f"{base_name}.experiment_result.json")
    summary_path = os.path.join(output_dir, f"{base_name}.experiment_summary.json")

    dump_kwargs = {
        "ensure_ascii": False,
        "indent": 2,
        "default": str,
    }

    with open(detail_path, "w", encoding="utf-8") as detail_file:
        json.dump(experiment_result_dict, detail_file, **dump_kwargs)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(experiment_summary_dict, summary_file, **dump_kwargs)

    return {
        "experiment_result_path": detail_path,
        "experiment_summary_path": summary_path,
    }


def find_best_parameters(
    csv_filename="experiment_results.csv", metric="ndcg@10", maximize=True
):
    """
    从CSV文件中读取所有实验结果，并根据指定的评价指标选取最优的参数组合。
    :param csv_filename: str, 存储实验结果的CSV文件名
    :param metric: str, 需要优化的评价指标名称
    :param maximize: bool, 如果为True，表示需要最大化指标；如果为False，表示需要最小化指标
    :return: dict, 包含最优参数组合及其对应的评价指标
    """
    # 从CSV文件读取数据
    df = pd.read_csv(csv_filename)

    # 检查指定的评价指标是否存在
    if metric not in df.columns:
        raise ValueError(f"指定的评价指标 '{metric}' 不存在，请检查 CSV 文件中的列名。")

    # 根据评价指标选择最优的参数组合
    if maximize:
        best_row = df.loc[df[metric].idxmax()]  # 最大化指标
    else:
        best_row = df.loc[df[metric].idxmin()]  # 最小化指标

    # 提取最优参数组合及其评价指标
    best_parameters = best_row.to_dict()

    return best_parameters


def mail_notice(config, content=None):
    """
    Email notice that the training is finished.
    :param config: the input arguments
    :return: None
    """
    import iMail
    from configs import private as uc
    import yaml

    # Set the Mail Sender
    mail_obj = iMail.EMAIL(
        host=uc.EMAIL_HOST,
        sender_addr=uc.SENDER_ADDRESS,
        pwd=uc.PASSWORD,
        sender_name=uc.SENDER_NAME,
    )
    mail_obj.set_receiver(uc.RECEIVER)

    # Create a new email
    mail_title = "[NOTICE FROM EXPERIMENT] {} on {}: {}-{}".format(
        config["model"], config["dataset"].upper(), config["type"], config["comment"]
    )
    mail_obj.new_mail(subject=mail_title, encoding="UTF-8")

    # Attach a text to the receiver
    mail_obj.add_text(content=yaml.dump(content))

    # Send the email
    mail_obj.send_mail()

    return mail_title, content


def dp_step(param, threshold=1.0, sigma=1.0):
    """差分隐私梯度步骤

    Args:
        param (torch.Tensor): 参数张量
        threshold (float, optional): 裁剪阈值. 默认为1.0.
        sigma (float, optional): 噪声标准差. 默认为1.0.
    """
    # 计算梯度范数
    grad_norm = torch.norm(param).item()

    # 计算裁剪值
    clip_value = threshold / max(1.0, grad_norm / threshold)

    # 梯度裁剪
    param.data.mul_(clip_value)

    # 添加噪声
    noise = torch.normal(0, sigma * threshold, size=param.shape, device=param.device)
    param.data.add_(noise)


def modal_ablation(
    item_embed,
    txt_embed,
    vision_embed,
    txt_mode=None,
    vis_mode=None,
    id_mode=None,
    txt_noise_scale=1.0,
    vis_noise_scale=1.0,
    txt_noise_type="gaussian",
    vis_noise_type="gaussian",
    device=None,
):
    """多模态消融测试函数

    Args:
        item_embed (torch.Tensor): 物品嵌入特征
        txt_embed (torch.Tensor): 文本嵌入特征
        vision_embed (torch.Tensor): 视觉嵌入特征
        txt_mode (str, optional): 文本消融模式, 可选值为 None, 'remove', 'noise'
        vis_mode (str, optional): 视觉消融模式, 可选值为 None, 'remove', 'noise'
        id_mode (str, optional): ID消融模式, 可选值为 None, 'remove', 'noise'
        txt_noise_scale (float): 文本噪声尺度, 当txt_mode为'noise'时使用
        vis_noise_scale (float): 视觉噪声尺度, 当vis_mode为'noise'时使用
        txt_noise_type (str): 文本噪声类型, 'gaussian'或'uniform'
        vis_noise_type (str): 视觉噪声类型, 'gaussian'或'uniform'
        device (torch.device, optional): 设备, 如果为None则使用item_embed的设备

    Returns:
        tuple: (processed_id, processed_txt, processed_vision)
    """

    # 标准化模式输入(将"None"字符串转为None)
    def normalize_mode(mode):
        if isinstance(mode, str) and mode.lower() == "none":
            return None
        return mode

    txt_mode = normalize_mode(txt_mode)
    vis_mode = normalize_mode(vis_mode)
    id_mode = normalize_mode(id_mode)

    # 验证参数
    valid_modes = [None, "remove", "noise"]
    valid_noise_types = ["gaussian", "uniform"]

    for name, mode in [
        ("txt_mode", txt_mode),
        ("vis_mode", vis_mode),
        ("id_mode", id_mode),
    ]:
        if mode not in valid_modes:
            raise ValueError(f"无效的{name}: {mode}, 支持的值为: {valid_modes}")

    for name, noise_type in [
        ("txt_noise_type", txt_noise_type),
        ("vis_noise_type", vis_noise_type),
    ]:
        if noise_type not in valid_noise_types:
            raise ValueError(
                f"无效的{name}: {noise_type}, 支持的值为: {valid_noise_types}"
            )

    # 确定设备和数据类型
    if device is None:
        device = item_embed.device
    dtype = item_embed.dtype

    # 辅助函数：生成噪声
    def generate_noise(tensor, noise_type, noise_scale):
        shape = tensor.shape
        if noise_type == "gaussian":
            return torch.randn(shape, device=device, dtype=dtype) * noise_scale
        else:  # uniform
            return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1) * noise_scale

    # 处理各模态 - 只在需要修改时才克隆
    processed_id = item_embed
    if id_mode == "remove":
        # 使用随机浮点数而不是整数
        processed_id = torch.rand(item_embed.shape, device=device, dtype=dtype)
    elif id_mode == "noise":
        processed_id = generate_noise(item_embed, "gaussian", 1.0)

    processed_txt = txt_embed
    if txt_mode == "remove":
        processed_txt = torch.zeros_like(txt_embed, dtype=dtype)
    elif txt_mode == "noise":
        processed_txt = generate_noise(txt_embed, txt_noise_type, txt_noise_scale)

    processed_vision = vision_embed
    if vis_mode == "remove":
        processed_vision = torch.zeros_like(vision_embed, dtype=dtype)
    elif vis_mode == "noise":
        processed_vision = generate_noise(vision_embed, vis_noise_type, vis_noise_scale)

    return processed_id, processed_txt, processed_vision


def get_resource_usage_gb():
    """
    获取当前进程内存、总内存、当前进程显存、总显存（单位GB）
    Returns:
        dict: { 'used_mem_gb', 'total_mem_gb', 'used_gpu_gb', 'total_gpu_gb' }
    """
    import psutil
    import torch
    process = psutil.Process()
    used_mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    total_mem_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    if torch.cuda.is_available():
        used_gpu_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    else:
        used_gpu_gb = total_gpu_gb = 0
    return {
        'used_mem_gb': used_mem_gb,
        'total_mem_gb': total_mem_gb,
        'used_gpu_gb': used_gpu_gb,
        'total_gpu_gb': total_gpu_gb
    }
