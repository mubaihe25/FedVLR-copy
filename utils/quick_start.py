# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
import os
import pickle
import platform
from logging import getLogger
from typing import Tuple, List, Dict, Any, Optional
import torch

from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader, FederatedDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import (
    init_seed,
    get_model,
    get_trainer,
    dict2str,
    get_combinations,
    save_experiment_results,
    find_best_parameters,
    mail_notice,
)


def _prepare_data(config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """准备数据加载器

    Args:
        config: 配置字典

    Returns:
        训练、验证和测试数据加载器
    """
    # 加载数据集
    dataset = RecDataset(config)

    config["count_user_inter"] = dataset.count_user_inter

    logger = getLogger()
    # 打印数据集统计信息
    logger.info(f">>> [{config['dataset']} stats] Overall: {dataset}")

    # 分割数据集
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info(
        ">>> [{} stats] Train:   ".format(config["dataset"]) + str(train_dataset)
    )
    logger.info(
        ">>> [{} stats] Valid:   ".format(config["dataset"]) + str(valid_dataset)
    )
    logger.info(
        ">>> [{} stats] Test:    ".format(config["dataset"]) + str(test_dataset)
    )

    # 包装为数据加载器
    if config["is_federated"]:
        train_data = FederatedDataLoader(
            config, train_dataset, batch_size=config["train_batch_size"], shuffle=True
        )
        valid_data = FederatedDataLoader(
            config,
            valid_dataset,
            additional_dataset=train_dataset,
            stage="valid",
            batch_size=config["eval_batch_size"],
        )
        test_data = FederatedDataLoader(
            config,
            test_dataset,
            additional_dataset=train_dataset,
            stage="test",
            batch_size=config["eval_batch_size"],
        )
    else:
        train_data = TrainDataLoader(
            config, train_dataset, batch_size=config["train_batch_size"], shuffle=True
        )
        valid_data = EvalDataLoader(
            config,
            valid_dataset,
            additional_dataset=train_dataset,
            batch_size=config["eval_batch_size"],
        )
        test_data = EvalDataLoader(
            config,
            test_dataset,
            additional_dataset=train_dataset,
            batch_size=config["eval_batch_size"],
        )

    return train_data, valid_data, test_data


def _to_cpu_recursive(data: Any) -> Any:
    """递归地将所有张量移到CPU

    Args:
        data: 任意数据结构

    Returns:
        处理后的数据结构，所有张量都在CPU上
    """
    if isinstance(data, torch.Tensor):
        return data.cpu() if data.device.type != "cpu" else data
    elif isinstance(data, dict):
        return {key: _to_cpu_recursive(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_to_cpu_recursive(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu_recursive(item) for item in data)
    elif isinstance(data, set):
        return {_to_cpu_recursive(item) for item in data}
    return data


def _save_model_params(config: Dict[str, Any], trainer: Any) -> None:
    """保存模型参数

    Args:
        config: 配置字典
        trainer: 训练器实例
    """
    logger = getLogger()
    try:
        model_dir = config["model_dir"].format(config["type"], config["comment"])

        # 提取需要保存的参数
        save_params = {}

        if config["save_model"]:
            save_params = {
                "t_feat": getattr(trainer, "t_feat", None),
                "v_feat": getattr(trainer, "v_feat", None),
                "train_loss": getattr(trainer, "train_loss_dict", {}),
            }

            if config["is_federated"]:
                save_params["client_models"] = getattr(trainer, "client_models", None)
                save_params["fusion"] = getattr(trainer, "fusion", None)
                save_params["item_commonality"] = getattr(
                    trainer, "item_commonality", None
                )
                
        if config["save_results"]:
            save_params["count_user_inter"] = config["count_user_inter"]
            save_params["train_loss"] = getattr(trainer, "train_loss_dict", {})
            save_params["eval"] = config["eval"]
            if config["is_federated"]:
                save_params["user_metrics"] = getattr(trainer, "user_metrics", {})

        if len(save_params) > 0:
            # 递归地将所有张量移到CPU
            save_params = _to_cpu_recursive(save_params)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)

            # 使用with语句确保文件正确关闭
            with open(model_dir, "wb") as f:
                pickle.dump(save_params, f)

            logger.info(f"Model saved to {model_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")


def _train_and_evaluate(
    config: Dict[str, Any],
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    save_model: bool = True,
    mg: bool = False,
) -> Tuple[List[Tuple], int]:
    """训练和评估模型

    Args:
        config: 配置字典
        train_data: 训练数据加载器
        valid_data: 验证数据加载器
        test_data: 测试数据加载器
        save_model: 是否保存模型
        mg: 是否使用多图

    Returns:
        超参数组合及其结果列表和最佳测试索引
    """
    logger = getLogger()
    hyper_ret = []
    val_metric = config["valid_metric"].lower()
    best_test_value = float("-inf")  # 使用负无穷初始化，适用于任何指标
    best_test_idx = 0

    logger.info("=" * 25 + " Combining Hyper-Parameters " + "=" * 25)

    # 获取所有超参数组合
    combinators, total_loops = get_combinations(config, config["result_file_name"])

    # 如果没有组合需要运行，提前返回
    if not combinators:
        logger.info("No new parameter combinations to evaluate.")
        return [], 0

    # 跟踪失败的参数组合
    failed_combinations = []

    # 遍历所有超参数组合找到最佳组合
    for idx, hyper_tuple in enumerate(combinators):
        # 更新配置
        hyper_dict = dict(zip(config["hyper_parameters"], hyper_tuple))
        for param, value in hyper_dict.items():
            config[param] = value

        # 重置随机种子
        init_seed(config["seed"])

        logger.info(
            "=" * 15 + f" [{idx + 1}/{total_loops} Parameter Combination] "
            f'{config["hyper_parameters"]}={hyper_tuple} ' + "=" * 15
        )

        try:
            # 设置数据加载器的随机状态
            train_data.pretrain_setup()

            # 加载和初始化模型
            model = get_model(config["model"])(config, train_data).to(config["device"])
            logger.info(model)

            # 加载和初始化训练器
            trainer = get_trainer(config["model"], config["is_federated"])(
                config, model, mg
            )

            # 模型训练
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
                train_data, valid_data=valid_data, test_data=test_data, saved=save_model
            )
            
            config["eval"] = best_test_upon_valid

            # 保存模型
            _save_model_params(config, trainer)

            # 记录结果
            hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

            # 更新最佳测试结果
            if best_test_upon_valid[val_metric] > best_test_value:
                best_test_value = best_test_upon_valid[val_metric]
                best_test_idx = idx

            # 记录日志
            logger.info(f"[Best Valid]: {dict2str(best_valid_result)}")
            logger.info(f"[Best Test]:  {dict2str(best_test_upon_valid)}")
            logger.info(
                f"████ Current BEST ████:\n"
                f'Parameters: {config["hyper_parameters"]}={hyper_ret[best_test_idx][0]},\n'
                f"Valid: {dict2str(hyper_ret[best_test_idx][1])},\n"
                f"Test:  {dict2str(hyper_ret[best_test_idx][2])}\n"
            )

            # 保存当前超参数组合的结果
            save_experiment_results(
                hyper_dict, best_test_upon_valid, config["result_file_name"]
            )
            logger.info(f'Results saved to {config["result_file_name"]}')

        except Exception as e:
            logger.error(f"Error in training with parameters {hyper_tuple}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())  # 打印完整堆栈跟踪

            # 记录失败的参数组合
            failed_combinations.append((hyper_tuple, str(e)))

            # 继续下一个参数组合，但不添加空结果

    # 如果有失败的组合，记录它们
    if failed_combinations:
        logger.warning("以下参数组合训练失败:")
        for comb, error in failed_combinations:
            logger.warning(f"参数: {comb}, 错误: {error}")

    if len(hyper_ret) == 0:
        logger.warning("所有参数组合均训练失败。")
        return [], 0

    return hyper_ret, best_test_idx


def quick_start(
    model: str,
    dataset: str,
    config_dict: Dict[str, Any],
    save_model: bool = True,
    mg: bool = False,
) -> None:
    """快速启动训练和评估过程

    Args:
        model: 模型名称
        dataset: 数据集名称
        config_dict: 配置字典
        save_model: 是否保存模型
        mg: 是否使用多图
    """
    # 合并配置字典
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()

    # 打印配置信息
    logger.info(f"██ Directory: {os.getcwd()} on Server: {platform.node()} ██")
    logger.info(config)

    try:
        # 准备数据
        train_data, valid_data, test_data = _prepare_data(config)

        # 训练和评估模型
        hyper_ret, best_test_idx = _train_and_evaluate(
            config, train_data, valid_data, test_data, save_model, mg
        )

        # 如果没有结果，提前返回
        if not hyper_ret:
            logger.info("No results to report.")
            return

        # 记录所有结果
        logger.info("============All Over=====================")
        for p, k, v in hyper_ret:
            logger.info(
                f'Parameters: {config["hyper_parameters"]}={p},\n'
                f"Best Valid: {dict2str(k)},\n"
                f"Best Test:  {dict2str(v)}"
            )

        # 记录最佳组合
        if best_test_idx < len(hyper_ret):
            logger.info(
                f'█████████████ {config["model"]} on {config["dataset"]} - BEST COMBINATION ████████████████'
            )
            logger.info(
                f'\tParameters: {config["hyper_parameters"]}={hyper_ret[best_test_idx][0]},\n'
                f"Valid: \t{dict2str(hyper_ret[best_test_idx][1])},\n"
                f"Test: \t{dict2str(hyper_ret[best_test_idx][2])}\n\n"
            )

        # 查找最佳超参数组合
        best_result = find_best_parameters(
            config["result_file_name"], metric=config["valid_metric"].lower()
        )
        notice_info = f"Best hyper-parameter combination: {dict2str(best_result)}"

        # 发送通知
        if config["notice"]:
            mail_notice(config, notice_info)
            logger.info("Sending notice email success!")

        logger.info(notice_info)

    except Exception as e:
        import traceback

        logger.error(f"Error in quick_start: {str(e)}")
        logger.error(traceback.format_exc())  # 打印完整堆栈跟踪
        raise
