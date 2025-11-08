# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start

os.environ["NUMEXPR_MAX_THREADS"] = "48"


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, default="FedNCF", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="Beauty", help="name of datasets"
    )
    parser.add_argument(
        "--mg",
        action="store_true",
        help="whether to use Mirror Gradient, default is False",
    )
    parser.add_argument("--gpu_id", "-g", type=int, default=0, help="set the gpu id")
    parser.add_argument(
        "--type", "-t", type=str, default="Test", help="variant of the type"
    )
    parser.add_argument(
        "--comment", "-c", type=str, default="Test", help="comment of the experiment"
    )
    parser.add_argument(
        "--id_mode",
        type=str,
        default="None",
        help="ID feature ablation mode: None, remove, or noise",
    )
    parser.add_argument(
        "--txt_mode",
        type=str,
        default="None",
        help="Text feature ablation mode: None, remove, or noise",
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="None",
        help="Visual feature ablation mode: None, remove, or noise",
    )

    parser.add_argument(
        "--fusion_module",
        type=str,
        default="moe",
        help="Fusion module: moe, sum, attention, mlp, gate",
    )


    # parser.add_argument("--alpha", type=float, default=1e-1)
    #
    # parser.add_argument("--beta", type=float, default=1e-1)
    #
    # parser.add_argument("--lr", type=float, default=1e-1)
    #
    # parser.add_argument("--l2_reg", type=float, default=1e-1)


    args, _ = parser.parse_known_args()

    config_dict = {}

    # 将args转为dict，更新到config_dict中
    config_dict.update(vars(args))

    return config_dict, args


if __name__ == "__main__":
    config_dict, args = load_config()

    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config_dict,
        save_model=True,
        mg=args.mg,
    )
