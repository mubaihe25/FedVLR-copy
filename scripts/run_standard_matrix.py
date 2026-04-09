from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.configurator import Config
from utils.logger import init_logger
from utils.quick_start import _prepare_data
from utils.utils import get_model, get_trainer, init_seed


SCENARIO_ORDER = [
    "baseline",
    "attack_only_scale",
    "attack_only_sign_flip",
    "attack_and_defense_clip",
    "attack_and_defense_filter",
]


def _build_comment(prefix: str, scenario_name: str) -> str:
    prefix = prefix.strip()
    return "{}.{}".format(prefix, scenario_name) if prefix else scenario_name


def build_scenarios(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    common_config: Dict[str, Any] = {
        "use_gpu": bool(args.use_gpu),
        "seed": int(args.seed),
        "type": args.type,
        "hyper_parameters": [],
        "alpha": 1e-1,
        "beta": 1e-1,
        "epochs": int(args.epochs),
        "local_epochs": int(args.local_epochs),
        "clients_sample_ratio": float(args.clients_sample_ratio),
        "eval_step": 1,
        "collect_round_metrics": True,
        "enable_experiment_hooks": False,
    }

    def scenario(
        name: str,
        goal: str,
        config_updates: Dict[str, Any],
        expected_mode: str,
        scenario_tags: List[str],
        focus_fields: List[str],
    ) -> Dict[str, Any]:
        config = deepcopy(common_config)
        config.update(config_updates)
        config["comment"] = _build_comment(args.comment_prefix, name)
        return {
            "name": name,
            "goal": goal,
            "model": args.model,
            "dataset": args.dataset,
            "config": config,
            "expected_experiment_mode": expected_mode,
            "expected_scenario_tags": scenario_tags,
            "focus_fields": focus_fields,
        }

    return {
        "baseline": scenario(
            name="baseline",
            goal="建立没有攻击、没有主动防御的基线结果，作为后续对照起点。",
            config_updates={
                "enabled_attacks": [],
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": False,
            },
            expected_mode="baseline",
            scenario_tags=["baseline"],
            focus_fields=[
                "experiment_mode",
                "scenario_tags",
                "final_eval",
                "round_summaries[*].avg_train_loss",
            ],
        ),
        "attack_only_scale": scenario(
            name="attack_only_scale",
            goal="观察轻量缩放攻击对结果场景标签和最终指标的影响。",
            config_updates={
                "enabled_attacks": ["client_update_scale"],
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": True,
                "malicious_client_mode": "ratio",
                "malicious_client_ratio": float(args.malicious_client_ratio),
                "attack_scale": float(args.attack_scale),
            },
            expected_mode="attack_only",
            scenario_tags=["attack_only", "malicious_clients_configured"],
            focus_fields=[
                "active_attacks",
                "malicious_client_summary",
                "round_summaries[*].malicious_client_count",
                "round_metrics[*].extra.attack_metrics.client_update_scale",
            ],
        ),
        "attack_only_sign_flip": scenario(
            name="attack_only_sign_flip",
            goal="观察更接近经典联邦更新攻击的符号翻转效果。",
            config_updates={
                "enabled_attacks": ["sign_flip"],
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": True,
                "malicious_client_mode": "ratio",
                "malicious_client_ratio": float(args.malicious_client_ratio),
                "sign_flip_scale": float(args.sign_flip_scale),
            },
            expected_mode="attack_only",
            scenario_tags=["attack_only", "malicious_clients_configured"],
            focus_fields=[
                "active_attacks",
                "malicious_client_summary",
                "round_summaries[*].malicious_client_count",
                "round_metrics[*].extra.attack_metrics.sign_flip",
            ],
        ),
        "attack_and_defense_clip": scenario(
            name="attack_and_defense_clip",
            goal="形成缩放攻击与范数裁剪防御的最小攻防闭环。",
            config_updates={
                "enabled_attacks": ["client_update_scale"],
                "enabled_defenses": ["norm_clip"],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": True,
                "malicious_client_mode": "ratio",
                "malicious_client_ratio": float(args.malicious_client_ratio),
                "attack_scale": float(args.attack_scale),
                "defense_clip_norm": float(args.defense_clip_norm),
            },
            expected_mode="attack_and_defense",
            scenario_tags=["attack_and_defense", "malicious_clients_configured"],
            focus_fields=[
                "active_attacks",
                "active_defenses",
                "round_summaries[*].attacked_client_count",
                "round_summaries[*].clipped_client_count",
            ],
        ),
        "attack_and_defense_filter": scenario(
            name="attack_and_defense_filter",
            goal="形成符号翻转攻击与更新过滤防御的最小攻防对照。",
            config_updates={
                "enabled_attacks": ["sign_flip"],
                "enabled_defenses": ["update_filter"],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": True,
                "malicious_client_mode": "ratio",
                "malicious_client_ratio": float(args.malicious_client_ratio),
                "sign_flip_scale": float(args.sign_flip_filter_scale),
                "filter_rule": "update_norm > mean + filter_std_factor * std",
                "filter_std_factor": float(args.filter_std_factor),
                "max_filtered_ratio": float(args.max_filtered_ratio),
            },
            expected_mode="attack_and_defense",
            scenario_tags=["attack_and_defense", "malicious_clients_configured"],
            focus_fields=[
                "active_attacks",
                "active_defenses",
                "round_summaries[*].attacked_client_count",
                "round_metrics[*].extra.defense_metrics.update_filter",
            ],
        ),
    }


def derive_output_paths(config: Config) -> Dict[str, str]:
    result_file_name = Path(config["result_file_name"])
    base_name = result_file_name.stem
    return {
        "csv": str(result_file_name),
        "experiment_result_json": str(
            result_file_name.with_name("{}.experiment_result.json".format(base_name))
        ),
        "experiment_summary_json": str(
            result_file_name.with_name("{}.experiment_summary.json".format(base_name))
        ),
    }


def reset_logging() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def run_single_scenario(
    model_name: str,
    dataset_name: str,
    scenario_spec: Dict[str, Any],
) -> Dict[str, Any]:
    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_dict=scenario_spec["config"],
        mg=False,
    )
    reset_logging()
    init_logger(config)
    init_seed(config["seed"])

    train_data, valid_data, test_data = _prepare_data(config)
    model = get_model(config["model"])(config, train_data).to(config["device"])
    trainer = get_trainer(config["model"], config["is_federated"])(config, model, False)
    trainer.fit(
        train_data,
        valid_data=valid_data,
        test_data=test_data,
        saved=False,
    )

    return {
        "scenario": scenario_spec["name"],
        "goal": scenario_spec["goal"],
        "expected_experiment_mode": scenario_spec["expected_experiment_mode"],
        "expected_scenario_tags": scenario_spec["expected_scenario_tags"],
        "focus_fields": scenario_spec["focus_fields"],
        "config": scenario_spec["config"],
        "result": trainer.experiment_result_dict,
        "summary": trainer.experiment_summary_dict,
        "output_paths": derive_output_paths(config),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal standard experiment matrix for FedRAP/KU."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIO_ORDER,
        help="Run only selected scenarios. Repeat this flag to run multiple scenarios.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Only print the available scenario names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configurations without launching training.",
    )
    parser.add_argument("--model", default="FedRAP")
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--type", default="StandardMatrix")
    parser.add_argument(
        "--comment-prefix",
        default="matrix",
        help="Prefix inserted into each scenario comment for output file separation.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--clients-sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--malicious-client-ratio", type=float, default=0.2)
    parser.add_argument("--attack-scale", type=float, default=3.0)
    parser.add_argument(
        "--sign-flip-scale",
        type=float,
        default=1.0,
        help="Scale for the standalone sign-flip attack scenario.",
    )
    parser.add_argument(
        "--sign-flip-filter-scale",
        type=float,
        default=3.0,
        help="Scale for the sign-flip + update-filter scenario.",
    )
    parser.add_argument("--defense-clip-norm", type=float, default=5.0)
    parser.add_argument("--filter-std-factor", type=float, default=2.0)
    parser.add_argument("--max-filtered-ratio", type=float, default=0.5)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        print(json.dumps(SCENARIO_ORDER, ensure_ascii=False, indent=2))
        return 0

    scenario_specs = build_scenarios(args)
    selected_names = args.scenario or SCENARIO_ORDER

    if args.dry_run:
        dry_run_payload = {
            name: {
                "goal": scenario_specs[name]["goal"],
                "model": scenario_specs[name]["model"],
                "dataset": scenario_specs[name]["dataset"],
                "config": scenario_specs[name]["config"],
                "expected_experiment_mode": scenario_specs[name][
                    "expected_experiment_mode"
                ],
                "expected_scenario_tags": scenario_specs[name][
                    "expected_scenario_tags"
                ],
                "focus_fields": scenario_specs[name]["focus_fields"],
            }
            for name in selected_names
        }
        print(json.dumps(dry_run_payload, ensure_ascii=False, indent=2))
        return 0

    run_results: List[Dict[str, Any]] = []
    for scenario_name in selected_names:
        scenario_spec = scenario_specs[scenario_name]
        print("\n===== Running scenario: {} =====".format(scenario_name))
        print("Goal: {}".format(scenario_spec["goal"]))
        run_result = run_single_scenario(
            model_name=scenario_spec["model"],
            dataset_name=scenario_spec["dataset"],
            scenario_spec=scenario_spec,
        )
        summary = run_result["summary"]
        print(
            json.dumps(
                {
                    "scenario": scenario_name,
                    "experiment_mode": summary.get("experiment_mode"),
                    "scenario_tags": summary.get("scenario_tags"),
                    "active_attacks": summary.get("active_attacks"),
                    "active_defenses": summary.get("active_defenses"),
                    "active_privacy_metrics": summary.get("active_privacy_metrics"),
                    "output_paths": run_result["output_paths"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        run_results.append(run_result)

    print("\nCompleted {} scenario(s).".format(len(run_results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
