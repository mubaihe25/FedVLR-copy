from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.run_strong_adv_showcase import (
    build_comment,
    extract_trimmed_mean_summary,
    load_json,
    max_round_value,
    max_trimmed_mean_round_value,
    run_single_scenario,
    safe_float,
    safe_int,
)


MODEL_ORDER = [
    "FedAvg",
    "FedNCF",
    "FedRAP",
    "FedVBPR",
    "PFedRec",
    "MMFedAvg",
    "MMFedNCF",
    "MMFedRAP",
    "MMPFedRec",
    "MMGCN",
]

SCENARIO_ORDER = [
    "baseline",
    "attack_only_model_replacement",
    "attack_and_defense_trimmed_mean",
]

MODEL_RISK_POINTS: Dict[str, List[str]] = {
    "FedAvg": [
        "单模态 FederatedTrainer 路径；默认 lr/l2_reg 为列表，需要脚本显式标量化。",
        "上传更新通常是模型参数字典，现有递归 hook 可处理。",
    ],
    "FedNCF": [
        "单模态 FederatedTrainer 路径；默认 lr/l2_reg 为列表，需要脚本显式标量化。",
        "上传更新通常是嵌套参数字典，需确认 model_replacement/trimmed_mean 不破坏聚合输入。",
    ],
    "FedRAP": [
        "已完成正式展示主线；本批次用于横向基准复核。",
        "上传更新主要包含 item_commonality.weight，结构最接近已验证路径。",
    ],
    "FedVBPR": [
        "标记为多模态联邦模型，依赖 KU 的 image/text 特征文件。",
        "默认 lr/reg_weight 为列表，需要脚本显式标量化。",
    ],
    "PFedRec": [
        "用户称 pFedRec，工程实际类名为 PFedRec。",
        "默认 lr/l2_reg 为列表，需要脚本显式标量化。",
    ],
    "MMFedAvg": [
        "多模态 FederatedTrainer 路径，依赖 fusion 层与 item_commonality 上传更新。",
        "默认 lr/l2_reg 为列表，需要脚本显式标量化。",
    ],
    "MMFedNCF": [
        "多模态 FederatedTrainer 路径，依赖 fusion 层与 item_commonality 上传更新。",
        "默认 lr/l2_reg 为列表，需要脚本显式标量化。",
    ],
    "MMFedRAP": [
        "已完成 MMFedRAP 正式展示验证；本批次用于横向复核。",
        "上传更新包含 fusion.* 梯度和 item_commonality.weight。",
    ],
    "MMPFedRec": [
        "多模态个性化联邦模型，依赖 fusion 层与个性化参数存储。",
        "默认 lr/l2_reg 为列表，需要脚本显式标量化。",
    ],
    "MMGCN": [
        "工程中存在 MMGCN 模型类，但没有 MMGCNTrainer。",
        "当前不在 FederatedTrainer hook 链路中，不适合强行验证 model_replacement + trimmed_mean。",
    ],
}

MODEL_CONFIG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "FedAvg": {"lr": 0.001, "l2_reg": 1e-5},
    "FedNCF": {"lr": 0.001, "l2_reg": 1e-5},
    "FedRAP": {"lr": 0.1, "l2_reg": 1e-7, "alpha": 0.1, "beta": 0.1},
    "FedVBPR": {"lr": 0.001, "reg_weight": 1e-4, "l2_reg": 1e-5},
    "PFedRec": {"lr": 0.001, "l2_reg": 1e-5},
    "MMFedAvg": {"lr": 0.001, "l2_reg": 1e-5},
    "MMFedNCF": {"lr": 0.001, "l2_reg": 1e-5},
    "MMFedRAP": {"lr": 0.001, "l2_reg": 1e-4, "alpha": 1e-4, "beta": 1e-2},
    "MMPFedRec": {"lr": 0.001, "l2_reg": 1e-5},
    "MMGCN": {"learning_rate": 0.001, "reg_weight": 1e-4},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate strong adversarial hook compatibility across model families."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_ORDER,
        help="Run only selected models. Repeat this flag to run multiple models.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIO_ORDER,
        help="Run only selected scenarios. Repeat this flag to run multiple scenarios.",
    )
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--type", default="ModelFamilyStrongAdvCompat")
    parser.add_argument("--comment-prefix", default="model_family_strong_adv_compat")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--clients-sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--malicious-client-ratio", type=float, default=0.2)
    parser.add_argument("--replacement-scale", type=float, default=5.0)
    parser.add_argument("--replacement-rule", default="aligned_mean")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--min-clients-for-trim", type=int, default=5)
    parser.add_argument("--trim-rule", default="coordinate_trimmed_mean")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not train; rebuild summary from existing per-model outputs.",
    )
    return parser


def model_module_exists(model_name: str) -> bool:
    return importlib.util.find_spec("models.{}".format(model_name.lower())) is not None


def federated_trainer_status(model_name: str) -> Tuple[bool, Optional[str]]:
    if not model_module_exists(model_name):
        return False, "model_module_missing"
    try:
        module = importlib.import_module("models.{}".format(model_name.lower()))
    except ModuleNotFoundError as exc:
        return False, "trainer_import_failed:missing_dependency:{}".format(exc.name)
    except Exception as exc:  # noqa: BLE001 - compatibility precheck must not stop the batch.
        return False, "trainer_import_failed:{}:{}".format(type(exc).__name__, exc)
    if not hasattr(module, "{}Trainer".format(model_name)):
        return False, "federated_trainer_missing"
    return True, None


def federated_trainer_exists(model_name: str) -> bool:
    return federated_trainer_status(model_name)[0]


def precheck_model(model_name: str) -> Tuple[bool, Optional[str]]:
    if not model_module_exists(model_name):
        return False, "model_module_missing"
    return federated_trainer_status(model_name)


def build_common_config(args: argparse.Namespace, model_name: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "use_gpu": bool(args.use_gpu),
        "seed": int(args.seed),
        "type": args.type,
        "hyper_parameters": [],
        "epochs": int(args.epochs),
        "local_epochs": int(args.local_epochs),
        "clients_sample_ratio": float(args.clients_sample_ratio),
        "eval_step": 1,
        "collect_round_metrics": True,
        "enable_experiment_hooks": False,
    }
    config.update(MODEL_CONFIG_OVERRIDES.get(model_name, {}))
    return config


def build_scenario_spec(
    args: argparse.Namespace,
    model_name: str,
    scenario_name: str,
) -> Dict[str, Any]:
    common_config = build_common_config(args, model_name)

    scenario_updates: Dict[str, Any]
    expected_mode: str
    expected_tags: List[str]
    if scenario_name == "baseline":
        scenario_updates = {
            "enabled_attacks": [],
            "enabled_defenses": [],
            "enabled_privacy_metrics": [],
            "enable_malicious_clients": False,
        }
        expected_mode = "baseline"
        expected_tags = ["baseline"]
    elif scenario_name == "attack_only_model_replacement":
        scenario_updates = {
            "enabled_attacks": ["model_replacement"],
            "enabled_defenses": [],
            "enabled_privacy_metrics": [],
            "enable_malicious_clients": True,
            "malicious_client_mode": "ratio",
            "malicious_client_ratio": float(args.malicious_client_ratio),
            "replacement_scale": float(args.replacement_scale),
            "replacement_rule": str(args.replacement_rule),
        }
        expected_mode = "attack_only"
        expected_tags = ["attack_only", "malicious_clients_configured"]
    elif scenario_name == "attack_and_defense_trimmed_mean":
        scenario_updates = {
            "enabled_attacks": ["model_replacement"],
            "enabled_defenses": ["trimmed_mean"],
            "enabled_privacy_metrics": [],
            "enable_malicious_clients": True,
            "malicious_client_mode": "ratio",
            "malicious_client_ratio": float(args.malicious_client_ratio),
            "replacement_scale": float(args.replacement_scale),
            "replacement_rule": str(args.replacement_rule),
            "trim_ratio": float(args.trim_ratio),
            "min_clients_for_trim": int(args.min_clients_for_trim),
            "trim_rule": str(args.trim_rule),
        }
        expected_mode = "attack_and_defense"
        expected_tags = ["attack_and_defense", "malicious_clients_configured"]
    else:
        raise ValueError("Unsupported scenario: {}".format(scenario_name))

    config = deepcopy(common_config)
    config.update(scenario_updates)
    config["comment"] = build_comment(
        args.comment_prefix,
        "{}.{}".format(model_name, scenario_name),
    )

    return {
        "name": scenario_name,
        "goal": "{} / {} compatibility validation".format(model_name, scenario_name),
        "display_note": "验证 {} 在 {} 场景下的强攻防 hook、结果记录与输出兼容性。".format(
            model_name, scenario_name
        ),
        "model": model_name,
        "dataset": args.dataset,
        "config": config,
        "expected_experiment_mode": expected_mode,
        "expected_scenario_tags": expected_tags,
        "focus_fields": [
            "experiment_mode",
            "active_attacks",
            "active_defenses",
            "round_metrics[*].extra.attack_metrics",
            "round_metrics[*].extra.defense_metrics",
        ],
    }


def find_output_paths(
    args: argparse.Namespace,
    model_name: str,
    scenario_name: str,
) -> Dict[str, Path]:
    result_dir = ROOT / "outputs" / "results" / model_name / args.dataset / args.type
    comment = build_comment(args.comment_prefix, "{}.{}".format(model_name, scenario_name))
    base_name = "[{}]-[{}]-[{}.{}]".format(
        model_name,
        args.dataset,
        args.type,
        comment,
    )
    return {
        "summary": result_dir / "{}.experiment_summary.json".format(base_name),
        "result": result_dir / "{}.experiment_result.json".format(base_name),
        "csv": result_dir / "{}.csv".format(base_name),
    }


def build_success_record(
    model_name: str,
    scenario_name: str,
    summary_payload: Dict[str, Any],
    result_payload: Dict[str, Any],
    summary_path: Path,
    result_path: Path,
) -> Dict[str, Any]:
    final_eval = summary_payload.get("final_eval", {}) or {}
    round_summaries = summary_payload.get("round_summaries", []) or []
    malicious_summary = summary_payload.get("malicious_client_summary", {}) or {}
    trimmed_mean_summary = extract_trimmed_mean_summary(result_payload)

    malicious_count = safe_int(
        malicious_summary.get("max_round_malicious_client_count")
    )
    if malicious_count is None:
        malicious_count = max_round_value(round_summaries, "malicious_client_count")

    trimmed_mean_applied_rounds = safe_int(
        trimmed_mean_summary.get("rounds_with_trimmed_mean")
    )
    effective_trim_count = max_trimmed_mean_round_value(
        result_payload, "effective_trim_count"
    )

    return {
        "model": model_name,
        "scenario": scenario_name,
        "dataset": summary_payload.get("dataset"),
        "success": True,
        "failed": False,
        "failure_stage": None,
        "failure_reason": None,
        "experiment_mode": summary_payload.get("experiment_mode"),
        "scenario_tags": summary_payload.get("scenario_tags", []),
        "active_attacks": summary_payload.get("active_attacks", []),
        "active_defenses": summary_payload.get("active_defenses", []),
        "active_privacy_metrics": summary_payload.get("active_privacy_metrics", []),
        "recall20": safe_float(final_eval.get("recall20")),
        "ndcg20": safe_float(final_eval.get("ndcg20")),
        "loss": safe_float(final_eval.get("loss")),
        "malicious_client_count": malicious_count or 0,
        "attacked_client_count": max_round_value(
            round_summaries, "attacked_client_count"
        ),
        "trimmed_mean_applied": bool(trimmed_mean_applied_rounds),
        "effective_trim_count": effective_trim_count or 0,
        "summary_path": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "compatibility_note": "{} / {} 已跑通，hook 指标和结果文件已生成。".format(
            model_name, scenario_name
        ),
    }


def build_failure_record(
    model_name: str,
    scenario_name: str,
    dataset: str,
    failure_stage: str,
    failure_reason: str,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "scenario": scenario_name,
        "dataset": dataset,
        "success": False,
        "failed": True,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "experiment_mode": None,
        "scenario_tags": [],
        "active_attacks": [],
        "active_defenses": [],
        "active_privacy_metrics": [],
        "recall20": None,
        "ndcg20": None,
        "loss": None,
        "malicious_client_count": 0,
        "attacked_client_count": 0,
        "trimmed_mean_applied": False,
        "effective_trim_count": 0,
        "summary_path": None,
        "result_path": None,
        "compatibility_note": "{} / {} 未跑通：{}。".format(
            model_name, scenario_name, failure_reason
        ),
    }


def load_success_from_disk(
    args: argparse.Namespace,
    model_name: str,
    scenario_name: str,
) -> Dict[str, Any]:
    paths = find_output_paths(args, model_name, scenario_name)
    summary_path = paths["summary"]
    result_path = paths["result"]
    if not summary_path.exists() or not result_path.exists():
        return build_failure_record(
            model_name,
            scenario_name,
            args.dataset,
            "summary_loading",
            "missing experiment_result or experiment_summary",
        )
    return build_success_record(
        model_name,
        scenario_name,
        load_json(summary_path),
        load_json(result_path),
        summary_path,
        result_path,
    )


def run_model_scenario(
    args: argparse.Namespace,
    model_name: str,
    scenario_name: str,
) -> Dict[str, Any]:
    ok, precheck_reason = precheck_model(model_name)
    if not ok:
        return build_failure_record(
            model_name,
            scenario_name,
            args.dataset,
            "precheck",
            precheck_reason or "precheck_failed",
        )

    scenario_spec = build_scenario_spec(args, model_name, scenario_name)
    try:
        run_single_scenario(scenario_spec)
    except Exception as exc:
        return build_failure_record(
            model_name,
            scenario_name,
            args.dataset,
            "training_or_export",
            "{}: {}".format(type(exc).__name__, exc),
        )

    return load_success_from_disk(args, model_name, scenario_name)


def build_markdown(summary_object: Dict[str, Any]) -> str:
    lines = [
        "# model_family_strong_adv_compat 批量兼容验证摘要",
        "",
        "- 生成时间：`{}`".format(summary_object["generated_at"]),
        "- 数据集：`{}`".format(summary_object["dataset"]),
        "- 实验类型：`{}`".format(summary_object["type"]),
        "- 场景：`{}`".format(", ".join(summary_object["scenarios"])),
        "",
        "## 模型风险点",
        "",
    ]
    for model_name in summary_object["models"]:
        lines.append("- `{}`：{}".format(model_name, "；".join(MODEL_RISK_POINTS[model_name])))

    lines.extend(
        [
            "",
            "## 运行结果",
            "",
            "| 模型 | 场景 | 状态 | 失败阶段 | Recall@20 | NDCG@20 | Loss | malicious | attacked | trimmed_mean | trim_count | 说明 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for record in summary_object["records"]:
        lines.append(
            "| {model} | {scenario} | {status} | {failure_stage} | {recall} | {ndcg} | {loss} | {malicious} | {attacked} | {trimmed} | {trim_count} | {note} |".format(
                model=record["model"],
                scenario=record["scenario"],
                status="success" if record["success"] else "failed",
                failure_stage=record.get("failure_stage") or "-",
                recall="{:.4f}".format(record["recall20"]) if record["recall20"] is not None else "-",
                ndcg="{:.4f}".format(record["ndcg20"]) if record["ndcg20"] is not None else "-",
                loss="{:.4f}".format(record["loss"]) if record["loss"] is not None else "-",
                malicious=record.get("malicious_client_count", 0),
                attacked=record.get("attacked_client_count", 0),
                trimmed="yes" if record.get("trimmed_mean_applied") else "no",
                trim_count=record.get("effective_trim_count", 0),
                note=record.get("compatibility_note") or record.get("failure_reason") or "",
            )
        )

    return "\n".join(lines) + "\n"


def write_summary_outputs(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Dict[str, str]:
    selected_models = args.model or MODEL_ORDER
    selected_scenarios = args.scenario or SCENARIO_ORDER
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "type": args.type,
        "comment_prefix": args.comment_prefix,
        "models": selected_models,
        "scenarios": selected_scenarios,
        "model_risk_points": MODEL_RISK_POINTS,
        "records": records,
    }
    result_dir = ROOT / "outputs" / "results" / "model_family_strong_adv_compat"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "model_family_strong_adv_compat_summary.json"
    markdown_path = result_dir / "model_family_strong_adv_compat_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")
    return {
        "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
        "markdown": str(markdown_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> int:
    args = build_parser().parse_args()
    selected_models = args.model or MODEL_ORDER
    selected_scenarios = args.scenario or SCENARIO_ORDER

    if args.dry_run:
        payload = {}
        for model_name in selected_models:
            precheck_ok, precheck_reason = precheck_model(model_name)
            payload[model_name] = {
                "risk_points": MODEL_RISK_POINTS[model_name],
                "precheck": [precheck_ok, precheck_reason],
                "scenarios": {
                    scenario_name: build_scenario_spec(args, model_name, scenario_name)
                    if precheck_ok
                    else {"skipped": precheck_reason}
                    for scenario_name in selected_scenarios
                },
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    records: List[Dict[str, Any]] = []
    for model_name in selected_models:
        for scenario_name in selected_scenarios:
            print("\n===== Validating {} / {} =====".format(model_name, scenario_name))
            if args.summary_only:
                record = load_success_from_disk(args, model_name, scenario_name)
            else:
                record = run_model_scenario(args, model_name, scenario_name)
            print(json.dumps(record, ensure_ascii=False, indent=2))
            records.append(record)

    output_paths = write_summary_outputs(args, records)
    print(
        json.dumps(
            {
                "summary_json": output_paths["json"],
                "summary_markdown": output_paths["markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
