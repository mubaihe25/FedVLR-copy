from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.configurator import Config
from utils.logger import init_logger
from utils.quick_start import _prepare_data
from utils.utils import get_model, get_trainer, init_seed


SHOWCASE_SCENARIOS = [
    "baseline",
    "attack_only_sign_flip",
    "attack_and_defense_clip",
]

DISPLAY_NOTES = {
    "baseline": "正常基线：无攻击、无主动防御，用于提供推荐性能参照。",
    "attack_only_sign_flip": "攻击场景：恶意客户端执行符号翻转，观察推荐性能退化。",
    "attack_and_defense_clip": "攻防场景：攻击后启用范数裁剪，对异常更新进行约束。",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FedVLR showcase_v1 three-scenario experiment set."
    )
    parser.add_argument("--model", default="FedRAP")
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--type", default="ShowcaseV1")
    parser.add_argument("--comment-prefix", default="showcase_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--clients-sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--malicious-client-ratio", type=float, default=0.2)
    parser.add_argument("--sign-flip-scale", type=float, default=1.0)
    parser.add_argument("--attack-scale", type=float, default=3.0)
    parser.add_argument("--defense-clip-norm", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not train; rebuild the comparison files from existing showcase outputs.",
    )
    return parser


def build_comment(prefix: str, scenario_name: str) -> str:
    prefix = prefix.strip()
    return "{}.{}".format(prefix, scenario_name) if prefix else scenario_name


def build_showcase_scenarios(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
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
        config_updates: Dict[str, Any],
        expected_mode: str,
        expected_tags: List[str],
    ) -> Dict[str, Any]:
        config = deepcopy(common_config)
        config.update(config_updates)
        config["comment"] = build_comment(args.comment_prefix, name)
        return {
            "name": name,
            "display_note": DISPLAY_NOTES[name],
            "model": args.model,
            "dataset": args.dataset,
            "config": config,
            "expected_experiment_mode": expected_mode,
            "expected_scenario_tags": expected_tags,
        }

    return {
        "baseline": scenario(
            name="baseline",
            config_updates={
                "enabled_attacks": [],
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": False,
            },
            expected_mode="baseline",
            expected_tags=["baseline"],
        ),
        "attack_only_sign_flip": scenario(
            name="attack_only_sign_flip",
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
            expected_tags=["attack_only", "malicious_clients_configured"],
        ),
        "attack_and_defense_clip": scenario(
            name="attack_and_defense_clip",
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
            expected_tags=["attack_and_defense", "malicious_clients_configured"],
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


def run_single_scenario(scenario_spec: Dict[str, Any]) -> Dict[str, Any]:
    config = Config(
        model=scenario_spec["model"],
        dataset=scenario_spec["dataset"],
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
        "display_note": scenario_spec["display_note"],
        "summary": trainer.experiment_summary_dict,
        "result": trainer.experiment_result_dict,
        "output_paths": derive_output_paths(config),
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def max_round_value(round_summaries: List[Dict[str, Any]], key: str) -> int:
    values = [
        safe_int(round_summary.get(key))
        for round_summary in round_summaries
        if round_summary.get(key) is not None
    ]
    filtered_values = [value for value in values if value is not None]
    return max(filtered_values) if filtered_values else 0


def find_result_paths(
    args: argparse.Namespace,
    scenario_name: str,
) -> Dict[str, Path]:
    result_dir = ROOT / "outputs" / "results" / args.model / args.dataset / args.type
    base_name = "[{}]-[{}]-[{}.{}]".format(
        args.model,
        args.dataset,
        args.type,
        build_comment(args.comment_prefix, scenario_name),
    )
    return {
        "summary": result_dir / "{}.experiment_summary.json".format(base_name),
        "result": result_dir / "{}.experiment_result.json".format(base_name),
        "csv": result_dir / "{}.csv".format(base_name),
    }


def extract_filtered_client_count(result_payload: Dict[str, Any]) -> int:
    defense_summaries = result_payload.get("metadata", {}).get("defense_summaries", {})
    update_filter_summary = defense_summaries.get("update_filter")
    if isinstance(update_filter_summary, dict):
        count = safe_int(update_filter_summary.get("max_filtered_client_count"))
        if count is not None:
            return count
    return 0


def build_comparison_record(
    scenario_name: str,
    summary_payload: Dict[str, Any],
    result_payload: Dict[str, Any],
    summary_path: Path,
    result_path: Path,
) -> Dict[str, Any]:
    final_eval = summary_payload.get("final_eval", {}) or {}
    round_summaries = summary_payload.get("round_summaries", []) or []
    malicious_summary = summary_payload.get("malicious_client_summary", {}) or {}
    malicious_count = safe_int(
        malicious_summary.get("max_round_malicious_client_count")
    )
    if malicious_count is None:
        malicious_count = max_round_value(round_summaries, "malicious_client_count")

    return {
        "scenario": scenario_name,
        "experiment_mode": summary_payload.get("experiment_mode"),
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
        "clipped_client_count": max_round_value(
            round_summaries, "clipped_client_count"
        ),
        "filtered_client_count": extract_filtered_client_count(result_payload),
        "summary_path": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "display_note": DISPLAY_NOTES[scenario_name],
    }


def build_markdown(comparison: Dict[str, Any]) -> str:
    lines = [
        "# showcase_v1 对比摘要",
        "",
        "- 生成时间：`{}`".format(comparison["generated_at"]),
        "- 模型：`{}`".format(comparison["model"]),
        "- 数据集：`{}`".format(comparison["dataset"]),
        "- 输出目录：`{}`".format(comparison["output_dir"]),
        "",
        "| 场景 | 模式 | 攻击 | 防御 | Recall@20 | NDCG@20 | Loss | malicious | attacked | clipped | filtered | 展示说明 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for record in comparison["records"]:
        lines.append(
            "| {scenario} | {mode} | {attacks} | {defenses} | {recall} | {ndcg} | {loss} | {malicious} | {attacked} | {clipped} | {filtered} | {note} |".format(
                scenario=record["scenario"],
                mode=record.get("experiment_mode") or "",
                attacks=", ".join(record.get("active_attacks", [])) or "-",
                defenses=", ".join(record.get("active_defenses", [])) or "-",
                recall="{:.4f}".format(record["recall20"]) if record["recall20"] is not None else "-",
                ndcg="{:.4f}".format(record["ndcg20"]) if record["ndcg20"] is not None else "-",
                loss="{:.4f}".format(record["loss"]) if record["loss"] is not None else "-",
                malicious=record.get("malicious_client_count", 0),
                attacked=record.get("attacked_client_count", 0),
                clipped=record.get("clipped_client_count", 0),
                filtered=record.get("filtered_client_count", 0),
                note=record["display_note"],
            )
        )

    lines.extend(["", "## 明细文件", ""])
    for record in comparison["records"]:
        lines.extend(
            [
                "- `{}`".format(record["scenario"]),
                "  - summary: `{}`".format(record["summary_path"]),
                "  - result: `{}`".format(record["result_path"]),
            ]
        )

    return "\n".join(lines) + "\n"


def build_showcase_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    result_dir = ROOT / "outputs" / "results" / args.model / args.dataset / args.type
    records: List[Dict[str, Any]] = []
    for scenario_name in SHOWCASE_SCENARIOS:
        paths = find_result_paths(args, scenario_name)
        summary_path = paths["summary"]
        result_path = paths["result"]
        if not summary_path.exists() or not result_path.exists():
            raise FileNotFoundError(
                "Missing showcase output for {}: {}, {}".format(
                    scenario_name, summary_path, result_path
                )
            )
        records.append(
            build_comparison_record(
                scenario_name=scenario_name,
                summary_payload=load_json(summary_path),
                result_payload=load_json(result_path),
                summary_path=summary_path,
                result_path=result_path,
            )
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "dataset": args.dataset,
        "type": args.type,
        "comment_prefix": args.comment_prefix,
        "output_dir": str(result_dir.relative_to(ROOT)).replace("\\", "/"),
        "records": records,
    }


def write_comparison_outputs(args: argparse.Namespace) -> Dict[str, str]:
    comparison = build_showcase_comparison(args)
    result_dir = ROOT / "outputs" / "results" / args.model / args.dataset / args.type
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "showcase_v1_comparison.json"
    markdown_path = result_dir / "showcase_v1_comparison.md"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(comparison), encoding="utf-8")
    return {
        "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
        "markdown": str(markdown_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> int:
    args = build_parser().parse_args()
    scenario_specs = build_showcase_scenarios(args)

    if args.dry_run:
        payload = {
            name: {
                "model": scenario_specs[name]["model"],
                "dataset": scenario_specs[name]["dataset"],
                "display_note": scenario_specs[name]["display_note"],
                "config": scenario_specs[name]["config"],
                "expected_experiment_mode": scenario_specs[name][
                    "expected_experiment_mode"
                ],
                "expected_scenario_tags": scenario_specs[name][
                    "expected_scenario_tags"
                ],
            }
            for name in SHOWCASE_SCENARIOS
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.summary_only:
        for scenario_name in SHOWCASE_SCENARIOS:
            scenario_spec = scenario_specs[scenario_name]
            print("\n===== Running showcase_v1 scenario: {} =====".format(scenario_name))
            run_output = run_single_scenario(scenario_spec)
            summary = run_output["summary"]
            print(
                json.dumps(
                    {
                        "scenario": scenario_name,
                        "experiment_mode": summary.get("experiment_mode"),
                        "active_attacks": summary.get("active_attacks"),
                        "active_defenses": summary.get("active_defenses"),
                        "output_paths": run_output["output_paths"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    output_paths = write_comparison_outputs(args)
    print(
        json.dumps(
            {
                "comparison_json": output_paths["json"],
                "comparison_markdown": output_paths["markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
