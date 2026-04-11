from __future__ import annotations

import argparse
import json
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


MMFEDRAP_SHOWCASE_SCENARIOS = [
    "baseline",
    "attack_only_model_replacement",
    "attack_and_defense_trimmed_mean",
]

DISPLAY_NOTES = {
    "baseline": "MMFedRAP 正式基线：无攻击、无主动防御，用于提供多模态强攻防展示参照。",
    "attack_only_model_replacement": "MMFedRAP 强攻击组：恶意客户端执行 minimal model-replacement-like 更新，观察多模态推荐性能退化。",
    "attack_and_defense_trimmed_mean": "MMFedRAP 强攻防组：在 model_replacement 攻击后启用 trimmed_mean，观察鲁棒聚合式防御的恢复效果。",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the formal MMFedRAP strong adversarial showcase experiment set."
    )
    parser.add_argument("--model", default="MMFedRAP")
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--type", default="MMFedRAPStrongAdvShowcase")
    parser.add_argument("--comment-prefix", default="mmfedrap_strong_adv_showcase")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--clients-sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-2)
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
        help="Do not train; rebuild summary files from existing MMFedRAP showcase outputs.",
    )
    return parser


def build_scenarios(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    common_config: Dict[str, Any] = {
        "use_gpu": bool(args.use_gpu),
        "seed": int(args.seed),
        "type": args.type,
        "hyper_parameters": [],
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "lr": float(args.lr),
        "l2_reg": float(args.l2_reg),
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
        expected_tags: List[str],
    ) -> Dict[str, Any]:
        config = deepcopy(common_config)
        config.update(config_updates)
        config["comment"] = build_comment(args.comment_prefix, name)
        return {
            "name": name,
            "goal": goal,
            "display_note": DISPLAY_NOTES[name],
            "model": args.model,
            "dataset": args.dataset,
            "config": config,
            "expected_experiment_mode": expected_mode,
            "expected_scenario_tags": expected_tags,
        }

    replacement_attack_config = {
        "enabled_attacks": ["model_replacement"],
        "enable_malicious_clients": True,
        "malicious_client_mode": "ratio",
        "malicious_client_ratio": float(args.malicious_client_ratio),
        "replacement_scale": float(args.replacement_scale),
        "replacement_rule": str(args.replacement_rule),
    }

    return {
        "baseline": scenario(
            name="baseline",
            goal="建立 MMFedRAP 多模态强攻防展示基线。",
            config_updates={
                "enabled_attacks": [],
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
                "enable_malicious_clients": False,
            },
            expected_mode="baseline",
            expected_tags=["baseline"],
        ),
        "attack_only_model_replacement": scenario(
            name="attack_only_model_replacement",
            goal="验证 MMFedRAP 上 model_replacement 强攻击的性能退化效果。",
            config_updates={
                **replacement_attack_config,
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
            },
            expected_mode="attack_only",
            expected_tags=["attack_only", "malicious_clients_configured"],
        ),
        "attack_and_defense_trimmed_mean": scenario(
            name="attack_and_defense_trimmed_mean",
            goal="验证 MMFedRAP 上 model_replacement + trimmed_mean 的强攻防主线效果。",
            config_updates={
                **replacement_attack_config,
                "enabled_defenses": ["trimmed_mean"],
                "enabled_privacy_metrics": [],
                "trim_ratio": float(args.trim_ratio),
                "min_clients_for_trim": int(args.min_clients_for_trim),
                "trim_rule": str(args.trim_rule),
            },
            expected_mode="attack_and_defense",
            expected_tags=["attack_and_defense", "malicious_clients_configured"],
        ),
    }


def find_output_paths(args: argparse.Namespace, scenario_name: str) -> Dict[str, Path]:
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


def build_summary_record(
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
        "scenario": scenario_name,
        "model": summary_payload.get("model"),
        "dataset": summary_payload.get("dataset"),
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
        "trimmed_mean_applied_rounds": trimmed_mean_applied_rounds or 0,
        "effective_trim_count": effective_trim_count or 0,
        "summary_path": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "display_note": DISPLAY_NOTES[scenario_name],
    }


def build_mmfedrap_showcase_summary(args: argparse.Namespace) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for scenario_name in MMFEDRAP_SHOWCASE_SCENARIOS:
        paths = find_output_paths(args, scenario_name)
        summary_path = paths["summary"]
        result_path = paths["result"]
        if not summary_path.exists() or not result_path.exists():
            raise FileNotFoundError(
                "Missing MMFedRAP showcase output for {}: {}, {}".format(
                    scenario_name, summary_path, result_path
                )
            )
        records.append(
            build_summary_record(
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
        "records": records,
    }


def build_markdown(summary_object: Dict[str, Any]) -> str:
    lines = [
        "# MMFedRAP strong_adv_showcase 正式展示结果",
        "",
        "- 生成时间：`{}`".format(summary_object["generated_at"]),
        "- 模型：`{}`".format(summary_object["model"]),
        "- 数据集：`{}`".format(summary_object["dataset"]),
        "- 实验类型：`{}`".format(summary_object["type"]),
        "- 注释前缀：`{}`".format(summary_object["comment_prefix"]),
        "",
        "| 场景 | 模式 | 攻击 | 防御 | Recall@20 | NDCG@20 | Loss | malicious | attacked | trimmed_mean | effective_trim_count | 展示说明 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for record in summary_object["records"]:
        lines.append(
            "| {scenario} | {mode} | {attacks} | {defenses} | {recall} | {ndcg} | {loss} | {malicious} | {attacked} | {trimmed} | {trim_count} | {note} |".format(
                scenario=record["scenario"],
                mode=record.get("experiment_mode") or "",
                attacks=", ".join(record.get("active_attacks", [])) or "-",
                defenses=", ".join(record.get("active_defenses", [])) or "-",
                recall="{:.4f}".format(record["recall20"]) if record["recall20"] is not None else "-",
                ndcg="{:.4f}".format(record["ndcg20"]) if record["ndcg20"] is not None else "-",
                loss="{:.4f}".format(record["loss"]) if record["loss"] is not None else "-",
                malicious=record.get("malicious_client_count", 0),
                attacked=record.get("attacked_client_count", 0),
                trimmed="yes" if record.get("trimmed_mean_applied") else "no",
                trim_count=record.get("effective_trim_count", 0),
                note=record["display_note"],
            )
        )

    lines.extend(["", "## 明细文件", ""])
    for record in summary_object["records"]:
        lines.extend(
            [
                "- `{}`".format(record["scenario"]),
                "  - summary: `{}`".format(record["summary_path"]),
                "  - result: `{}`".format(record["result_path"]),
            ]
        )
    return "\n".join(lines) + "\n"


def write_summary_outputs(args: argparse.Namespace) -> Dict[str, str]:
    summary = build_mmfedrap_showcase_summary(args)
    result_dir = ROOT / "outputs" / "results" / args.model / args.dataset / args.type
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "mmfedrap_strong_adv_showcase_summary.json"
    markdown_path = result_dir / "mmfedrap_strong_adv_showcase_summary.md"
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
    scenario_specs = build_scenarios(args)

    if args.dry_run:
        payload = {
            name: {
                "goal": scenario_specs[name]["goal"],
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
            for name in MMFEDRAP_SHOWCASE_SCENARIOS
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.summary_only:
        for scenario_name in MMFEDRAP_SHOWCASE_SCENARIOS:
            scenario_spec = scenario_specs[scenario_name]
            print("\n===== Running MMFedRAP strong_adv_showcase scenario: {} =====".format(scenario_name))
            print("Goal: {}".format(scenario_spec["goal"]))
            run_output = run_single_scenario(scenario_spec)
            summary = run_output["summary"]
            print(
                json.dumps(
                    {
                        "scenario": scenario_name,
                        "experiment_mode": summary.get("experiment_mode"),
                        "scenario_tags": summary.get("scenario_tags"),
                        "active_attacks": summary.get("active_attacks"),
                        "active_defenses": summary.get("active_defenses"),
                        "output_paths": run_output["output_paths"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    output_paths = write_summary_outputs(args)
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
