from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ORDER = [
    "baseline",
    "attack_only_scale",
    "attack_only_sign_flip",
    "attack_and_defense_clip",
    "attack_and_defense_filter",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize finished standard-matrix experiment outputs."
    )
    parser.add_argument("--model", default="FedRAP")
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--type", default="StandardMatrix")
    parser.add_argument(
        "--comment-prefix",
        default="",
        help="Optional filename filter such as matrix_v1.",
    )
    parser.add_argument(
        "--output-name",
        default="standard_matrix_summary",
        help="Base file name for generated summary outputs.",
    )
    return parser


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_scenario(file_name: str) -> Optional[str]:
    for scenario in sorted(SCENARIO_ORDER, key=len, reverse=True):
        if scenario in file_name:
            return scenario
    return None


def select_summary_files(result_dir: Path, comment_prefix: str) -> Dict[str, Path]:
    candidates: Dict[str, List[Path]] = {}
    for path in result_dir.glob("*.experiment_summary.json"):
        file_name = path.name
        if comment_prefix and comment_prefix not in file_name:
            continue
        scenario = detect_scenario(file_name)
        if not scenario:
            continue
        candidates.setdefault(scenario, []).append(path)

    selected: Dict[str, Path] = {}
    for scenario, paths in candidates.items():
        selected[scenario] = max(paths, key=lambda candidate: candidate.stat().st_mtime)
    return selected


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


def max_round_value(round_summaries: List[Dict[str, Any]], key: str) -> Optional[int]:
    values = [
        safe_int(round_summary.get(key))
        for round_summary in round_summaries
        if round_summary.get(key) is not None
    ]
    filtered_values = [value for value in values if value is not None]
    return max(filtered_values) if filtered_values else None


def extract_filtered_client_count(result_payload: Dict[str, Any]) -> Optional[int]:
    metadata = result_payload.get("metadata", {})
    defense_summaries = metadata.get("defense_summaries", {})
    update_filter_summary = defense_summaries.get("update_filter")
    if isinstance(update_filter_summary, dict):
        max_filtered = safe_int(update_filter_summary.get("max_filtered_client_count"))
        if max_filtered is not None:
            return max_filtered

    round_metrics = result_payload.get("round_metrics", [])
    values: List[int] = []
    for round_metric in round_metrics:
        defense_metrics = (
            round_metric.get("extra", {}).get("defense_metrics", {})
            if isinstance(round_metric, dict)
            else {}
        )
        update_filter_metrics = defense_metrics.get("update_filter", {})
        if not isinstance(update_filter_metrics, dict):
            continue
        value = safe_int(update_filter_metrics.get("filtered_client_count"))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def build_record(summary_path: Path) -> Dict[str, Any]:
    result_path = summary_path.with_name(
        summary_path.name.replace(".experiment_summary.json", ".experiment_result.json")
    )
    summary_payload = load_json(summary_path)
    result_payload = load_json(result_path) if result_path.exists() else {}

    round_summaries = summary_payload.get("round_summaries", [])
    final_eval = summary_payload.get("final_eval", {}) or {}
    malicious_summary = summary_payload.get("malicious_client_summary", {}) or {}

    scenario = detect_scenario(summary_path.name) or "unknown"
    attacked_client_count = max_round_value(round_summaries, "attacked_client_count")
    clipped_client_count = max_round_value(round_summaries, "clipped_client_count")
    malicious_client_count = safe_int(
        malicious_summary.get("max_round_malicious_client_count")
    )
    if malicious_client_count is None:
        malicious_client_count = max_round_value(round_summaries, "malicious_client_count")

    record = {
        "scenario": scenario,
        "experiment_id": summary_payload.get("experiment_id"),
        "experiment_mode": summary_payload.get("experiment_mode"),
        "scenario_tags": summary_payload.get("scenario_tags", []),
        "active_attacks": summary_payload.get("active_attacks", []),
        "active_defenses": summary_payload.get("active_defenses", []),
        "active_privacy_metrics": summary_payload.get("active_privacy_metrics", []),
        "recall20": safe_float(final_eval.get("recall20")),
        "ndcg20": safe_float(final_eval.get("ndcg20")),
        "loss": safe_float(final_eval.get("loss")),
        "attacked_client_count": attacked_client_count or 0,
        "clipped_client_count": clipped_client_count or 0,
        "filtered_client_count": extract_filtered_client_count(result_payload) or 0,
        "malicious_client_count": malicious_client_count or 0,
        "summary_path": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
    }
    return record


def build_markdown(summary_object: Dict[str, Any]) -> str:
    lines = [
        "# 标准实验矩阵结果汇总",
        "",
        "- 生成时间：`{}`".format(summary_object["generated_at"]),
        "- 模型：`{}`".format(summary_object["model"]),
        "- 数据集：`{}`".format(summary_object["dataset"]),
        "- 实验类型：`{}`".format(summary_object["type"]),
        "- 注释筛选：`{}`".format(summary_object["comment_prefix"] or "(未指定)"),
        "",
        "| 场景 | 模式 | 攻击 | 防御 | 隐私观测 | Recall@20 | NDCG@20 | Loss | malicious | attacked | clipped | filtered |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for record in summary_object["records"]:
        lines.append(
            "| {scenario} | {mode} | {attacks} | {defenses} | {privacy} | {recall} | {ndcg} | {loss} | {malicious} | {attacked} | {clipped} | {filtered} |".format(
                scenario=record["scenario"],
                mode=record.get("experiment_mode") or "",
                attacks=", ".join(record.get("active_attacks", [])) or "-",
                defenses=", ".join(record.get("active_defenses", [])) or "-",
                privacy=", ".join(record.get("active_privacy_metrics", [])) or "-",
                recall="{:.4f}".format(record["recall20"]) if record["recall20"] is not None else "-",
                ndcg="{:.4f}".format(record["ndcg20"]) if record["ndcg20"] is not None else "-",
                loss="{:.4f}".format(record["loss"]) if record["loss"] is not None else "-",
                malicious=record.get("malicious_client_count", 0),
                attacked=record.get("attacked_client_count", 0),
                clipped=record.get("clipped_client_count", 0),
                filtered=record.get("filtered_client_count", 0),
            )
        )

    lines.extend(
        [
            "",
            "## 明细文件路径",
            "",
        ]
    )

    for record in summary_object["records"]:
        lines.extend(
            [
                "- `{}`".format(record["scenario"]),
                "  - summary: `{}`".format(record["summary_path"]),
                "  - result: `{}`".format(record["result_path"]),
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    result_dir = ROOT / "outputs" / "results" / args.model / args.dataset / args.type

    if not result_dir.exists():
        raise SystemExit("Result directory not found: {}".format(result_dir))

    selected_summary_files = select_summary_files(result_dir, args.comment_prefix)
    records = [
        build_record(selected_summary_files[scenario])
        for scenario in SCENARIO_ORDER
        if scenario in selected_summary_files
    ]

    if not records:
        raise SystemExit(
            "No standard-matrix summary files found in {} with comment prefix {!r}".format(
                result_dir, args.comment_prefix
            )
        )

    output_suffix = ".{}".format(args.comment_prefix) if args.comment_prefix else ""
    json_output_path = result_dir / "{}{}.json".format(args.output_name, output_suffix)
    markdown_output_path = result_dir / "{}{}.md".format(args.output_name, output_suffix)

    summary_object = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "dataset": args.dataset,
        "type": args.type,
        "comment_prefix": args.comment_prefix,
        "records": records,
    }

    json_output_path.write_text(
        json.dumps(summary_object, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_output_path.write_text(
        build_markdown(summary_object),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "records": len(records),
                "json_output": str(json_output_path.relative_to(ROOT)).replace("\\", "/"),
                "markdown_output": str(markdown_output_path.relative_to(ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
