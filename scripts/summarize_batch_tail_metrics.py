from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "batch_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize batch CSV tail-mean Recall@50/NDCG@50 metrics."
    )
    parser.add_argument(
        "--batch-summary",
        action="append",
        required=True,
        help="Path to outputs/batch_runs/<batch>_summary.json. Can be repeated.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output prefix under outputs/batch_runs. Defaults to the first batch name.",
    )
    return parser


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def tail_window_size(total_rounds: int) -> int:
    if total_rounds <= 0:
        return 0
    if total_rounds >= 100:
        return min(20, total_rounds)
    return max(1, math.ceil(total_rounds * 0.2))


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def metric_source(round_rows: List[Dict[str, Any]]) -> str:
    for row in round_rows:
        if parse_float(row.get("test_recall50")) is not None or parse_float(row.get("test_ndcg50")) is not None:
            return "test"
    return "valid"


def compute_tail_from_rounds(round_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    window = tail_window_size(len(round_rows))
    tail_rows = round_rows[-window:] if window else []
    previous_rows = round_rows[-(2 * window):-window] if window and len(round_rows) > window else []
    source = metric_source(tail_rows)
    prefix = "test" if source == "test" else "valid"

    tail_recall = mean(parse_float(row.get(f"{prefix}_recall50")) for row in tail_rows)
    tail_ndcg = mean(parse_float(row.get(f"{prefix}_ndcg50")) for row in tail_rows)
    previous_recall = mean(parse_float(row.get(f"{prefix}_recall50")) for row in previous_rows)
    previous_ndcg = mean(parse_float(row.get(f"{prefix}_ndcg50")) for row in previous_rows)

    return {
        "tail_source": source,
        "tail_window_size": window,
        "tail_recall50": tail_recall,
        "tail_ndcg50": tail_ndcg,
        "tail_start_round": parse_int(tail_rows[0].get("round_index")) if tail_rows else None,
        "tail_end_round": parse_int(tail_rows[-1].get("round_index")) if tail_rows else None,
        "tail_valid_recall50": mean(parse_float(row.get("valid_recall50")) for row in tail_rows),
        "tail_valid_ndcg50": mean(parse_float(row.get("valid_ndcg50")) for row in tail_rows),
        "tail_test_recall50": mean(parse_float(row.get("test_recall50")) for row in tail_rows),
        "tail_test_ndcg50": mean(parse_float(row.get("test_ndcg50")) for row in tail_rows),
        "previous_tail_recall50": previous_recall,
        "previous_tail_ndcg50": previous_ndcg,
        "collapse_ratio": (
            (previous_recall - tail_recall) / previous_recall
            if previous_recall not in (None, 0) and tail_recall is not None
            else None
        ),
    }


def summarize_csv(path: Path) -> Dict[str, Any]:
    rows = load_csv_rows(path)
    round_rows = [row for row in rows if row.get("row_type") == "round"]
    tail_rows = [row for row in rows if row.get("row_type") == "tail_mean_summary"]
    summary = compute_tail_from_rounds(round_rows)

    if tail_rows:
        tail = tail_rows[-1]
        summary.update(
            {
                "tail_source": tail.get("tail_source") or summary.get("tail_source"),
                "tail_window_size": parse_int(tail.get("tail_window_size")) or summary.get("tail_window_size"),
                "tail_recall50": parse_float(tail.get("tail_recall50")),
                "tail_ndcg50": parse_float(tail.get("tail_ndcg50")),
                "tail_start_round": parse_int(tail.get("tail_start_round")),
                "tail_end_round": parse_int(tail.get("tail_end_round")),
                "tail_valid_recall50": parse_float(tail.get("tail_valid_recall50")),
                "tail_valid_ndcg50": parse_float(tail.get("tail_valid_ndcg50")),
                "tail_test_recall50": parse_float(tail.get("tail_test_recall50")),
                "tail_test_ndcg50": parse_float(tail.get("tail_test_ndcg50")),
            }
        )

    summary["round_row_count"] = len(round_rows)
    summary["last_row_type"] = rows[-1].get("row_type") if rows else None
    summary["has_best_summary"] = any(row.get("row_type") == "best_summary" for row in rows)
    summary["has_tail_mean_summary"] = bool(tail_rows)
    return summary


def config_value(config: Dict[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def summarize_batch(summary_path: Path) -> Dict[str, Any]:
    batch = load_json(summary_path)
    rows = []
    for item in batch.get("results", []):
        run = item.get("run") or {}
        csv_path = resolve_path(run.get("csv_path"))
        row = {
            "batch_name": batch.get("batch_name"),
            "index": item.get("index"),
            "slug": item.get("slug"),
            "sweep_label": item.get("sweep_label"),
            "sweep_values": item.get("sweep_values") or {},
            "ok": bool(run.get("ok")),
            "experiment_id": run.get("experiment_id"),
            "summary_path": run.get("summary_path"),
            "result_path": run.get("result_path"),
            "csv_path": run.get("csv_path"),
            "csv_exists": bool(csv_path and csv_path.exists()),
        }
        config = item.get("config") or {}
        row.update(
            {
                "model": config.get("model"),
                "dataset": config.get("dataset"),
                "scenario": config.get("scenario"),
                "lr": config_value(config, "training_params.lr"),
                "l2_reg": config_value(config, "training_params.l2_reg"),
                "local_epochs": config_value(config, "training_params.local_epochs"),
                "clients_sample_ratio": config_value(config, "training_params.clients_sample_ratio"),
                "malicious_ratio": config_value(config, "malicious_client_config.ratio"),
                "poisoning_attack_scale": config_value(
                    config, "attack_params.poisoning_attack.poisoning_attack_scale"
                ),
                "robust_defense_mode": config_value(
                    config, "defense_params.robust_defense.robust_defense_mode"
                ),
            }
        )
        if csv_path and csv_path.exists():
            row.update(summarize_csv(csv_path))
        rows.append(row)
    return {
        "batch_name": batch.get("batch_name"),
        "batch_summary_path": str(summary_path),
        "rows": rows,
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Batch Tail Metrics Summary",
        "",
        "| batch | # | slug | ok | scenario | lr | l2 | local_epochs | sample | malicious | attack_scale | defense_mode | tail_source | window | tail_recall50 | tail_ndcg50 | collapse | csv |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for batch in payload.get("batches", []):
        for row in batch.get("rows", []):
            lines.append(
                "| {batch} | {index} | {slug} | {ok} | {scenario} | {lr} | {l2} | {local_epochs} | {sample} | {malicious} | {attack_scale} | {defense_mode} | {source} | {window} | {recall} | {ndcg} | {collapse} | {csv} |".format(
                    batch=row.get("batch_name") or "",
                    index=row.get("index") or "",
                    slug=row.get("slug") or "",
                    ok=row.get("ok"),
                    scenario=row.get("scenario") or "",
                    lr=row.get("lr") if row.get("lr") is not None else "",
                    l2=row.get("l2_reg") if row.get("l2_reg") is not None else "",
                    local_epochs=row.get("local_epochs") if row.get("local_epochs") is not None else "",
                    sample=row.get("clients_sample_ratio") if row.get("clients_sample_ratio") is not None else "",
                    malicious=row.get("malicious_ratio") if row.get("malicious_ratio") is not None else "",
                    attack_scale=row.get("poisoning_attack_scale") if row.get("poisoning_attack_scale") is not None else "",
                    defense_mode=row.get("robust_defense_mode") or "",
                    source=row.get("tail_source") or "",
                    window=row.get("tail_window_size") or "",
                    recall="" if row.get("tail_recall50") is None else "{:.6f}".format(row["tail_recall50"]),
                    ndcg="" if row.get("tail_ndcg50") is None else "{:.6f}".format(row["tail_ndcg50"]),
                    collapse="" if row.get("collapse_ratio") is None else "{:.2%}".format(row["collapse_ratio"]),
                    csv=row.get("csv_path") or "",
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    summary_paths = []
    for raw_path in args.batch_summary:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        summary_paths.append(path)

    batches = [summarize_batch(path) for path in summary_paths]
    prefix = args.output_prefix or batches[0].get("batch_name") or "batch_tail_metrics"
    output_json = RUN_ROOT / f"{prefix}_tail_metrics.json"
    output_md = RUN_ROOT / f"{prefix}_tail_metrics.md"
    payload = {
        "output_json": str(output_json),
        "output_md": str(output_md),
        "batches": batches,
    }
    write_json(output_json, payload)
    write_markdown(output_md, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
