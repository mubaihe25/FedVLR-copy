"""Compare baseline and attack target_rank_summary.json files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare target rank/score summaries.")
    parser.add_argument("--baseline-rank-summary", required=True)
    parser.add_argument("--attack-rank-summary", required=True)
    parser.add_argument("--previous-attack-rank-summary")
    parser.add_argument("--item-metadata")
    parser.add_argument("--target-item-ids", nargs="*", default=[])
    parser.add_argument("--output-json", required=True)
    return parser


def read_json(path: Optional[str]) -> Any:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def load_metadata(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    if payload is None:
        return {}
    items = payload.get("items") if isinstance(payload, dict) else payload
    output: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                item_id = item.get("itemID", item.get("item_id", item.get("id")))
                if item_id is not None:
                    output[str(item_id)] = item
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                output[str(key)] = value
    return output


def records_by_item(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for record in payload.get("user_target_records", []):
        if isinstance(record, dict):
            output.setdefault(str(record.get("item_id")), []).append(record)
    return output


def best_score(records: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [
        float(record["unmasked_score"])
        for record in records
        if isinstance(record.get("unmasked_score"), (int, float))
    ]
    return max(values) if values else None


def stats(payload: Dict[str, Any], item_id: str) -> Dict[str, Any]:
    summary = payload.get("target_summaries", {}).get(str(item_id), {})
    records = records_by_item(payload).get(str(item_id), [])
    return {
        "best_unmasked_rank": summary.get("best_unmasked_rank"),
        "mean_unmasked_rank": summary.get("average_unmasked_rank"),
        "best_unmasked_score": best_score(records),
        "mean_unmasked_score": summary.get("average_unmasked_score"),
    }


def rank_shift(before: Any, after: Any) -> Optional[float]:
    if before is None or after is None:
        return None
    return float(before) - float(after)


def score_gain(before: Any, after: Any) -> Optional[float]:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def metadata_value(item: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def infer_target_ids(
    baseline: Dict[str, Any],
    attack: Dict[str, Any],
    explicit: Sequence[str],
) -> List[str]:
    if explicit:
        return sorted({str(item) for item in explicit}, key=lambda value: (len(value), value))
    ids = set(baseline.get("target_summaries", {}).keys()) | set(
        attack.get("target_summaries", {}).keys()
    )
    return sorted(ids, key=lambda value: (len(value), value))


def build_comparison(
    baseline: Dict[str, Any],
    attack: Dict[str, Any],
    previous_attack: Optional[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    target_ids: Sequence[str],
    baseline_path: str,
    attack_path: str,
    previous_path: Optional[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for item_id in target_ids:
        item = metadata.get(str(item_id), {})
        base_stats = stats(baseline, item_id)
        attack_stats = stats(attack, item_id)
        previous_stats = stats(previous_attack or {}, item_id) if previous_attack else None
        row = {
            "item_id": str(item_id),
            "title": metadata_value(item, "title", "item_title", "name"),
            "category": metadata_value(item, "category", "main_category", "target_group", "tag"),
            "image_url": metadata_value(item, "image_url", "image", "img_url"),
            "baseline": base_stats,
            "attack": attack_stats,
            "best_rank_shift_positive_is_better": rank_shift(
                base_stats["best_unmasked_rank"],
                attack_stats["best_unmasked_rank"],
            ),
            "mean_rank_shift_positive_is_better": rank_shift(
                base_stats["mean_unmasked_rank"],
                attack_stats["mean_unmasked_rank"],
            ),
            "best_score_gain": score_gain(
                base_stats["best_unmasked_score"],
                attack_stats["best_unmasked_score"],
            ),
            "mean_score_gain": score_gain(
                base_stats["mean_unmasked_score"],
                attack_stats["mean_unmasked_score"],
            ),
            "target_entered_top50": bool(
                attack_stats["best_unmasked_rank"] is not None
                and int(attack_stats["best_unmasked_rank"]) <= 50
            ),
        }
        if previous_stats is not None:
            row["previous_attack"] = previous_stats
            row["attack_vs_previous_best_rank_shift_positive_is_better"] = rank_shift(
                previous_stats["best_unmasked_rank"],
                attack_stats["best_unmasked_rank"],
            )
            row["attack_vs_previous_mean_rank_shift_positive_is_better"] = rank_shift(
                previous_stats["mean_unmasked_rank"],
                attack_stats["mean_unmasked_rank"],
            )
        rows.append(row)
    return {
        "metric_type": "target_rank_score",
        "summary_type": "target_rank_comparison",
        "score_type": "unmasked_and_masked",
        "baseline_rank_summary": baseline_path,
        "attack_rank_summary": attack_path,
        "previous_attack_rank_summary": previous_path,
        "baseline_evaluated_user_count": baseline.get("evaluated_user_count"),
        "attack_evaluated_user_count": attack.get("evaluated_user_count"),
        "target_item_ids": list(target_ids),
        "rows": rows,
        "target_entered_top50": any(row["target_entered_top50"] for row in rows),
        "target_entered_top50_count": sum(1 for row in rows if row["target_entered_top50"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Positive rank shift means attack moved closer to rank 1. This is a diagnostic summary, not a success claim by itself.",
    }


def main() -> int:
    args = build_parser().parse_args()
    baseline = read_json(args.baseline_rank_summary) or {}
    attack = read_json(args.attack_rank_summary) or {}
    previous = read_json(args.previous_attack_rank_summary)
    target_ids = infer_target_ids(baseline, attack, args.target_item_ids)
    payload = build_comparison(
        baseline,
        attack,
        previous if isinstance(previous, dict) else None,
        load_metadata(args.item_metadata),
        target_ids,
        args.baseline_rank_summary,
        args.attack_rank_summary,
        args.previous_attack_rank_summary,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
