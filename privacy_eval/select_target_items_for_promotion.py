"""Select target items that are plausible for target promotion smoke tests.

The selector consumes an existing target_rank_summary.json. It does not infer
missing scores and does not download metadata or images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select rank-near target items for promotion.")
    parser.add_argument("--baseline-rank-summary", required=True)
    parser.add_argument("--item-metadata")
    parser.add_argument("--inter-file")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rank", type=int, default=50)
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument("--min-train-count", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
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
    metadata: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("itemID", item.get("item_id", item.get("id")))
            if item_id is not None:
                metadata[str(item_id)] = item
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("itemID", key)
                metadata[str(key)] = item
    return metadata


def train_counts(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}
    inter_path = Path(path)
    if not inter_path.exists():
        return {}
    counts: Dict[str, int] = {}
    try:
        with inter_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split_label") not in (None, "", "0"):
                    continue
                item_id = row.get("itemID") or row.get("item_id") or row.get("item")
                if item_id not in (None, ""):
                    counts[str(item_id)] = counts.get(str(item_id), 0) + 1
    except Exception:
        return {}
    return counts


def metadata_value(item: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def select_targets(
    rank_summary: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    counts: Dict[str, int],
    min_rank: int,
    max_rank: int,
    min_train_count: int,
    top_n: int,
) -> Dict[str, Any]:
    warnings: List[str] = []
    selected: List[Dict[str, Any]] = []
    summaries = rank_summary.get("target_summaries", {})
    if not isinstance(summaries, dict):
        summaries = {}
    if not summaries:
        warnings.append("baseline rank summary has no target_summaries; cannot select targets")

    for item_id, summary in summaries.items():
        if not isinstance(summary, dict):
            continue
        best_rank = summary.get("best_unmasked_rank")
        mean_rank = summary.get("average_unmasked_rank")
        if best_rank is None:
            continue
        try:
            best_rank_int = int(best_rank)
        except (TypeError, ValueError):
            continue
        if best_rank_int <= min_rank or best_rank_int > max_rank:
            continue
        train_count = counts.get(str(item_id), 0)
        if counts and train_count < min_train_count:
            continue
        item = metadata.get(str(item_id), {})
        title = metadata_value(item, "title", "item_title", "name")
        category = metadata_value(item, "category", "main_category", "target_group", "tag")
        image_url = metadata_value(item, "image_url", "image", "img_url")
        if not (title and category and image_url):
            continue
        selected.append(
            {
                "item_id": str(item_id),
                "baseline_best_rank": best_rank_int,
                "baseline_mean_rank": mean_rank,
                "train_interaction_count": train_count if counts else None,
                "title": title,
                "category": category,
                "image_url": image_url,
                "selection_reason": "baseline rank is between {} and {}, not already Top50, metadata complete".format(
                    min_rank,
                    max_rank,
                ),
            }
        )

    selected = sorted(selected, key=lambda item: (item["baseline_best_rank"], item["item_id"]))[:top_n]
    if not selected:
        warnings.append("no target items matched the rank/metadata/train-count filters")
    return {
        "summary_type": "target_items_promotion",
        "selection_strategy": "rank_near_top_metadata_complete",
        "min_rank": min_rank,
        "max_rank": max_rank,
        "min_train_count": min_train_count,
        "target_item_count": len(selected),
        "target_items": selected,
        "warnings": warnings,
        "note": "Selected items are candidates for smoke testing target promotion; selection does not imply successful manipulation.",
    }


def main() -> int:
    args = build_parser().parse_args()
    rank_summary = read_json(args.baseline_rank_summary) or {}
    payload = select_targets(
        rank_summary,
        load_metadata(args.item_metadata),
        train_counts(args.inter_file),
        args.min_rank,
        args.max_rank,
        args.min_train_count,
        args.top_n,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
