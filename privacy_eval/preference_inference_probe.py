"""Recommendation-list preference inference probe.

This module infers coarse item groups from existing recommendation artifacts.
When real item metadata is unavailable it can only produce an item-id-group
proxy, which is explicitly marked in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


GROUP_KEYS = ("group", "category", "tag", "tags", "modality", "genre", "class")


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def row_value(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    normalized = {normalize_key(key): key for key in row.keys()}
    for key in keys:
        original = normalized.get(normalize_key(key))
        if original is not None and row.get(original) not in (None, ""):
            return row.get(original)
    return None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    for delimiter in ("\t", ","):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter=delimiter))
            if rows and len(rows[0].keys()) > 1:
                return rows
        except Exception:
            continue
    return []


def is_top_item_column(key: str) -> bool:
    normalized = normalize_key(key)
    return normalized.startswith("top") and normalized[3:].isdigit()


def to_rank(value: Any, default: int) -> int:
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return max(1, int(default))


def expand_recommendation_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        top_columns = sorted(
            [key for key in row.keys() if is_top_item_column(str(key))],
            key=lambda key: int(normalize_key(key)[3:] or "0"),
        )
        if top_columns:
            user_id = row_value(row, ("id", "user_id", "client_id", "user"))
            for rank, column in enumerate(top_columns, start=1):
                item_id = row.get(column)
                if item_id not in (None, ""):
                    expanded.append(
                        {
                            "user_id": user_id,
                            "item_id": item_id,
                            "rank": rank,
                            "score": 1.0 / float(rank),
                        }
                    )
            continue
        item_id = row_value(row, ("item_id", "item", "recommended_item", "recommended_item_id"))
        if item_id is not None:
            expanded.append(
                {
                    "user_id": row_value(row, ("user_id", "client_id", "id")),
                    "item_id": item_id,
                    "rank": to_rank(row_value(row, ("rank", "position", "top_rank")), len(expanded) + 1),
                    "score": row_value(row, ("score", "pred_score", "prediction_score", "rank_score")),
                }
            )
    return expanded


def load_item_metadata(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                item_id = str(value.get("item_id", key))
                metadata[item_id] = value
            else:
                metadata[str(key)] = {"group": value}
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                item_id = item.get("item_id", item.get("id", item.get("item")))
                if item_id is not None:
                    metadata[str(item_id)] = item
    return metadata


def metadata_group(item_id: str, metadata: Dict[str, Dict[str, Any]]) -> Optional[str]:
    item = metadata.get(str(item_id))
    if not isinstance(item, dict):
        return None
    for key in GROUP_KEYS:
        value = item.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if value not in (None, ""):
            return str(value)
    return None


def item_id_proxy_group(item_id: str, bucket_size: int = 50) -> str:
    try:
        numeric_id = int(float(item_id))
        return "item_id_group_{}".format(numeric_id // max(1, bucket_size))
    except (TypeError, ValueError):
        text = str(item_id)
        prefix = text[:2] if text else "unknown"
        return "item_id_prefix_{}".format(prefix)


class PreferenceInferenceProbe(BasePrivacyMetric):
    """Infer coarse preference groups from TopK recommendation rows."""

    def __init__(
        self,
        name: str = "preference_inference_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        config = config or {}
        self.bucket_size = int(config.get("item_id_group_bucket_size", 50))
        self.history: List[Dict[str, Any]] = []

    def evaluate_recommendations(
        self,
        rows: Iterable[Dict[str, Any]],
        item_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        input_source: str = "recommendation_rows",
    ) -> Dict[str, Any]:
        item_metadata = item_metadata or {}
        expanded_rows = expand_recommendation_rows(rows)
        if not expanded_rows:
            return self._not_available("no recommendation rows found", input_source)

        group_weights: Counter[str] = Counter()
        used_metadata = False
        for index, row in enumerate(expanded_rows, start=1):
            item_id = str(row.get("item_id"))
            group = metadata_group(item_id, item_metadata)
            if group is not None:
                used_metadata = True
            else:
                group = item_id_proxy_group(item_id, self.bucket_size)
            rank = to_rank(row.get("rank"), index)
            group_weights[group] += 1.0 / float(rank)

        total_weight = float(sum(group_weights.values()))
        inferred_groups = [
            {"group": group, "weight": float(weight)}
            for group, weight in group_weights.most_common()
        ]
        top_group = inferred_groups[0]["group"] if inferred_groups else None
        confidence = (
            float(inferred_groups[0]["weight"] / total_weight)
            if inferred_groups and total_weight > 0
            else None
        )
        risk_level = "not_available"
        if confidence is not None:
            if confidence >= 0.60:
                risk_level = "high"
            elif confidence >= 0.35:
                risk_level = "medium"
            else:
                risk_level = "low"

        summary = {
            "probe_type": "preference_inference",
            "status": "available",
            "input_source": input_source,
            "recommendation_count": len(expanded_rows),
            "inferred_groups": inferred_groups,
            "confidence": confidence,
            "top_group": top_group,
            "risk_level": risk_level,
            "warning": None if used_metadata else "using item_id_group proxy; no semantic item metadata found",
            "demo_or_proxy": not used_metadata,
            "metadata_used": used_metadata,
            "note": "metadata/rank-based preference inference probe, not a full user profiling attack",
        }
        self.history.append(summary)
        return summary

    def _not_available(self, reason: str, input_source: str) -> Dict[str, Any]:
        summary = {
            "probe_type": "preference_inference",
            "status": "not_available",
            "input_source": input_source,
            "inferred_groups": [],
            "confidence": None,
            "top_group": None,
            "risk_level": "not_available",
            "warning": reason,
            "demo_or_proxy": True,
            "note": "preference inference requires recommendation rows and optional item metadata",
        }
        self.history.append(summary)
        return summary

    def evaluate_round(self, round_state, participant_params, aggregation_result):
        del participant_params
        rows = []
        if isinstance(round_state, dict):
            rows = round_state.get("recommendation_rows", [])
        if not rows and isinstance(aggregation_result, dict):
            rows = aggregation_result.get("recommendation_rows", [])
        return self.evaluate_recommendations(rows, input_source="round_state")

    def summarize(self, experiment_metadata=None):
        if not self.history:
            return self._not_available("no probe history", "unknown")
        latest = dict(self.history[-1])
        latest["num_rounds"] = len(self.history)
        return latest


def collect_recommendation_files(files: Sequence[str], directory: Optional[str]) -> List[Path]:
    paths = [Path(path) for path in files]
    if directory:
        root = Path(directory)
        if root.exists():
            paths.extend(sorted(root.rglob("*.csv")))
            paths.extend(sorted(root.rglob("*.tsv")))
    return [path for path in dict.fromkeys(paths) if path.exists() and path.is_file()]


def run_probe_from_files(
    recommendation_files: Sequence[Path],
    item_metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in recommendation_files:
        rows.extend(read_csv_rows(path))
    metadata = load_item_metadata(item_metadata_path)
    probe = PreferenceInferenceProbe()
    if not rows:
        return probe._not_available("recommendation file could not be parsed", "recommendation_files")
    summary = probe.evaluate_recommendations(
        rows,
        item_metadata=metadata,
        input_source="recommendation_files",
    )
    summary["source_files"] = [str(path) for path in recommendation_files]
    if item_metadata_path is not None:
        summary["item_metadata"] = str(item_metadata_path)
    return summary


def run_synthetic_smoke() -> Dict[str, Any]:
    rows = [
        {"user_id": "u1", "item_id": "101", "rank": "1"},
        {"user_id": "u1", "item_id": "102", "rank": "2"},
        {"user_id": "u1", "item_id": "260", "rank": "3"},
    ]
    metadata = {
        "101": {"group": "visual_items"},
        "102": {"group": "visual_items"},
        "260": {"group": "text_items"},
    }
    summary = PreferenceInferenceProbe().evaluate_recommendations(
        rows,
        item_metadata=metadata,
        input_source="synthetic_smoke",
    )
    assert summary["status"] == "available"
    assert summary["top_group"] == "visual_items"
    assert summary["confidence"] is not None
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preference inference from recommendation files.")
    parser.add_argument("--recommendation-file", action="append", default=[])
    parser.add_argument("--recommendation-dir")
    parser.add_argument("--item-metadata")
    parser.add_argument("--output-json")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        summary = run_synthetic_smoke()
    else:
        files = collect_recommendation_files(args.recommendation_file, args.recommendation_dir)
        metadata_path = Path(args.item_metadata) if args.item_metadata else None
        summary = run_probe_from_files(files, item_metadata_path=metadata_path)
    if args.output_json:
        write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


register_privacy_metric("preference_inference_probe", PreferenceInferenceProbe)
register_privacy_metric("preference_inference", PreferenceInferenceProbe)
register_privacy_metric("preferenceinferenceprobe", PreferenceInferenceProbe)
