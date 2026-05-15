"""Run the score-based membership probe from exported recommendation rows.

This utility is intentionally lightweight: it reads existing recommendation
artifacts and only runs the probe when membership labels plus score or rank
signals are present. Legacy TopK files with only ``top_0`` ... ``top_k`` item ids
can use rank-based proxy scores, but still require a membership_labels.json
sidecar. If labels are missing or one class is absent, the output is
``not_available`` instead of fabricated member/non-member samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_eval.membership_inference_probe import MembershipInferenceProbe


LABEL_KEYS = (
    "is_member",
    "member",
    "membership",
    "membership_label",
    "label",
    "split",
)
SCORE_KEYS = (
    "score",
    "pred_score",
    "prediction_score",
    "rank_score",
    "confidence",
)
RANK_KEYS = ("rank", "position", "top_rank")
MEMBER_VALUES = {"1", "true", "yes", "member", "train", "training", "seen"}
NON_MEMBER_VALUES = {
    "0",
    "false",
    "no",
    "non_member",
    "nonmember",
    "test",
    "negative",
    "unseen",
}


def empty_label_metadata() -> Dict[str, Any]:
    return {
        "label_source": None,
        "label_granularity": None,
        "member_pairs": set(),
        "non_member_pairs": set(),
        "member_items": set(),
        "non_member_items": set(),
        "warnings": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a score/rank-based membership probe from recommendation files."
    )
    parser.add_argument(
        "--recommendation-file",
        action="append",
        default=[],
        help="Recommendation CSV/TSV file. Can be provided multiple times.",
    )
    parser.add_argument(
        "--recommendation-dir",
        help="Directory containing recommendation CSV/TSV files.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for writing the membership inference summary JSON.",
    )
    parser.add_argument(
        "--membership-labels",
        help="Optional membership_labels.json with real member/non-member labels.",
    )
    parser.add_argument(
        "--score-direction",
        default="higher_is_member",
        choices=["higher_is_member", "lower_is_member", "loss_lower_is_member"],
        help="Probe score direction passed to MembershipInferenceProbe.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a synthetic recommendation-row smoke instead of reading files.",
    )
    return parser


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


def normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def row_value(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    normalized = {normalize_key(key): key for key in row.keys()}
    for key in keys:
        original = normalized.get(normalize_key(key))
        if original is None:
            continue
        value = row.get(original)
        if value not in (None, ""):
            return value
    return None


def item_key(user_id: Any, item_id: Any) -> str:
    return "{}::{}".format(str(user_id), str(item_id))


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_membership_label(value: Any) -> Optional[bool]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in MEMBER_VALUES:
        return True
    if normalized in NON_MEMBER_VALUES:
        return False
    return None


def row_label_from_sets(
    row: Dict[str, Any],
    member_pairs: Set[str],
    non_member_pairs: Set[str],
    member_items: Optional[Set[str]] = None,
    non_member_items: Optional[Set[str]] = None,
) -> Optional[bool]:
    user_id = row_value(row, ("user_id", "client_id", "id", "user"))
    item_id = row_value(row, ("item_id", "item", "recommended_item", "recommended_item_id"))
    if item_id is None:
        return None
    if user_id is not None:
        pair_key = item_key(user_id, item_id)
        if pair_key in member_pairs:
            return True
        if pair_key in non_member_pairs:
            return False
    member_items = member_items or set()
    non_member_items = non_member_items or set()
    if str(item_id) in member_items:
        return True
    if str(item_id) in non_member_items:
        return False
    return None


def add_pair_from_value(target: Set[str], value: Any, default_user_id: Any = None) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            add_pair_from_value(target, item, default_user_id=default_user_id)
        return
    if isinstance(value, dict):
        user_id = value.get("user_id", value.get("client_id", value.get("id", default_user_id)))
        item_id = value.get("item_id", value.get("item", value.get("recommended_item_id")))
        if user_id is not None and item_id is not None:
            target.add(item_key(user_id, item_id))
            return
        for key, item in value.items():
            if isinstance(item, bool):
                continue
            if isinstance(item, (list, tuple, set)):
                for nested_item in item:
                    add_pair_from_value(target, {"user_id": key, "item_id": nested_item})
            else:
                add_pair_from_value(target, item, default_user_id=key)
        return
    if isinstance(value, str) and "::" in value:
        target.add(value)
        return
    if default_user_id is not None:
        target.add(item_key(default_user_id, value))


def add_item_values(target: Set[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            add_item_values(target, item)
        return
    if isinstance(value, dict):
        item_id = value.get("item_id", value.get("item", value.get("recommended_item_id")))
        if item_id is not None:
            target.add(str(item_id))
            return
        for item in value.values():
            add_item_values(target, item)
        return
    target.add(str(value))


def load_membership_labels(path: Optional[Path]) -> Dict[str, Any]:
    metadata = empty_label_metadata()
    if path is None or not path.exists():
        return metadata
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        metadata["warnings"].append("membership_labels could not be parsed")
        return metadata

    member_pairs: Set[str] = set()
    non_member_pairs: Set[str] = set()
    member_items: Set[str] = set()
    non_member_items: Set[str] = set()

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            label = parse_membership_label(row_value(item, LABEL_KEYS))
            target = member_pairs if label is True else non_member_pairs if label is False else None
            if target is not None:
                add_pair_from_value(target, item)
        metadata.update(
            {
                "label_source": "inline_rows",
                "label_granularity": "user_item_pair",
                "member_pairs": member_pairs,
                "non_member_pairs": non_member_pairs,
                "member_items": member_items,
                "non_member_items": non_member_items,
            }
        )
        return metadata

    if not isinstance(payload, dict):
        metadata["warnings"].append("membership_labels root must be object or list")
        return metadata

    metadata["label_source"] = payload.get("label_source")
    metadata["label_granularity"] = payload.get("label_granularity")
    metadata["warnings"].extend(payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else [])

    for key in ("members", "member", "member_samples", "member_pairs"):
        add_pair_from_value(member_pairs, payload.get(key))
    for key in (
        "non_members",
        "non_member",
        "non_member_samples",
        "non_member_pairs",
        "negative_samples",
    ):
        add_pair_from_value(non_member_pairs, payload.get(key))
    for key in ("member_items", "members_items", "train_items"):
        add_item_values(member_items, payload.get(key))
    for key in ("non_member_items", "non_members_items", "test_only_items", "negative_items"):
        add_item_values(non_member_items, payload.get(key))

    labels = payload.get("labels")
    if isinstance(labels, dict):
        for pair_key, label_value in labels.items():
            label = parse_membership_label(label_value)
            if label is True:
                member_pairs.add(str(pair_key))
            elif label is False:
                non_member_pairs.add(str(pair_key))
    elif isinstance(labels, list):
        for item in labels:
            if not isinstance(item, dict):
                continue
            label = parse_membership_label(row_value(item, LABEL_KEYS))
            target = member_pairs if label is True else non_member_pairs if label is False else None
            if target is not None:
                add_pair_from_value(target, item)

    if metadata["label_granularity"] is None:
        metadata["label_granularity"] = (
            "user_item_pair" if member_pairs or non_member_pairs else "item_level_proxy"
        )
    metadata.update(
        {
            "member_pairs": member_pairs,
            "non_member_pairs": non_member_pairs,
            "member_items": member_items,
            "non_member_items": non_member_items,
        }
    )
    return metadata


def load_membership_label_sets(path: Optional[Path]) -> Tuple[Set[str], Set[str]]:
    metadata = load_membership_labels(path)
    return metadata["member_pairs"], metadata["non_member_pairs"]


def score_from_row(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    score = to_float(row_value(row, SCORE_KEYS))
    if score is not None:
        source = row.get("_score_source") or "score"
        return score, str(source)
    rank = to_float(row_value(row, RANK_KEYS))
    if rank is None:
        return None, None
    return 1.0 / (rank + 1.0), "rank_based_proxy"


def split_member_scores(
    rows: Iterable[Dict[str, Any]],
    member_pairs: Optional[Set[str]] = None,
    non_member_pairs: Optional[Set[str]] = None,
    member_items: Optional[Set[str]] = None,
    non_member_items: Optional[Set[str]] = None,
) -> Tuple[List[float], List[float], Dict[str, int]]:
    member_pairs = member_pairs or set()
    non_member_pairs = non_member_pairs or set()
    member_items = member_items or set()
    non_member_items = non_member_items or set()
    member_scores: List[float] = []
    non_member_scores: List[float] = []
    score_sources: Dict[str, int] = {}
    for row in rows:
        label = parse_membership_label(row_value(row, LABEL_KEYS))
        if label is None:
            label = row_label_from_sets(
                row,
                member_pairs,
                non_member_pairs,
                member_items=member_items,
                non_member_items=non_member_items,
            )
        score, score_source = score_from_row(row)
        if label is None or score is None:
            continue
        if score_source:
            score_sources[score_source] = score_sources.get(score_source, 0) + 1
        if label:
            member_scores.append(score)
        else:
            non_member_scores.append(score)
    return member_scores, non_member_scores, score_sources


def is_top_item_column(key: str) -> bool:
    normalized = normalize_key(key)
    return normalized.startswith("top") and normalized[3:].isdigit()


def expand_legacy_topk_rows(
    rows: Iterable[Dict[str, Any]],
    member_pairs: Set[str],
    non_member_pairs: Set[str],
    member_items: Optional[Set[str]] = None,
    non_member_items: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    member_items = member_items or set()
    non_member_items = non_member_items or set()
    expanded_rows: List[Dict[str, Any]] = []
    for row in rows:
        top_columns = sorted(
            [key for key in row.keys() if is_top_item_column(str(key))],
            key=lambda key: int(normalize_key(key)[3:] or "0"),
        )
        if not top_columns:
            expanded_rows.append(row)
            continue
        user_id = row_value(row, ("id", "user_id", "client_id", "user"))
        for rank, column in enumerate(top_columns):
            item_id = row.get(column)
            if item_id in (None, ""):
                continue
            expanded_row = {
                "user_id": user_id,
                "item_id": item_id,
                "rank": rank,
                "_score_source": "rank_based_proxy",
            }
            label = row_label_from_sets(
                expanded_row,
                member_pairs,
                non_member_pairs,
                member_items=member_items,
                non_member_items=non_member_items,
            )
            if label is not None:
                expanded_row["is_member"] = "1" if label else "0"
            expanded_rows.append(expanded_row)
    return expanded_rows


def not_available(
    reason: str,
    source_files: Sequence[Path],
    warnings: Optional[Sequence[str]] = None,
    label_source: Optional[str] = None,
    label_granularity: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": "not_available",
        "probe_type": "membership_inference",
        "label_source": label_source,
        "label_granularity": label_granularity,
        "score_source": None,
        "sample_count": 0,
        "member_count": 0,
        "non_member_count": 0,
        "attack_accuracy": None,
        "attack_auc": None,
        "member_score_gap": None,
        "risk_level": "not_available",
        "missing_inputs": ["membership label", "score or rank"],
        "source_files": [str(path) for path in source_files],
        "warnings": list(warnings or []),
        "proxy_only": False,
        "note": reason,
    }


def run_probe_from_rows(
    rows: Iterable[Dict[str, Any]],
    score_direction: str = "higher_is_member",
    source_files: Optional[Sequence[Path]] = None,
    member_pairs: Optional[Set[str]] = None,
    non_member_pairs: Optional[Set[str]] = None,
    member_items: Optional[Set[str]] = None,
    non_member_items: Optional[Set[str]] = None,
    label_source: Optional[str] = None,
    label_granularity: Optional[str] = None,
    label_warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    source_files = list(source_files or [])
    member_pairs = member_pairs or set()
    non_member_pairs = non_member_pairs or set()
    member_items = member_items or set()
    non_member_items = non_member_items or set()
    label_warnings = list(label_warnings or [])
    expanded_rows = expand_legacy_topk_rows(
        rows,
        member_pairs,
        non_member_pairs,
        member_items=member_items,
        non_member_items=non_member_items,
    )
    has_inline_labels = any(
        parse_membership_label(row_value(row, LABEL_KEYS)) is not None
        for row in expanded_rows
    )
    if not (member_pairs or non_member_pairs or member_items or non_member_items or has_inline_labels):
        return not_available(
            "membership_labels.json is required for recommendation-based membership inference",
            source_files,
            warnings=label_warnings + ["membership labels missing"],
            label_source=label_source,
            label_granularity=label_granularity,
        )

    member_scores, non_member_scores, score_sources = split_member_scores(
        expanded_rows,
        member_pairs=member_pairs,
        non_member_pairs=non_member_pairs,
        member_items=member_items,
        non_member_items=non_member_items,
    )
    if not member_scores or not non_member_scores:
        return not_available(
            "real membership inference probe requires member and non-member rows with score or rank",
            source_files,
            warnings=label_warnings
            + [
                "insufficient labeled recommendation rows: member_count={}, non_member_count={}".format(
                    len(member_scores),
                    len(non_member_scores),
                )
            ],
            label_source=label_source,
            label_granularity=label_granularity,
        )

    probe = MembershipInferenceProbe(config={"score_direction": score_direction})
    summary = probe.evaluate_scores(member_scores, non_member_scores)
    summary["status"] = "available"
    summary["input_source"] = "recommendation_rows"
    summary["label_source"] = label_source or ("inline_recommendation_rows" if has_inline_labels else None)
    summary["label_granularity"] = label_granularity or ("row_label" if has_inline_labels else None)
    summary["score_source"] = (
        "rank_based_proxy"
        if score_sources and set(score_sources.keys()) == {"rank_based_proxy"}
        else "mixed" if len(score_sources) > 1 else next(iter(score_sources), None)
    )
    summary["rank_based_proxy_score"] = "rank_based_proxy" in score_sources
    summary["proxy_only"] = bool(
        summary["rank_based_proxy_score"] or label_granularity == "item_level_proxy"
    )
    summary["warnings"] = list(label_warnings)
    summary["source_files"] = [str(path) for path in source_files]
    summary["requires_real_membership_labels"] = True
    return summary


def collect_recommendation_files(
    recommendation_files: Sequence[str],
    recommendation_dir: Optional[str],
) -> List[Path]:
    paths = [Path(path) for path in recommendation_files]
    if recommendation_dir:
        directory = Path(recommendation_dir)
        if directory.exists():
            paths.extend(sorted(directory.rglob("*.csv")))
            paths.extend(sorted(directory.rglob("*.tsv")))
    return [path for path in dict.fromkeys(paths) if path.exists() and path.is_file()]


def run_probe_from_recommendation_files(
    recommendation_files: Sequence[Path],
    score_direction: str = "higher_is_member",
    membership_labels: Optional[Path] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in recommendation_files:
        rows.extend(read_csv_rows(path))
    if not rows:
        return not_available("recommendation file could not be parsed", recommendation_files)
    if membership_labels is None:
        candidate_dirs = {path.parent for path in recommendation_files}
        candidate_dirs.update(
            path.parent.parent for path in recommendation_files if path.parent.parent
        )
        for directory in candidate_dirs:
            candidate = directory / "membership_labels.json"
            if candidate.exists():
                membership_labels = candidate
                break
    label_metadata = load_membership_labels(membership_labels)
    summary = run_probe_from_rows(
        rows,
        score_direction=score_direction,
        source_files=recommendation_files,
        member_pairs=label_metadata["member_pairs"],
        non_member_pairs=label_metadata["non_member_pairs"],
        member_items=label_metadata["member_items"],
        non_member_items=label_metadata["non_member_items"],
        label_source=label_metadata["label_source"],
        label_granularity=label_metadata["label_granularity"],
        label_warnings=label_metadata["warnings"],
    )
    if membership_labels is not None:
        summary["membership_labels"] = str(membership_labels)
    return summary


def run_synthetic_recommendation_smoke() -> Dict[str, Any]:
    rows = [
        {"user_id": "u1", "item_id": "i1", "rank": "1", "score": "0.91", "is_member": "1"},
        {"user_id": "u1", "item_id": "i2", "rank": "2", "score": "0.83", "is_member": "member"},
        {"user_id": "u1", "item_id": "i3", "rank": "8", "score": "0.42", "is_member": "0"},
        {"user_id": "u1", "item_id": "i4", "rank": "9", "score": "0.37", "is_member": "non_member"},
    ]
    summary = run_probe_from_rows(rows)
    assert summary["status"] == "available"
    assert summary["attack_accuracy"] is not None
    assert summary["member_score_gap"] is not None
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        summary = run_synthetic_recommendation_smoke()
    else:
        membership_labels = Path(args.membership_labels) if args.membership_labels else None
        files = collect_recommendation_files(
            args.recommendation_file,
            args.recommendation_dir,
        )
        summary = run_probe_from_recommendation_files(
            files,
            score_direction=args.score_direction,
            membership_labels=membership_labels,
        )

    if args.output_json:
        write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
