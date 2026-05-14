"""Run the score-based membership probe from exported recommendation rows.

This utility is intentionally lightweight: it reads existing recommendation
artifacts and only runs the probe when real membership labels plus score or rank
signals are present. Legacy TopK files with only ``top_0`` ... ``top_k`` item ids
are reported as not available instead of fabricating member/non-member samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def score_from_row(row: Dict[str, Any]) -> Optional[float]:
    score = to_float(row_value(row, SCORE_KEYS))
    if score is not None:
        return score
    rank = to_float(row_value(row, RANK_KEYS))
    if rank is None:
        return None
    return 1.0 / max(rank, 1.0)


def split_member_scores(rows: Iterable[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    member_scores: List[float] = []
    non_member_scores: List[float] = []
    for row in rows:
        label = parse_membership_label(row_value(row, LABEL_KEYS))
        score = score_from_row(row)
        if label is None or score is None:
            continue
        if label:
            member_scores.append(score)
        else:
            non_member_scores.append(score)
    return member_scores, non_member_scores


def not_available(reason: str, source_files: Sequence[Path]) -> Dict[str, Any]:
    return {
        "status": "not_available",
        "probe_type": "membership_inference",
        "sample_count": 0,
        "member_count": 0,
        "non_member_count": 0,
        "attack_accuracy": None,
        "attack_auc": None,
        "member_score_gap": None,
        "risk_level": "not_available",
        "missing_inputs": ["membership label", "score or rank"],
        "source_files": [str(path) for path in source_files],
        "note": reason,
    }


def run_probe_from_rows(
    rows: Iterable[Dict[str, Any]],
    score_direction: str = "higher_is_member",
    source_files: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    source_files = list(source_files or [])
    member_scores, non_member_scores = split_member_scores(rows)
    if not member_scores or not non_member_scores:
        return not_available(
            "real membership inference probe requires member and non-member rows with score or rank",
            source_files,
        )

    probe = MembershipInferenceProbe(config={"score_direction": score_direction})
    summary = probe.evaluate_scores(member_scores, non_member_scores)
    summary["status"] = "available"
    summary["input_source"] = "recommendation_rows"
    summary["source_files"] = [str(path) for path in source_files]
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
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in recommendation_files:
        rows.extend(read_csv_rows(path))
    if not rows:
        return not_available("recommendation file could not be parsed", recommendation_files)
    return run_probe_from_rows(
        rows,
        score_direction=score_direction,
        source_files=recommendation_files,
    )


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
        files = collect_recommendation_files(
            args.recommendation_file,
            args.recommendation_dir,
        )
        summary = run_probe_from_recommendation_files(
            files,
            score_direction=args.score_direction,
        )

    if args.output_json:
        write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
