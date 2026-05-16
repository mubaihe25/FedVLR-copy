"""Export membership pair scores from existing sidecar evidence.

This utility is a lightweight bridge toward real score-based membership
inference. It never fabricates model scores: when an explicit score file is
available it uses that score; when only exported TopK ranks are available it
emits a clearly marked rank proxy; when neither is available it writes rows with
empty score/rank and a not_available summary.

Checkpoint-based FedVLR scoring is intentionally left as future adapter work
because model reconstruction requires matching the exact model/data config.
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

from privacy_eval.run_membership_probe_from_recommendations import (
    RANK_KEYS,
    SCORE_KEYS,
    expand_legacy_topk_rows,
    item_key,
    load_membership_labels,
    read_csv_rows,
    row_value,
    to_float,
)


PAIR_SCORE_FIELDS = ["user_id", "item_id", "label", "score", "rank", "score_source"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export membership_pair_scores.csv from labels plus score/rank evidence."
    )
    parser.add_argument("--membership-labels", help="membership_labels.json sidecar.")
    parser.add_argument("--recommendation-file", action="append", default=[])
    parser.add_argument("--recommendation-dir")
    parser.add_argument("--score-file", action="append", default=[])
    parser.add_argument("--result-dir", help="Optional result dir for auto-discovering sidecars.")
    parser.add_argument("--model-checkpoint", help="Reserved for future exact-model scoring adapter.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def split_pair_key(pair: str) -> Tuple[Optional[str], str]:
    if "::" in pair:
        user_id, item_id = pair.split("::", 1)
        return user_id, item_id
    return None, pair


def label_rows(label_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in sorted(label_metadata.get("member_pairs", [])):
        user_id, item_id = split_pair_key(str(pair))
        rows.append({"user_id": user_id, "item_id": item_id, "label": "member"})
    for pair in sorted(label_metadata.get("non_member_pairs", [])):
        user_id, item_id = split_pair_key(str(pair))
        rows.append({"user_id": user_id, "item_id": item_id, "label": "non_member"})
    if not rows:
        for item_id in sorted(label_metadata.get("member_items", []), key=str):
            rows.append({"user_id": None, "item_id": str(item_id), "label": "member"})
        for item_id in sorted(label_metadata.get("non_member_items", []), key=str):
            rows.append({"user_id": None, "item_id": str(item_id), "label": "non_member"})
    return rows


def pair_lookup_key(user_id: Any, item_id: Any) -> str:
    if user_id not in (None, ""):
        return item_key(user_id, item_id)
    return "item::{}".format(str(item_id))


def collect_existing_files(paths: Sequence[str], directory: Optional[str]) -> List[Path]:
    files = [Path(path) for path in paths]
    if directory:
        root = Path(directory)
        if root.exists():
            files.extend(sorted(root.rglob("*.csv")))
            files.extend(sorted(root.rglob("*.tsv")))
    return [path for path in dict.fromkeys(files) if path.exists() and path.is_file()]


def auto_discover_result_files(result_dir: Optional[str]) -> Tuple[Optional[Path], List[Path], List[Path]]:
    if not result_dir:
        return None, [], []
    root = Path(result_dir)
    if not root.exists():
        return None, [], []
    labels = sorted(root.rglob("membership_labels.json"))
    score_files = sorted(root.rglob("membership_pair_scores.csv"))
    recommendation_files = []
    for topk_dir in root.rglob("recommend_topk"):
        if topk_dir.is_dir():
            recommendation_files.extend(sorted(topk_dir.rglob("*.csv")))
            recommendation_files.extend(sorted(topk_dir.rglob("*.tsv")))
    return (labels[0] if labels else None), score_files, recommendation_files


def score_from_flat_row(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    score = to_float(row_value(row, SCORE_KEYS))
    if score is not None:
        return score, to_float(row_value(row, RANK_KEYS)), "score_file"
    rank = to_float(row_value(row, RANK_KEYS))
    if rank is not None:
        return 1.0 / (rank + 1.0), rank, "rank_proxy"
    return None, None, None


def build_score_index(files: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in files:
        for row in read_csv_rows(path):
            user_id = row_value(row, ("user_id", "client_id", "id", "user"))
            item_id = row_value(row, ("item_id", "recommended_item_id", "item", "recommended_item"))
            if item_id in (None, ""):
                continue
            score, rank, source = score_from_flat_row(row)
            if score is None and rank is None:
                continue
            payload = {
                "score": score,
                "rank": rank,
                "score_source": source,
            }
            index[pair_lookup_key(user_id, item_id)] = payload
            index.setdefault(pair_lookup_key(None, item_id), payload)
    return index


def build_rank_index(files: Sequence[Path], label_metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    member_pairs = label_metadata.get("member_pairs", set())
    non_member_pairs = label_metadata.get("non_member_pairs", set())
    member_items = label_metadata.get("member_items", set())
    non_member_items = label_metadata.get("non_member_items", set())
    for path in files:
        expanded = expand_legacy_topk_rows(
            read_csv_rows(path),
            member_pairs,
            non_member_pairs,
            member_items=member_items,
            non_member_items=non_member_items,
        )
        for row in expanded:
            user_id = row_value(row, ("user_id", "client_id", "id", "user"))
            item_id = row_value(row, ("item_id", "recommended_item_id", "item", "recommended_item"))
            if item_id in (None, ""):
                continue
            rank = to_float(row_value(row, RANK_KEYS))
            if rank is None:
                continue
            payload = {
                "score": 1.0 / (rank + 1.0),
                "rank": rank,
                "score_source": "rank_proxy",
            }
            index[pair_lookup_key(user_id, item_id)] = payload
            index.setdefault(pair_lookup_key(None, item_id), payload)
    return index


def enrich_rows(
    rows: Iterable[Dict[str, Any]],
    score_index: Dict[str, Dict[str, Any]],
    rank_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        user_id = row.get("user_id")
        item_id = row.get("item_id")
        payload = score_index.get(pair_lookup_key(user_id, item_id))
        if payload is None:
            payload = score_index.get(pair_lookup_key(None, item_id))
        if payload is None:
            payload = rank_index.get(pair_lookup_key(user_id, item_id))
        if payload is None:
            payload = rank_index.get(pair_lookup_key(None, item_id))
        output.append(
            {
                "user_id": "" if user_id is None else str(user_id),
                "item_id": "" if item_id is None else str(item_id),
                "label": row.get("label"),
                "score": "" if not payload else payload.get("score", ""),
                "rank": "" if not payload else payload.get("rank", ""),
                "score_source": "" if not payload else payload.get("score_source", ""),
            }
        )
    return output


def summarize(
    rows: Sequence[Dict[str, Any]],
    label_metadata: Dict[str, Any],
    warnings: Sequence[str],
    model_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("score") not in (None, "")]
    ranked = [row for row in rows if row.get("rank") not in (None, "")]
    sources = sorted({str(row.get("score_source")) for row in rows if row.get("score_source")})
    member_count = sum(1 for row in rows if row.get("label") == "member")
    non_member_count = sum(1 for row in rows if row.get("label") == "non_member")
    status = "available" if scored and member_count and non_member_count else "not_available"
    if scored and len(scored) < len(rows):
        status = "partial"
    checkpoint_note = None
    if model_checkpoint:
        checkpoint_note = (
            "model checkpoint scoring adapter is not implemented in this lightweight exporter"
        )
    return {
        "export_type": "membership_pair_scores",
        "status": status,
        "pair_count": len(rows),
        "member_count": member_count,
        "non_member_count": non_member_count,
        "scored_pair_count": len(scored),
        "ranked_pair_count": len(ranked),
        "score_source": "mixed" if len(sources) > 1 else sources[0] if sources else None,
        "label_source": label_metadata.get("label_source"),
        "label_granularity": label_metadata.get("label_granularity"),
        "warnings": list(warnings) + ([checkpoint_note] if checkpoint_note else []),
        "note": (
            "Rows with score_source=score_file use supplied model scores. Rows with "
            "score_source=rank_proxy use 1/(rank+1) from exported TopK; this is not "
            "unmasked checkpoint scoring."
        ),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_export(
    membership_labels: Path,
    output_csv: Path,
    output_json: Path,
    score_files: Sequence[Path],
    recommendation_files: Sequence[Path],
    model_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    label_metadata = load_membership_labels(membership_labels)
    warnings = list(label_metadata.get("warnings", []))
    rows = label_rows(label_metadata)
    if not rows:
        warnings.append("membership labels did not contain member/non-member pairs or items")
    score_index = build_score_index(score_files)
    rank_index = build_rank_index(recommendation_files, label_metadata)
    output_rows = enrich_rows(rows, score_index, rank_index)
    summary = summarize(output_rows, label_metadata, warnings, model_checkpoint=model_checkpoint)
    summary["membership_labels"] = str(membership_labels)
    summary["score_files"] = [str(path) for path in score_files]
    summary["recommendation_files"] = [str(path) for path in recommendation_files]
    write_csv(output_csv, output_rows)
    write_json(output_json, summary)
    return summary


def run_smoke(output_csv: Path, output_json: Path) -> Dict[str, Any]:
    labels_path = output_json.parent / "membership_labels_for_pair_score_smoke.json"
    score_path = output_json.parent / "membership_scores_for_pair_score_smoke.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps(
            {
                "label_source": "synthetic",
                "label_granularity": "user_item_pair",
                "member_pairs": [{"user_id": "u1", "item_id": "i1"}],
                "non_member_pairs": [{"user_id": "u1", "item_id": "i9"}],
                "warnings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id", "score"])
        writer.writeheader()
        writer.writerow({"user_id": "u1", "item_id": "i1", "score": "0.91"})
        writer.writerow({"user_id": "u1", "item_id": "i9", "score": "0.22"})
    return run_export(labels_path, output_csv, output_json, [score_path], [])


def main() -> int:
    args = build_parser().parse_args()
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    if args.smoke:
        summary = run_smoke(output_csv, output_json)
    else:
        discovered_labels, discovered_scores, discovered_recommendations = auto_discover_result_files(
            args.result_dir
        )
        membership_labels = Path(args.membership_labels) if args.membership_labels else discovered_labels
        if membership_labels is None:
            summary = {
                "export_type": "membership_pair_scores",
                "status": "not_available",
                "warnings": ["membership_labels.json is required"],
                "note": "No membership pair score rows were exported.",
            }
            write_csv(output_csv, [])
            write_json(output_json, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        score_files = collect_existing_files(args.score_file, None) + discovered_scores
        recommendation_files = collect_existing_files(
            args.recommendation_file,
            args.recommendation_dir,
        ) + discovered_recommendations
        summary = run_export(
            membership_labels,
            output_csv,
            output_json,
            list(dict.fromkeys(score_files)),
            list(dict.fromkeys(recommendation_files)),
            model_checkpoint=args.model_checkpoint,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
