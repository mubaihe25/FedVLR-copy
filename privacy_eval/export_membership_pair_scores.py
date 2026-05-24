"""Export membership pair scores from existing sidecar evidence.

This utility is a lightweight bridge toward real score-based membership
inference. It never fabricates model scores: when an explicit score file is
available it uses that score; when only exported TopK ranks are available it
emits a clearly marked rank proxy; when neither is available it writes rows with
empty score/rank and a not_available summary.

Checkpoint-based FedVLR scoring is supported for the lightweight FedAvg/FedRAP
pickle parameter format saved by this repository. Unsupported checkpoints emit a
feasibility summary instead of guessed scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
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


PAIR_SCORE_FIELDS = [
    "user_id",
    "item_id",
    "label",
    "score",
    "rank",
    "score_source",
    "available",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export membership_pair_scores.csv from labels plus score/rank evidence."
    )
    parser.add_argument("--membership-labels", help="membership_labels.json sidecar.")
    parser.add_argument("--recommendation-file", action="append", default=[])
    parser.add_argument("--recommendation-dir")
    parser.add_argument("--score-file", action="append", default=[])
    parser.add_argument("--result-dir", help="Optional result dir for auto-discovering sidecars.")
    parser.add_argument("--checkpoint-path", help="FedVLR saved model parameter pickle.")
    parser.add_argument("--model-checkpoint", help="Alias for --checkpoint-path.")
    parser.add_argument("--model", help="Optional model name for checkpoint feasibility reporting.")
    parser.add_argument("--dataset", help="Optional dataset name for checkpoint feasibility reporting.")
    parser.add_argument("--target-rank-summary", action="append", default=[])
    parser.add_argument(
        "--score-mode",
        default="auto",
        choices=["checkpoint_score", "unmasked_rank", "rank_proxy", "auto"],
        help="Preferred score source. auto uses checkpoint/score files, then unmasked rank, then TopK rank proxy.",
    )
    parser.add_argument("--output-dir", help="Directory for membership_pair_scores.csv and membership_score_summary.json.")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-smoke", action="store_true")
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


def auto_discover_result_files(
    result_dir: Optional[str],
) -> Tuple[Optional[Path], List[Path], List[Path], List[Path]]:
    if not result_dir:
        return None, [], [], []
    root = Path(result_dir)
    if not root.exists():
        return None, [], [], []
    labels = sorted(root.rglob("membership_labels.json"))
    score_files = sorted(root.rglob("membership_pair_scores.csv"))
    target_rank_files = sorted(root.rglob("target_rank_summary*.json"))
    recommendation_files = []
    for topk_dir in root.rglob("recommend_topk"):
        if topk_dir.is_dir():
            recommendation_files.extend(sorted(topk_dir.rglob("*.csv")))
            recommendation_files.extend(sorted(topk_dir.rglob("*.tsv")))
    return (
        labels[0] if labels else None,
        score_files,
        recommendation_files,
        target_rank_files,
    )


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


def build_unmasked_rank_index(files: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        records = payload.get("user_target_records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            user_id = record.get("user_id")
            item_id = record.get("item_id")
            if item_id in (None, ""):
                continue
            rank = to_float(record.get("unmasked_rank"))
            if rank is None:
                continue
            score = to_float(record.get("unmasked_score"))
            if score is None:
                score = 1.0 / (rank + 1.0)
            payload_row = {
                "score": score,
                "rank": int(rank),
                "score_source": "unmasked_rank",
            }
            index[pair_lookup_key(user_id, item_id)] = payload_row
            index.setdefault(pair_lookup_key(None, item_id), payload_row)
    return index


def import_torch() -> Tuple[Any, Optional[str]]:
    try:
        import torch  # type: ignore

        return torch, None
    except Exception as exc:  # noqa: BLE001 - summary should carry the exact blocker.
        return None, str(exc)


def tensor_to_list(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    if hasattr(value, "weight"):
        value = getattr(value, "weight")
    if isinstance(value, dict):
        value = value.get("weight")
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        data = tolist()
    else:
        data = value
    if not isinstance(data, list) or not data:
        return None
    if data and not isinstance(data[0], list):
        data = [data]
    return [[float(item) for item in row] for row in data]


def state_value(state: Dict[str, Any], key: str) -> Any:
    if key in state:
        return state[key]
    for candidate, value in state.items():
        if str(candidate).endswith(key):
            return value
    return None


def vector_from_state(state: Dict[str, Any], key: str) -> Optional[List[float]]:
    value = state_value(state, key)
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    data = tolist() if callable(tolist) else value
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if isinstance(data, list):
        return [float(item) for item in data]
    try:
        return [float(data)]
    except (TypeError, ValueError):
        return None


def client_state_for_user(client_models: Any, user_id: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(client_models, dict) or user_id in (None, ""):
        return None
    candidates = [user_id, str(user_id)]
    try:
        candidates.append(int(str(user_id)))
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        state = client_models.get(candidate)
        if isinstance(state, dict):
            return state
    return None


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def add_vectors(left: Sequence[float], right: Optional[Sequence[float]]) -> List[float]:
    if right is None:
        return [float(value) for value in left]
    return [float(a) + float(b) for a, b in zip(left, right)]


def build_checkpoint_score_index(
    checkpoint_path: Optional[Path],
    rows: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    summary = {
        "checkpoint_scoring_available": False,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "model": model,
        "dataset": dataset,
        "missing_reason": None,
        "recommended_adapter_points": [
            "utils.quick_start._save_model_params stores federated parameter pickle files",
            "FedAvg/FedRAP-style checkpoints need item_commonality plus per-user affine_output state",
            "full model reconstruction can be added later through Config + get_model + load_state_dict",
        ],
        "scored_pair_count": 0,
        "ranked_pair_count": 0,
        "warnings": [],
    }
    if checkpoint_path is None:
        summary["missing_reason"] = "checkpoint path not provided"
        return {}, summary
    if not checkpoint_path.exists():
        summary["missing_reason"] = "checkpoint path does not exist"
        return {}, summary
    torch, torch_error = import_torch()
    if torch is None:
        summary["missing_reason"] = "torch unavailable for loading checkpoint: {}".format(torch_error)
        return {}, summary

    try:
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        summary["missing_reason"] = "checkpoint could not be loaded: {}".format(exc)
        return {}, summary
    if not isinstance(payload, dict):
        summary["missing_reason"] = "checkpoint root is not a dict"
        return {}, summary

    item_commonality = tensor_to_list(payload.get("item_commonality"))
    client_models = payload.get("client_models")
    if item_commonality is None:
        summary["missing_reason"] = "item_commonality weights not found in checkpoint"
        return {}, summary
    if not isinstance(client_models, dict):
        summary["missing_reason"] = "client_models dict not found in checkpoint"
        return {}, summary

    index: Dict[str, Dict[str, Any]] = {}
    ranked_pairs = 0
    for row in rows:
        user_id = row.get("user_id")
        item_id = row.get("item_id")
        try:
            item_index = int(str(item_id))
        except (TypeError, ValueError):
            summary["warnings"].append("non-integer item_id skipped: {}".format(item_id))
            continue
        if item_index < 0 or item_index >= len(item_commonality):
            summary["warnings"].append("item_id out of checkpoint range: {}".format(item_id))
            continue
        state = client_state_for_user(client_models, user_id)
        if state is None:
            summary["warnings"].append("client model state not found for user_id={}".format(user_id))
            continue
        weight = vector_from_state(state, "affine_output.weight")
        bias = vector_from_state(state, "affine_output.bias") or [0.0]
        if weight is None:
            summary["warnings"].append("affine_output.weight missing for user_id={}".format(user_id))
            continue
        personality = tensor_to_list(state_value(state, "item_personality.weight"))
        item_vector = add_vectors(
            item_commonality[item_index],
            personality[item_index] if personality is not None and item_index < len(personality) else None,
        )
        score = sigmoid(dot(item_vector, weight) + float(bias[0]))
        all_scores = []
        for idx, common_vec in enumerate(item_commonality):
            personal_vec = personality[idx] if personality is not None and idx < len(personality) else None
            vec = add_vectors(common_vec, personal_vec)
            all_scores.append(sigmoid(dot(vec, weight) + float(bias[0])))
        rank = 1 + sum(1 for candidate_score in all_scores if candidate_score > score)
        ranked_pairs += 1
        index[pair_lookup_key(user_id, item_id)] = {
            "score": float(score),
            "rank": int(rank),
            "score_source": "checkpoint_model_score",
        }

    if index:
        summary["checkpoint_scoring_available"] = True
        summary["missing_reason"] = None
    else:
        summary["missing_reason"] = "no labeled pairs could be scored from this checkpoint"
    summary["scored_pair_count"] = len(index)
    summary["ranked_pair_count"] = ranked_pairs
    return index, summary


def enrich_rows(
    rows: Iterable[Dict[str, Any]],
    score_index: Dict[str, Dict[str, Any]],
    rank_index: Dict[str, Dict[str, Any]],
    checkpoint_index: Optional[Dict[str, Dict[str, Any]]] = None,
    unmasked_rank_index: Optional[Dict[str, Dict[str, Any]]] = None,
    score_mode: str = "auto",
) -> List[Dict[str, Any]]:
    checkpoint_index = checkpoint_index or {}
    unmasked_rank_index = unmasked_rank_index or {}
    output: List[Dict[str, Any]] = []
    for row in rows:
        user_id = row.get("user_id")
        item_id = row.get("item_id")
        lookup_keys = [pair_lookup_key(user_id, item_id), pair_lookup_key(None, item_id)]
        if score_mode == "checkpoint_score":
            source_indexes = [checkpoint_index, score_index]
        elif score_mode == "unmasked_rank":
            source_indexes = [unmasked_rank_index]
        elif score_mode == "rank_proxy":
            source_indexes = [rank_index]
        else:
            source_indexes = [checkpoint_index, score_index, unmasked_rank_index, rank_index]
        payload = None
        for source_index in source_indexes:
            for lookup_key in lookup_keys:
                payload = source_index.get(lookup_key)
                if payload is not None:
                    break
            if payload is not None:
                break
        output.append(
            {
                "user_id": "" if user_id is None else str(user_id),
                "item_id": "" if item_id is None else str(item_id),
                "label": row.get("label"),
                "score": "" if not payload else payload.get("score", ""),
                "rank": "" if not payload else payload.get("rank", ""),
                "score_source": "" if not payload else payload.get("score_source", ""),
                "available": bool(payload),
            }
        )
    return output


def auc_from_scores(member_scores: Sequence[float], non_member_scores: Sequence[float]) -> Optional[float]:
    if not member_scores or not non_member_scores:
        return None
    wins = 0.0
    total = 0
    for member_score in member_scores:
        for non_member_score in non_member_scores:
            total += 1
            if member_score > non_member_score:
                wins += 1.0
            elif member_score == non_member_score:
                wins += 0.5
    return float(wins / total) if total else None


def mia_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    member_scores = [
        float(row["score"])
        for row in rows
        if row.get("label") == "member" and to_float(row.get("score")) is not None
    ]
    non_member_scores = [
        float(row["score"])
        for row in rows
        if row.get("label") == "non_member" and to_float(row.get("score")) is not None
    ]
    if not member_scores or not non_member_scores:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "member_score_gap": None,
            "risk_level": "not_available",
        }
    member_mean = sum(member_scores) / len(member_scores)
    non_member_mean = sum(non_member_scores) / len(non_member_scores)
    threshold = (member_mean + non_member_mean) / 2.0
    higher_is_member = member_mean >= non_member_mean
    correct = 0
    total = 0
    for row in rows:
        score = to_float(row.get("score"))
        if score is None or row.get("label") not in {"member", "non_member"}:
            continue
        predicted_member = score >= threshold if higher_is_member else score <= threshold
        is_member = row.get("label") == "member"
        correct += int(predicted_member == is_member)
        total += 1
    auc = auc_from_scores(member_scores, non_member_scores)
    if auc is not None and not higher_is_member:
        auc = 1.0 - auc
    accuracy = float(correct / total) if total else None
    gap = float(member_mean - non_member_mean)
    risk_level = "low"
    if accuracy is not None and accuracy >= 0.75 and auc is not None and auc >= 0.75:
        risk_level = "high"
    elif accuracy is not None and accuracy >= 0.6:
        risk_level = "medium"
    return {
        "attack_accuracy": accuracy,
        "attack_auc": auc,
        "member_score_gap": gap,
        "risk_level": risk_level,
    }


def summarize(
    rows: Sequence[Dict[str, Any]],
    label_metadata: Dict[str, Any],
    warnings: Sequence[str],
    model_checkpoint: Optional[str] = None,
    checkpoint_summary: Optional[Dict[str, Any]] = None,
    unmasked_rank_available: bool = False,
    score_mode: str = "auto",
) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("score") not in (None, "")]
    ranked = [row for row in rows if row.get("rank") not in (None, "")]
    sources = sorted({str(row.get("score_source")) for row in rows if row.get("score_source")})
    member_count = sum(1 for row in rows if row.get("label") == "member")
    non_member_count = sum(1 for row in rows if row.get("label") == "non_member")
    status = "available" if scored and member_count and non_member_count else "not_available"
    if scored and len(scored) < len(rows):
        status = "partial"
    checkpoint_summary = checkpoint_summary or {}
    checkpoint_warning = None
    if model_checkpoint and not checkpoint_summary.get("checkpoint_scoring_available"):
        checkpoint_warning = "checkpoint scoring unavailable: {}".format(
            checkpoint_summary.get("missing_reason") or "unsupported checkpoint"
        )
    metrics = mia_metrics(rows)
    score_source = "mixed" if len(sources) > 1 else sources[0] if sources else None
    proxy_only = bool(score_source in {"rank_proxy", "unmasked_rank"} or (sources and all(source in {"rank_proxy", "unmasked_rank"} for source in sources)))
    return {
        "export_type": "membership_pair_scores",
        "metric_type": "membership_pair_scores",
        "summary_type": "membership_score_summary",
        "status": status,
        "pair_count": len(rows),
        "member_count": member_count,
        "non_member_count": non_member_count,
        "scored_pair_count": len(scored),
        "ranked_pair_count": len(ranked),
        "available_pair_count": sum(1 for row in rows if row.get("available") is True),
        "score_source": score_source,
        "score_mode": score_mode,
        "checkpoint_available": bool(
            checkpoint_summary.get("checkpoint_scoring_available", False)
        ),
        "unmasked_rank_available": bool(unmasked_rank_available),
        "checkpoint_scoring_available": bool(
            checkpoint_summary.get("checkpoint_scoring_available", False)
        ),
        "checkpoint_scoring": checkpoint_summary or None,
        "attack_accuracy": metrics["attack_accuracy"],
        "attack_auc": metrics["attack_auc"],
        "member_score_gap": metrics["member_score_gap"],
        "risk_level": metrics["risk_level"],
        "proxy_only": proxy_only,
        "feasibility": {
            "checkpoint_available": bool(
                checkpoint_summary.get("checkpoint_scoring_available", False)
            ),
            "unmasked_rank_available": bool(unmasked_rank_available),
            "rank_proxy_available": any(source == "rank_proxy" for source in sources),
            "future_adapter": (
                "MMFedRAP/full-model scoring should use a Config + model reconstruction adapter; "
                "unsupported checkpoints are not guessed."
            ),
        },
        "label_source": label_metadata.get("label_source"),
        "label_granularity": label_metadata.get("label_granularity"),
        "warnings": list(warnings) + ([checkpoint_warning] if checkpoint_warning else []),
        "note": (
            "Rows with score_source=score_file use supplied model scores. Rows with "
            "score_source=checkpoint_model_score use a supported FedVLR checkpoint. "
            "Rows with score_source=unmasked_rank use target_rank_summary full-score ranks. "
            "Rows with score_source=rank_proxy use 1/(rank+1) from exported TopK."
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
    target_rank_summary_files: Sequence[Path] = (),
    model_checkpoint: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    score_mode: str = "auto",
) -> Dict[str, Any]:
    label_metadata = load_membership_labels(membership_labels)
    warnings = list(label_metadata.get("warnings", []))
    rows = label_rows(label_metadata)
    if not rows:
        warnings.append("membership labels did not contain member/non-member pairs or items")
    score_index = build_score_index(score_files)
    rank_index = build_rank_index(recommendation_files, label_metadata)
    unmasked_rank_index = build_unmasked_rank_index(target_rank_summary_files)
    checkpoint_index, checkpoint_summary = build_checkpoint_score_index(
        Path(model_checkpoint) if model_checkpoint else None,
        rows,
        model=model,
        dataset=dataset,
    )
    output_rows = enrich_rows(
        rows,
        score_index,
        rank_index,
        checkpoint_index=checkpoint_index,
        unmasked_rank_index=unmasked_rank_index,
        score_mode=score_mode,
    )
    summary = summarize(
        output_rows,
        label_metadata,
        warnings,
        model_checkpoint=model_checkpoint,
        checkpoint_summary=checkpoint_summary,
        unmasked_rank_available=bool(unmasked_rank_index),
        score_mode=score_mode,
    )
    summary["membership_labels"] = str(membership_labels)
    summary["score_files"] = [str(path) for path in score_files]
    summary["recommendation_files"] = [str(path) for path in recommendation_files]
    summary["target_rank_summary_files"] = [str(path) for path in target_rank_summary_files]
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


def run_checkpoint_smoke(output_csv: Path, output_json: Path) -> Dict[str, Any]:
    torch, torch_error = import_torch()
    if torch is None:
        summary = {
            "export_type": "membership_pair_scores",
            "status": "not_available",
            "checkpoint_scoring_available": False,
            "warnings": ["torch unavailable for checkpoint smoke: {}".format(torch_error)],
        }
        write_csv(output_csv, [])
        write_json(output_json, summary)
        return summary

    labels_path = output_json.parent / "membership_labels_for_checkpoint_smoke.json"
    checkpoint_path = output_json.parent / "fedavg_checkpoint_smoke.pkl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps(
            {
                "label_source": "synthetic_checkpoint",
                "label_granularity": "user_item_pair",
                "member_pairs": [{"user_id": "1", "item_id": "1"}],
                "non_member_pairs": [{"user_id": "1", "item_id": "3"}],
                "warnings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = {
        "item_commonality": torch.nn.Embedding.from_pretrained(
            torch.tensor(
                [[0.0, 0.0], [2.0, 0.0], [0.5, 0.0], [-1.0, 0.0]],
                dtype=torch.float32,
            ),
            freeze=False,
        ),
        "client_models": {
            1: {
                "affine_output.weight": torch.tensor([[2.0, 0.0]], dtype=torch.float32),
                "affine_output.bias": torch.tensor([0.0], dtype=torch.float32),
            }
        },
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return run_export(
        labels_path,
        output_csv,
        output_json,
        [],
        [],
        [],
        model_checkpoint=str(checkpoint_path),
        model="FedAvg",
        dataset="synthetic",
        score_mode="checkpoint_score",
    )


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.output_dir:
        output_dir = Path(args.output_dir)
        return output_dir / "membership_pair_scores.csv", output_dir / "membership_score_summary.json"
    if not args.output_csv or not args.output_json:
        raise SystemExit("--output-dir or both --output-csv/--output-json are required")
    return Path(args.output_csv), Path(args.output_json)


def main() -> int:
    args = build_parser().parse_args()
    output_csv, output_json = resolve_output_paths(args)
    if args.smoke:
        summary = run_smoke(output_csv, output_json)
    elif args.checkpoint_smoke:
        summary = run_checkpoint_smoke(output_csv, output_json)
    else:
        (
            discovered_labels,
            discovered_scores,
            discovered_recommendations,
            discovered_target_rank_summaries,
        ) = auto_discover_result_files(args.result_dir)
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
        target_rank_summary_files = collect_existing_files(
            args.target_rank_summary,
            None,
        ) + discovered_target_rank_summaries
        summary = run_export(
            membership_labels,
            output_csv,
            output_json,
            list(dict.fromkeys(score_files)),
            list(dict.fromkeys(recommendation_files)),
            list(dict.fromkeys(target_rank_summary_files)),
            model_checkpoint=args.checkpoint_path or args.model_checkpoint,
            model=args.model,
            dataset=args.dataset,
            score_mode=args.score_mode,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
