"""Build versioned Workbench results from current-job training evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path | None, fallback: Any = None) -> Any:
    if path is None or not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8-sig"))


def repo_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def anonymize_client(job_id: str, value: Any) -> str:
    digest = hashlib.sha256("{}:{}".format(job_id, value).encode("utf-8")).hexdigest()
    return "client_{}".format(digest[:10])


def load_experiment_result(pointer: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(pointer, dict):
        return {}
    payload = read_json(repo_path(pointer.get("result_path")), {})
    return payload if isinstance(payload, dict) else {}


def _round_eval(round_metric: Dict[str, Any], name: str) -> Dict[str, Any]:
    extra = round_metric.get("extra") if isinstance(round_metric.get("extra"), dict) else {}
    value = extra.get(name)
    return value if isinstance(value, dict) else {}


def training_rounds(pointer: Dict[str, Any] | None, job_id: str) -> List[Dict[str, Any]]:
    result = load_experiment_result(pointer)
    rounds: List[Dict[str, Any]] = []
    for item in result.get("round_metrics", []) if isinstance(result.get("round_metrics"), list) else []:
        if not isinstance(item, dict):
            continue
        test_result = _round_eval(item, "test_result")
        rounds.append(
            {
                "round": int(item.get("round_id") or item.get("round_index") or len(rounds) + 1),
                "loss": number(item.get("train_loss") if item.get("train_loss") is not None else item.get("avg_train_loss")),
                "recall_at_50": number(test_result.get("recall@50")),
                "ndcg_at_50": number(test_result.get("ndcg@50")),
                "participant_count": int(item.get("participant_count") or item.get("num_participants") or 0),
                "malicious_client_count": int(item.get("malicious_client_count") or 0),
                "malicious_client_ids": [anonymize_client(job_id, value) for value in item.get("malicious_clients", [])],
            }
        )
    return rounds


def training_block(metrics: Dict[str, Any], pointer: Dict[str, Any] | None, job_id: str) -> Dict[str, Any]:
    rounds = training_rounds(pointer, job_id)
    return {
        "loss": number(metrics.get("loss")),
        "recall_at_50": number(metrics.get("recall_at_50")),
        "ndcg_at_50": number(metrics.get("ndcg_at_50")),
        "epochs": int(metrics["epochs"]) if number(metrics.get("epochs")) is not None else None,
        "rounds": rounds,
    }


def load_item_metadata(dataset: str) -> Dict[str, Dict[str, Any]]:
    payload = read_json(ROOT / "datasets" / dataset / "item_metadata.json", {})
    rows = payload.get("items", []) if isinstance(payload, dict) else []
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        item_id = row.get("itemID", row.get("item_id"))
        if item_id is not None:
            index[str(item_id)] = row
    return index


def item_info(dataset: str, item_id: Any) -> Dict[str, Any]:
    row = load_item_metadata(dataset).get(str(item_id), {})
    return {
        "item_id": str(item_id),
        "title": row.get("title") or "商品 {}".format(item_id),
        "category": row.get("main_category") or row.get("target_group"),
        "thumbnail_url": "/showcase/images/{}/{}?size=thumb".format(dataset, item_id),
    }


def recommendation_list(preview: Dict[str, Any] | None, dataset: str) -> List[Dict[str, Any]]:
    if not isinstance(preview, dict):
        return []
    metadata = load_item_metadata(dataset)
    output = []
    for index, item_id in enumerate(preview.get("items", []) if isinstance(preview.get("items"), list) else []):
        row = metadata.get(str(item_id), {})
        output.append(
            {
                "item_id": str(item_id),
                "rank": index + 1,
                "title": row.get("title") or "商品 {}".format(item_id),
                "category": row.get("main_category") or row.get("target_group"),
                "thumbnail_url": "/showcase/images/{}/{}?size=thumb".format(dataset, item_id),
            }
        )
    return output


def recommendation_change_stats(baseline: Sequence[Dict[str, Any]], attack: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    before = {str(item["item_id"]): int(item["rank"]) for item in baseline}
    after = {str(item["item_id"]): int(item["rank"]) for item in attack}
    before_ids = set(before)
    after_ids = set(after)
    union = before_ids | after_ids
    intersection = before_ids & after_ids
    moved_up = sum(1 for item_id in intersection if after[item_id] < before[item_id])
    moved_down = sum(1 for item_id in intersection if after[item_id] > before[item_id])
    unchanged = sum(1 for item_id in intersection if after[item_id] == before[item_id])
    return {
        "jaccard": len(intersection) / len(union) if union else None,
        "new_count": len(after_ids - before_ids),
        "removed_count": len(before_ids - after_ids),
        "moved_up_count": moved_up,
        "moved_down_count": moved_down,
        "unchanged_count": unchanged,
        "changed_item_count": len(union) - unchanged,
        "changed_user_count": int(before != after),
    }


def build_recommendation_result(
    normalized: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    attack_metrics: Dict[str, Any],
    defense_metrics: Dict[str, Any] | None,
    target_comparison: Dict[str, Any],
    baseline_preview: Dict[str, Any],
    attack_preview: Dict[str, Any],
    defense_preview: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], List[str]]:
    dataset = str(normalized.get("dataset") or attack_metrics.get("dataset") or "")
    target_id = target_comparison.get("target_item_id") or (normalized.get("attack") or {}).get("target_item_id")
    baseline_list = recommendation_list(baseline_preview, dataset)
    attack_list = recommendation_list(attack_preview, dataset)
    defense_list = recommendation_list(defense_preview, dataset) if defense_preview else []
    changes = recommendation_change_stats(baseline_list, attack_list)
    before_rank = number(target_comparison.get("target_rank_before"))
    after_rank = number(target_comparison.get("target_rank_after"))
    rank_gain = number(target_comparison.get("rank_gain"))
    normalized_gain = rank_gain / max(before_rank - 1.0, 1.0) if rank_gain is not None and before_rank is not None else None
    top50_hit = target_comparison.get("masked_top50_hit") if isinstance(target_comparison.get("masked_top50_hit"), bool) else None
    manipulation_index = normalized_gain * (1.0 if top50_hit else 0.5) if normalized_gain is not None and top50_hit is not None else None
    result = {
        "target_item_id": str(target_id) if target_id is not None else None,
        "target_item_title": (normalized.get("attack") or {}).get("target_item_title"),
        "target_item_category": item_info(dataset, target_id).get("category") if target_id is not None else None,
        "target_item_info": item_info(dataset, target_id) if target_id is not None else None,
        "baseline_unmasked_rank": before_rank,
        "attack_unmasked_rank": after_rank,
        "rank_gain": rank_gain,
        "masked_top50_hit": top50_hit,
        "masked_top50_hit_count": target_comparison.get("masked_top50_hit_count"),
        "masked_top50_hit_rate": number(target_comparison.get("masked_top50_hit_rate")),
        "baseline_recommendations": baseline_list,
        "attack_recommendations": attack_list,
        "defense_recommendations": defense_list if defense_preview else None,
        "recommendation_counts": {
            "baseline": len(baseline_list),
            "attack": len(attack_list),
            "defense": len(defense_list) if defense_preview else None,
        },
        "recommendation_jaccard": changes["jaccard"],
        "list_change_stats": changes,
        "target_manipulation_index": manipulation_index,
        "target_manipulation_formula": "normalized_rank_gain * (1.0 if masked_top50_hit else 0.5)",
        "baseline_metrics": metric_triplet(baseline_metrics),
        "attack_metrics": metric_triplet(attack_metrics),
        "defense_metrics": metric_triplet(defense_metrics or {}) if defense_metrics else None,
    }
    missing = [key for key in ("target_item_id", "baseline_unmasked_rank", "attack_unmasked_rank") if result.get(key) is None]
    if not baseline_list or not attack_list:
        missing.append("recommendation_lists")
    return result, missing


def collect_topk_files(pointer: Dict[str, Any]) -> List[Path]:
    directory = repo_path(pointer.get("recommend_topk_dir"))
    return sorted(directory.rglob("*.csv")) if directory and directory.exists() else []


def build_membership_result(
    job_id: str,
    job_dir: Path,
    normalized: Dict[str, Any],
    pointer: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    from privacy_eval.export_membership_pair_scores import run_export
    from privacy_eval.generate_membership_labels import generate_membership_labels, write_json

    privacy = normalized.get("privacy") if isinstance(normalized.get("privacy"), dict) else {}
    sample_count = max(2, int(privacy.get("membership_sample_count") or 200))
    ratio = max(0.1, float(privacy.get("member_nonmember_ratio") or 1.0))
    member_count = max(1, int(round(sample_count * ratio / (1.0 + ratio))))
    non_member_count = max(1, sample_count - member_count)
    label_source = str(privacy.get("label_source") or "membership_labels")
    warnings: List[str] = []
    missing: List[str] = []
    if label_source in {"scenario labels", "scenario_labels"}:
        return {}, ["scenario_labels_not_available_for_current_job"], ["membership_labels"], {}
    labels = generate_membership_labels(
        dataset=str(normalized.get("dataset")),
        max_members=member_count,
        max_non_members=non_member_count,
        seed=int((normalized.get("training") or {}).get("seed") or 42),
    )
    labels_path = job_dir / "membership_labels.json"
    write_json(labels_path, labels)
    recommendation_files = collect_topk_files(pointer)
    evidence_source = str(privacy.get("mia_evidence_source") or "auto")
    score_mode = {
        "rank": "rank_proxy",
        "unmasked rank": "unmasked_rank",
        "checkpoint score": "checkpoint_score",
        "auto": "auto",
    }.get(evidence_source, "auto")
    rank_paths = [repo_path(pointer.get("target_rank_summary"))] if pointer.get("target_rank_summary") else []
    rank_paths = [path for path in rank_paths if path and path.exists()]
    if evidence_source == "unmasked rank" and not rank_paths:
        warnings.append("requested_unmasked_rank_evidence_not_available")
    if evidence_source == "checkpoint score":
        warnings.append("requested_checkpoint_score_not_available")
    output_csv = job_dir / "membership_pair_scores.csv"
    output_json = job_dir / "membership_score_summary.json"
    summary = run_export(
        labels_path,
        output_csv,
        output_json,
        [],
        recommendation_files,
        target_rank_summary_files=rank_paths,
        model=str(normalized.get("model")),
        dataset=str(normalized.get("dataset")),
        score_mode=score_mode,
        threshold_strategy=str(privacy.get("threshold_strategy") or "auto"),
        mia_model=str(privacy.get("mia_model") or "rank_proxy"),
        anonymize_salt=job_id,
    )
    warnings.extend(str(item) for item in summary.get("warnings", []))
    if summary.get("status") not in {"available", "partial"}:
        missing.append("membership_scores")
    result = {
        "auc": summary.get("attack_auc"),
        "accuracy": summary.get("attack_accuracy"),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": summary.get("f1"),
        "score_gap": summary.get("member_score_gap"),
        "threshold": summary.get("decision_threshold"),
        "evidence_source": summary.get("score_source") or evidence_source,
        "label_source": labels.get("label_source"),
        "mia_model": privacy.get("mia_model"),
        "threshold_strategy": privacy.get("threshold_strategy"),
        "member_count": summary.get("member_count"),
        "non_member_count": summary.get("non_member_count"),
        "sample_count": summary.get("pair_count"),
        "scored_sample_count": summary.get("scored_pair_count"),
        "member_nonmember_ratio": ratio,
        "roc_curve": summary.get("roc_curve", []),
        "score_distribution": summary.get("score_distribution", {}),
        "pair_scores": {
            "path": repo_relative(output_csv),
            "total": summary.get("pair_count", 0),
            "returned": summary.get("pair_count", 0),
            "truncated": False,
            "anonymized": True,
        },
    }
    pointers = {
        "membership_labels": repo_relative(labels_path),
        "membership_pair_scores": repo_relative(output_csv),
        "membership_score_summary": repo_relative(output_json),
    }
    return result, list(dict.fromkeys(warnings)), missing, pointers


def build_update_leakage_result(
    normalized: Dict[str, Any], pointer: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    experiment = load_experiment_result(pointer)
    metadata = experiment.get("metadata") if isinstance(experiment.get("metadata"), dict) else {}
    summaries = metadata.get("privacy_metric_summaries") if isinstance(metadata.get("privacy_metric_summaries"), dict) else {}
    summary = summaries.get("interaction_reconstruction_probe") if isinstance(summaries.get("interaction_reconstruction_probe"), dict) else {}
    privacy = normalized.get("privacy") if isinstance(normalized.get("privacy"), dict) else {}
    dataset = str(normalized.get("dataset") or "")
    metadata_index = load_item_metadata(dataset)
    candidates = []
    for item in summary.get("candidate_scores", []) if isinstance(summary.get("candidate_scores"), list) else []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id"))
        row = metadata_index.get(item_id, {})
        candidates.append(
            {
                **item,
                "item_id": item_id,
                "title": row.get("title") or "商品 {}".format(item_id),
                "category": row.get("main_category") or row.get("target_group"),
                "thumbnail_url": "/showcase/images/{}/{}?size=thumb".format(dataset, item_id),
            }
        )
    result = {
        "hit_at_10": summary.get("hit_at_10"),
        "hit_at_20": summary.get("hit_at_20"),
        "hit_at_50": summary.get("hit_at_50"),
        "input_source": summary.get("input_source") or privacy.get("update_input_source"),
        "target_modality": summary.get("target_modality") or privacy.get("risk_modality"),
        "similarity_method": summary.get("similarity_method") or privacy.get("similarity_method"),
        "audit_client_count": summary.get("audit_client_count") or summary.get("client_count"),
        "candidate_pool_size": summary.get("candidate_pool_size") or privacy.get("candidate_pool_size"),
        "returned_candidate_count": summary.get("returned_candidate_count") or len(candidates),
        "candidates": candidates,
        "per_client_evidence": summary.get("per_client_candidates", {}),
        "highest_risk_modality": summary.get("highest_risk_modality"),
        "candidate_evidence": {
            "total": len(candidates),
            "returned": len(candidates),
            "truncated": False,
        },
    }
    warnings = [str(item) for item in summary.get("warnings", [])] if isinstance(summary.get("warnings"), list) else []
    missing = [] if summary.get("status") == "available" else ["interaction_reconstruction"]
    return result, warnings, missing


def metric_triplet(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss": number(metrics.get("loss")),
        "recall_at_50": number(metrics.get("recall_at_50")),
        "ndcg_at_50": number(metrics.get("ndcg_at_50")),
    }


def _defense_round_detail(pointer: Dict[str, Any], job_id: str) -> List[Dict[str, Any]]:
    experiment = load_experiment_result(pointer)
    output = []
    for item in experiment.get("round_metrics", []) if isinstance(experiment.get("round_metrics"), list) else []:
        if not isinstance(item, dict):
            continue
        extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
        defenses = extra.get("defense_metrics") if isinstance(extra.get("defense_metrics"), dict) else {}
        defense_record = next((value for key, value in defenses.items() if key in {"krum", "median", "trimmed_mean", "bulyan"} and isinstance(value, dict)), {})
        rejected = [str(value) for value in defense_record.get("rejected_clients", [])]
        selected = [str(value) for value in defense_record.get("selected_clients", [])]
        malicious = {str(value) for value in item.get("malicious_clients", [])}
        output.append(
            {
                "round": int(item.get("round_id") or item.get("round_index") or len(output) + 1),
                "participant_count": int(item.get("participant_count") or item.get("num_participants") or 0),
                "malicious_client_count": len(malicious),
                "accepted_client_count": int(defense_record.get("selected_client_count") or len(selected)),
                "rejected_client_count": int(defense_record.get("rejected_client_count") or len(rejected)),
                "rejected_client_ids": [anonymize_client(job_id, value) for value in rejected],
                "correctly_filtered_malicious_count": len(set(rejected) & malicious),
                "false_rejected_normal_count": len(set(rejected) - malicious),
                "aggregation_algorithm": defense_record.get("defense_type"),
                "anomaly_summary": defense_record.get("distance_summary") or defense_record.get("warning_summary") or {},
            }
        )
    return output


def _align_rounds(
    baseline: Sequence[Dict[str, Any]],
    attacked: Sequence[Dict[str, Any]],
    defended: Sequence[Dict[str, Any]],
    defense_details: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_round = lambda rows: {int(row["round"]): row for row in rows}
    baseline_index, attacked_index, defended_index, detail_index = map(by_round, (baseline, attacked, defended, defense_details))
    round_ids = sorted(set(baseline_index) | set(attacked_index) | set(defended_index))
    output = []
    for round_id in round_ids:
        base = baseline_index.get(round_id, {})
        attack = attacked_index.get(round_id, {})
        defense = defended_index.get(round_id, {})
        output.append(
            {
                "round": round_id,
                "baseline": {key: base.get(key) for key in ("loss", "recall_at_50", "ndcg_at_50")},
                "attacked": {key: attack.get(key) for key in ("loss", "recall_at_50", "ndcg_at_50")} if attack else None,
                "defended": {key: defense.get(key) for key in ("loss", "recall_at_50", "ndcg_at_50")} if defense else None,
                **detail_index.get(round_id, {}),
            }
        )
    return output


def recovery(baseline: Optional[float], attacked: Optional[float], defended: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if baseline is None or attacked is None:
        return None, None, None
    degradation = baseline - attacked
    recovered = defended - attacked if defended is not None else None
    rate = recovered / degradation if recovered is not None and degradation > 0 else None
    return degradation, recovered, rate


def build_aggregation_result(
    job_id: str,
    normalized: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    baseline_pointer: Dict[str, Any],
    attacked_metrics: Dict[str, Any] | None,
    attacked_pointer: Dict[str, Any] | None,
    defended_metrics: Dict[str, Any] | None,
    defended_pointer: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], List[str]]:
    base_attack = str((normalized.get("defense") or {}).get("base_attack") or "none")
    baseline = metric_triplet(baseline_metrics)
    attacked = metric_triplet(attacked_metrics or {}) if attacked_metrics else None
    defended = metric_triplet(defended_metrics or {}) if defended_metrics else None
    recall_drop, recall_recovered, recall_rate = recovery(
        baseline.get("recall_at_50"), attacked.get("recall_at_50") if attacked else None, defended.get("recall_at_50") if defended else None
    )
    ndcg_drop, ndcg_recovered, ndcg_rate = recovery(
        baseline.get("ndcg_at_50"), attacked.get("ndcg_at_50") if attacked else None, defended.get("ndcg_at_50") if defended else None
    )
    defense_details = _defense_round_detail(defended_pointer or {}, job_id) if defended_pointer else []
    rounds = _align_rounds(
        training_rounds(baseline_pointer, job_id),
        training_rounds(attacked_pointer or {}, job_id),
        training_rounds(defended_pointer or {}, job_id),
        defense_details,
    )
    latest = defense_details[-1] if defense_details else {}
    result = {
        "base_attack": base_attack,
        "malicious_client_ratio": (normalized.get("attack") or {}).get("malicious_client_ratio") if base_attack == "malicious_update" else None,
        "perturbation_type": ((normalized.get("unified_experiment_config") or {}).get("attack_params") or {}).get("poisoning_attack", {}).get("perturbation_type") if base_attack == "malicious_update" else None,
        "perturbation_strength": ((normalized.get("unified_experiment_config") or {}).get("attack_params") or {}).get("poisoning_attack", {}).get("perturbation_strength") if base_attack == "malicious_update" else None,
        "defense_algorithm": (normalized.get("robust_aggregators") or [None])[0],
        "defense_parameters": normalized.get("defense", {}),
        "dp_noise_enabled": normalized.get("dp_noise_enabled"),
        "update_perturbation": {
            "dp_noise_enabled": normalized.get("dp_noise_enabled"),
            "dp_noise_std": normalized.get("dp_noise_std"),
            "noise_multiplier": normalized.get("noise_multiplier"),
            "max_grad_norm": normalized.get("max_grad_norm"),
            "target_delta": normalized.get("target_delta"),
            "dp_seed": normalized.get("dp_seed"),
        },
        "baseline": baseline,
        "attacked": attacked,
        "defended": defended,
        "recall_degradation": recall_drop,
        "recall_recovery_amount": recall_recovered,
        "recovery_rate_recall": recall_rate,
        "ndcg_degradation": ndcg_drop,
        "ndcg_recovery_amount": ndcg_recovered,
        "recovery_rate_ndcg": ndcg_rate,
        "total_client_count": latest.get("participant_count"),
        "malicious_client_count": latest.get("malicious_client_count") if base_attack == "malicious_update" else None,
        "accepted_client_count": latest.get("accepted_client_count"),
        "rejected_client_count": latest.get("rejected_client_count"),
        "correctly_filtered_malicious_count": latest.get("correctly_filtered_malicious_count") if base_attack == "malicious_update" else None,
        "false_rejected_normal_count": latest.get("false_rejected_normal_count") if base_attack == "malicious_update" else None,
        "filter_precision": (
            latest.get("correctly_filtered_malicious_count") / latest.get("rejected_client_count")
            if base_attack == "malicious_update" and latest.get("rejected_client_count")
            else None
        ),
        "rounds": rounds,
    }
    missing: List[str] = []
    if base_attack == "malicious_update" and attacked is None:
        missing.append("attacked_metrics")
    if normalized.get("robust_aggregators") and defended is None:
        missing.append("defended_metrics")
    if normalized.get("robust_aggregators") and not rounds:
        missing.append("defense_rounds")
    return result, missing


def compatibility_metrics(direction: str, training: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {
        "loss": training.get("loss"),
        "recall_at_50": training.get("recall_at_50"),
        "ndcg_at_50": training.get("ndcg_at_50"),
        "epochs": training.get("epochs"),
    }
    if direction == "recommendation_manipulation":
        metrics.update(
            {
                "target_item_id": result.get("target_item_id"),
                "target_item_title": result.get("target_item_title"),
                "target_item_category": result.get("target_item_category"),
                "target_item_info": result.get("target_item_info"),
                "target_rank_before": result.get("baseline_unmasked_rank"),
                "target_rank_after": result.get("attack_unmasked_rank"),
                "rank_gain": result.get("rank_gain"),
                "masked_top50_hit": result.get("masked_top50_hit"),
                "recommendation_jaccard": result.get("recommendation_jaccard"),
                "target_manipulation_index": result.get("target_manipulation_index"),
                "baseline_top50": {"items": [item["item_id"] for item in result.get("baseline_recommendations", [])], "count": len(result.get("baseline_recommendations", []))},
                "attack_top50": {"items": [item["item_id"] for item in result.get("attack_recommendations", [])], "count": len(result.get("attack_recommendations", []))},
                "defense_top50": {"items": [item["item_id"] for item in result.get("defense_recommendations") or []], "count": len(result.get("defense_recommendations") or [])} if result.get("defense_recommendations") is not None else None,
            }
        )
    elif direction == "membership_inference":
        metrics.update(result)
        metrics["mia_score_distribution"] = result.get("score_distribution")
        metrics["roc_curve"] = result.get("roc_curve")
    elif direction == "update_leakage":
        metrics.update(
            {
                "hit_at_10": result.get("hit_at_10"),
                "hit_at_20": result.get("hit_at_20"),
                "hit_at_50": result.get("hit_at_50"),
                "update_input_source": result.get("input_source"),
                "risk_modality": result.get("target_modality"),
                "similarity_method": result.get("similarity_method"),
                "client_count": result.get("audit_client_count"),
                "candidate_pool_size": result.get("candidate_pool_size"),
                "candidate_k": result.get("returned_candidate_count"),
                "leakage_candidates": result.get("candidates"),
                "per_client_evidence": result.get("per_client_evidence"),
            }
        )
    elif direction == "aggregation_defense":
        defended = result.get("defended") or {}
        metrics.update(
            {
                "base_attack": result.get("base_attack"),
                "defense_algorithm": result.get("defense_algorithm"),
                "defended_recall_at_50": defended.get("recall_at_50"),
                "defended_ndcg_at_50": defended.get("ndcg_at_50"),
                "recovery_rate_recall": result.get("recovery_rate_recall"),
                "recovery_rate_ndcg": result.get("recovery_rate_ndcg"),
                "selected_client_count": result.get("accepted_client_count"),
                "rejected_client_count": result.get("rejected_client_count"),
                "aggregation_rounds": result.get("rounds"),
            }
        )
    return metrics
