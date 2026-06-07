from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_ID = "amazon_beauty_poc_security_v3"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "showcase_artifacts"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "AMAZON_BEAUTY_POC"
DEFAULT_SIDECARS = ROOT / "outputs" / "security_sidecars" / "AMAZON_BEAUTY_POC"
DEFAULT_IMAGE_MANIFEST = DEFAULT_DATASET_ROOT / "item_image_manifest.json"
DEFAULT_MODEL_MATRIX = (
    ROOT
    / "outputs"
    / "model_security_capability_matrix"
    / "model_security_capability_matrix.json"
)
DEFAULT_SOURCE_DIRS = [
    ROOT / "outputs" / "showcase_artifacts" / "amazon_beauty_poc_v25_backend_smoke",
    ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "AMAZON_BEAUTY_POC"
    / "AmazonBeautyPOCTargetPromotionV25Smoke",
    ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "AMAZON_BEAUTY_POC"
    / "AmazonBeautyPOCBaselineObservationSmoke",
    ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "AMAZON_BEAUTY_POC"
    / "AmazonBeautyPOCInteractionReconstructionV25Smoke",
    ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "AMAZON_BEAUTY_POC"
    / "AmazonBeautyPOCSecuritySmoke",
    ROOT / "outputs" / "security_smokes",
    ROOT / "outputs" / "showcase_artifacts" / "model_security_capability_matrix",
    ROOT / "outputs" / "model_security_capability_matrix",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export structured V3 security showcase panels from existing FedVLR artifacts."
    )
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--display-name", default="Amazon Beauty PoC Security Artifact V3")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--security-sidecars", default=str(DEFAULT_SIDECARS))
    parser.add_argument("--image-manifest", default=str(DEFAULT_IMAGE_MANIFEST))
    parser.add_argument("--model-matrix", default=str(DEFAULT_MODEL_MATRIX))
    parser.add_argument("--source-dir", action="append", default=[])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_relative(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_item_id(value: Any) -> Optional[str]:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def newest(paths: Iterable[Path]) -> Optional[Path]:
    files = [path for path in paths if path.exists() and path.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def source_dirs(explicit: Sequence[str]) -> List[Path]:
    dirs: List[Path] = [Path(raw) for raw in explicit] if explicit else list(DEFAULT_SOURCE_DIRS)
    existing: List[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            existing.append(path)
    return existing


def find_file(sources: Sequence[Path], names: Sequence[str] = (), patterns: Sequence[str] = ()) -> Optional[Path]:
    candidates: List[Path] = []
    for source in sources:
        if source.is_file():
            if source.name in names or any(source.match(pattern) for pattern in patterns):
                candidates.append(source)
            continue
        for name in names:
            path = source / name
            if path.exists() and path.is_file():
                candidates.append(path)
            candidates.extend(source.rglob(name))
        for pattern in patterns:
            candidates.extend(source.rglob(pattern))
    return newest(candidates)


def read_csv_rows(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def find_round_csv(sources: Sequence[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for source in sources:
        if not source.exists() or source.is_file():
            continue
        for path in source.rglob("*.csv"):
            if "recommend_topk" in [part.lower() for part in path.parts]:
                continue
            try:
                header = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
            except Exception:
                continue
            if "round_index" in header and ("train_loss" in header or "test_recall50" in header):
                candidates.append(path)
    return newest(candidates)


def metadata_index(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(dataset_root / "item_metadata.json")
    items = payload.get("items") if isinstance(payload, dict) else payload
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = normalize_item_id(item.get("itemID") or item.get("item_id") or item.get("id"))
            if item_id is not None:
                result[item_id] = item
    elif isinstance(items, dict):
        for key, value in items.items():
            if isinstance(value, dict):
                result[str(key)] = dict(value)
                result[str(key)].setdefault("itemID", str(key))
    return result


def image_index(image_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for item in image_manifest.get("items", []):
        if isinstance(item, dict):
            item_id = normalize_item_id(item.get("item_id") or item.get("itemID"))
            if item_id is not None:
                result[item_id] = item
    return result


def first_text(item: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def category_text(item: Dict[str, Any]) -> Optional[str]:
    direct = first_text(item, ["category", "item_category", "main_category"])
    if direct:
        return direct
    categories = item.get("categories")
    if isinstance(categories, list):
        values: List[str] = []
        for value in categories:
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, list):
                values.extend(str(part).strip() for part in value if str(part).strip())
        if values:
            return " / ".join(values[:3])
    return None


def item_card(item_id: Any, metadata: Dict[str, Dict[str, Any]], images: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    normalized = normalize_item_id(item_id)
    item = metadata.get(normalized or "", {})
    image = images.get(normalized or "", {})
    return {
        "item_id": normalized,
        "title": first_text(item, ["title", "item_title", "name"]) or image.get("title"),
        "category": category_text(item) or image.get("category"),
        "image_url": first_text(item, ["image_url", "image", "img_url"]) or image.get("image_url"),
        "local_image_url": image.get("local_image_path"),
        "thumbnail_url": image.get("thumbnail_path"),
    }


def experiment_identity(summary: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    return (
        str(summary.get("model") or "FedAvg"),
        str(summary.get("dataset") or "AMAZON_BEAUTY_POC"),
        list(summary.get("active_attacks") or ["target_interaction_injection"]),
        list(summary.get("active_defenses") or []),
    )


def build_training_curves(round_csv: Optional[Path], manipulation: Dict[str, Any], attack_defense: Dict[str, Any]) -> Dict[str, Any]:
    rows = [
        row for row in read_csv_rows(round_csv)
        if str(row.get("row_type", "")).lower() == "round" or row.get("round_index")
    ]
    loss: List[Dict[str, Any]] = []
    recall: List[Dict[str, Any]] = []
    ndcg: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        round_index = int(to_float(row.get("round_index")) or index)
        loss_value = to_float(row.get("train_loss"))
        recall_value = to_float(row.get("test_recall50") or row.get("recall50"))
        ndcg_value = to_float(row.get("test_ndcg50") or row.get("ndcg50"))
        if loss_value is not None:
            loss.append({"round": round_index, "value": loss_value})
        if recall_value is not None:
            recall.append({"round": round_index, "value": recall_value})
        if ndcg_value is not None:
            ndcg.append({"round": round_index, "value": ndcg_value})

    curve_source = "real_points" if loss or recall or ndcg else "unavailable"
    warning = None if curve_source == "real_points" else "round-level training CSV not found"
    attack_risk = {
        "curve_source": "summary_curve" if manipulation else "unavailable",
        "risk_level": manipulation.get("manipulation_risk_level"),
        "target_exposure_gain": manipulation.get("target_exposure_gain"),
        "baseline_attack_jaccard": manipulation.get("baseline_attack_jaccard"),
    }
    defense_recovery = {
        "curve_source": "summary_curve" if attack_defense else "unavailable",
        "recall_recovery_rate": attack_defense.get("recall_recovery_rate"),
        "ndcg_recovery_rate": attack_defense.get("ndcg_recovery_rate"),
        "warning": "no defense run is attached" if not attack_defense.get("defense") else None,
    }
    return {
        "summary_type": "security_artifact_v3_training_curves",
        "curve_source": curve_source,
        "loss": loss,
        "recall_at_50": recall,
        "ndcg_at_50": ndcg,
        "attack_risk": attack_risk,
        "defense_recovery": defense_recovery,
        "source_csv": repo_relative(round_csv),
        "warning": warning,
    }


def build_target_metrics(
    target_rank: Dict[str, Any],
    manipulation: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    images: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    targets: List[Dict[str, Any]] = []
    target_rows = target_rank.get("rows", []) if isinstance(target_rank.get("rows"), list) else []
    topk_exposure_gain = to_float(manipulation.get("target_exposure_gain")) or 0.0
    recommendation_jaccard = to_float(manipulation.get("baseline_attack_jaccard"))
    list_change_score = 1.0 - recommendation_jaccard if recommendation_jaccard is not None else 0.0
    list_change_score = clamp(list_change_score)
    exposure_score = clamp(topk_exposure_gain)
    attack_topk_hit = bool((to_float(manipulation.get("target_hit_rate_attack")) or 0.0) > 0.0)
    baseline_topk_hit = bool((to_float(manipulation.get("target_hit_rate_baseline")) or 0.0) > 0.0)

    for row in target_rows:
        if not isinstance(row, dict):
            continue
        item_id = normalize_item_id(row.get("item_id"))
        baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
        attack = row.get("attack") if isinstance(row.get("attack"), dict) else {}
        baseline_rank = to_float(baseline.get("best_unmasked_rank"))
        attack_rank = to_float(attack.get("best_unmasked_rank"))
        rank_gain = (baseline_rank - attack_rank) if baseline_rank is not None and attack_rank is not None else None
        normalized_rank_gain = (
            clamp(rank_gain / max(baseline_rank - 1.0, 1.0))
            if rank_gain is not None and baseline_rank is not None
            else None
        )
        reciprocal_rank_gain = (
            (1.0 / attack_rank) - (1.0 / baseline_rank)
            if baseline_rank not in (None, 0) and attack_rank not in (None, 0)
            else None
        )
        rank_gain_score = normalized_rank_gain or 0.0
        index = rank_gain_score * 0.45 + exposure_score * 0.35 + list_change_score * 0.20
        item = item_card(item_id, metadata, images)
        item.update(
            {
                "target_item_id": item_id,
                "target_title": row.get("title") or item.get("title"),
                "target_image_url": row.get("image_url") or item.get("image_url"),
                "baseline_unmasked_rank": int(baseline_rank) if baseline_rank is not None else None,
                "attack_unmasked_rank": int(attack_rank) if attack_rank is not None else None,
                "rank_gain": rank_gain,
                "normalized_rank_gain": normalized_rank_gain,
                "reciprocal_rank_gain": reciprocal_rank_gain,
                "baseline_topk_hit": baseline_topk_hit,
                "attack_topk_hit": attack_topk_hit,
                "topk_exposure_gain": topk_exposure_gain,
                "recommendation_jaccard": recommendation_jaccard,
                "changed_user_count": manipulation.get("changed_user_count"),
                "changed_item_count": manipulation.get("changed_item_count"),
                "target_entered_unmasked_top50": bool(row.get("target_entered_top50")),
                "target_manipulation_index": index,
                "interpretation": (
                    "rank_significantly_pushed_but_masked_topk_not_hit"
                    if row.get("target_entered_top50") and not attack_topk_hit
                    else "topk_exposure_hit" if attack_topk_hit else "rank_diagnostic_only"
                ),
            }
        )
        targets.append(item)

    primary = targets[0] if targets else {}
    result = {
        "summary_type": "security_artifact_v3_target_manipulation_metrics",
        "status": "available" if targets else "unavailable",
        "targets": targets,
        "index_formula": "rank_gain_score * 0.45 + exposure_score * 0.35 + list_change_score * 0.20",
        "rank_gain_score_definition": "clamp((baseline_unmasked_rank - attack_unmasked_rank) / max(baseline_unmasked_rank - 1, 1), 0, 1)",
        "exposure_score_definition": "clamp(target_exposure_gain, 0, 1)",
        "list_change_score_definition": "clamp(1 - recommendation_jaccard, 0, 1)",
        "boundary": "If masked Top50 exposure is not hit, attack_topk_hit remains false even when unmasked rank improves.",
    }
    result.update(primary)
    return result


def summarize_score_distribution(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    by_label: Dict[str, List[float]] = {"member": [], "non_member": []}
    for row in rows:
        score = to_float(row.get("score"))
        if score is None:
            continue
        label = str(row.get("label"))
        key = "member" if label in {"1", "member", "true", "True"} else "non_member"
        by_label[key].append(score)

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return {"member": stats(by_label["member"]), "non_member": stats(by_label["non_member"])}


def anonymized_examples(rows: List[Dict[str, str]], limit: int = 5) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if len(examples) >= limit:
            break
        if str(row.get("available", "")).lower() not in {"true", "1", "yes"}:
            continue
        examples.append(
            {
                "example_id": "pair_{:03d}".format(index + 1),
                "label": row.get("label"),
                "item_id": normalize_item_id(row.get("item_id")),
                "score": to_float(row.get("score")),
                "rank": to_float(row.get("rank")),
                "score_source": row.get("score_source"),
            }
        )
    return examples


def build_membership_panel(summary: Dict[str, Any], pair_csv: Optional[Path]) -> Dict[str, Any]:
    rows = read_csv_rows(pair_csv)
    score_source = summary.get("score_source")
    evidence_type = "checkpoint_score" if score_source == "checkpoint_model_score" else score_source
    if summary.get("proxy_only"):
        evidence_type = "{}_proxy".format(score_source or "rank")
    return {
        "summary_type": "security_artifact_v3_membership_inference_panel",
        "status": summary.get("status") or ("available" if summary else "unavailable"),
        "evidence_type": evidence_type,
        "auc": summary.get("attack_auc"),
        "accuracy": summary.get("attack_accuracy"),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": summary.get("f1"),
        "score_gap": summary.get("member_score_gap"),
        "member_count": summary.get("member_count"),
        "non_member_count": summary.get("non_member_count"),
        "threshold": summary.get("membership_threshold"),
        "score_distribution_summary": summarize_score_distribution(rows),
        "anonymized_examples": anonymized_examples(rows),
        "proxy_only": bool(summary.get("proxy_only")),
        "source_csv": repo_relative(pair_csv),
        "warnings": list(summary.get("warnings", [])),
        "boundary": "Rank/unmasked-rank evidence is proxy evidence and is not reported as checkpoint score.",
    }


def build_update_leakage_panel(
    interaction: Dict[str, Any],
    update_risk: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    images: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_ids = interaction.get("candidate_item_ids") or []
    candidate_scores = interaction.get("candidate_scores") or {}
    candidates: List[Dict[str, Any]] = []
    for rank, item_id in enumerate(candidate_ids[:20], start=1):
        card = item_card(item_id, metadata, images)
        card.update({"rank": rank, "score": candidate_scores.get(str(item_id)) if isinstance(candidate_scores, dict) else None})
        candidates.append(card)

    return {
        "summary_type": "security_artifact_v3_update_leakage_panel",
        "status": interaction.get("status") or update_risk.get("risk_level") or "unavailable",
        "client_count": interaction.get("client_count") or update_risk.get("client_count"),
        "candidate_item_count": interaction.get("candidate_item_count"),
        "hit_at_10": interaction.get("hit_at_10"),
        "hit_at_20": interaction.get("hit_at_20"),
        "hit_at_50": interaction.get("hit_at_50"),
        "highest_risk_modality": interaction.get("highest_risk_modality") or update_risk.get("highest_risk_modality"),
        "modality_risk_breakdown": interaction.get("modality_breakdown") or update_risk.get("modality_risk_breakdown"),
        "update_norm_summary": {
            "source": update_risk.get("source"),
            "update_norm_mean": update_risk.get("update_norm_mean"),
            "update_norm_max": update_risk.get("update_norm_max"),
            "update_norm_std": update_risk.get("update_norm_std"),
            "sparsity_mean": update_risk.get("sparsity_mean"),
            "energy_mean": update_risk.get("energy_mean"),
            "leakage_risk_score": update_risk.get("leakage_risk_score"),
            "risk_level": update_risk.get("risk_level"),
        },
        "candidate_items": candidates,
        "limitation": "candidate reconstruction, not full user history",
        "warnings": list(interaction.get("warnings", [])) + list(update_risk.get("warnings", [])),
    }


def build_aggregation_defense_panel(defense_trace: Dict[str, Any], defense_matrix: Dict[str, Any], attack_defense: Dict[str, Any]) -> Dict[str, Any]:
    cases = defense_matrix.get("cases") or defense_trace.get("defense_matrix", {}).get("cases") or []
    algorithms = sorted({str(case.get("defense_mode")) for case in cases if case.get("defense_mode")})
    config_only = bool(cases) and all(str(case.get("source_type")) == "config" for case in cases)
    return {
        "summary_type": "security_artifact_v3_aggregation_defense_panel",
        "status": "configured_only" if config_only else ("available" if defense_trace or defense_matrix else "unavailable"),
        "defense_algorithm": algorithms or attack_defense.get("defense_type"),
        "aggregation_visibility": "plaintext_updates",
        "selected_clients": defense_trace.get("selected_indices"),
        "rejected_clients": defense_trace.get("rejected_indices"),
        "outlier_score_summary": defense_trace.get("outlier_score_summary"),
        "recall_before": (attack_defense.get("attack") or {}).get("recall50") if isinstance(attack_defense.get("attack"), dict) else None,
        "recall_after": (attack_defense.get("defense") or {}).get("recall50") if isinstance(attack_defense.get("defense"), dict) else None,
        "ndcg_before": (attack_defense.get("attack") or {}).get("ndcg50") if isinstance(attack_defense.get("attack"), dict) else None,
        "ndcg_after": (attack_defense.get("defense") or {}).get("ndcg50") if isinstance(attack_defense.get("defense"), dict) else None,
        "recovery_rate": {
            "recall": attack_defense.get("recall_recovery_rate"),
            "ndcg": attack_defense.get("ndcg_recovery_rate"),
        },
        "defense_trace": {
            "matrix_status": defense_matrix.get("matrix_status"),
            "case_count": defense_matrix.get("case_count"),
            "cases": cases,
        },
        "incompatible_with_secure_aggregation": True,
        "warning": "config/validate-only cases do not imply a real defense benchmark" if config_only else None,
    }


def build_privacy_defense_panel(opacus: Dict[str, Any], secure_agg: Dict[str, Any], defense_trace: Dict[str, Any], defense_matrix: Dict[str, Any]) -> Dict[str, Any]:
    dp_noise = defense_trace.get("dp_noise") or {"status": "not_available", "formal_accountant": False}
    formal_epsilon = opacus.get("epsilon") if opacus.get("opacus_available") else None
    return {
        "summary_type": "security_artifact_v3_privacy_defense_panel",
        "dp_noise": {
            "status": dp_noise.get("status"),
            "formal_accountant": False,
            "description": "central/noise style update defense, not formal DP",
        },
        "opacus": {
            "status": "available" if opacus.get("opacus_available") else "unavailable",
            "opacus_available": bool(opacus.get("opacus_available")),
            "demo_only": opacus.get("demo_only"),
            "epsilon": formal_epsilon,
            "delta": opacus.get("delta"),
        },
        "secure_aggregation": {
            "status": "demo" if secure_agg else "unavailable",
            "demo_only": secure_agg.get("demo_only"),
            "masked_individual_updates_visible": secure_agg.get("masked_individual_updates_visible"),
            "not_production_cryptographic_protocol": secure_agg.get("not_production_cryptographic_protocol"),
        },
        "robust_aggregation": {
            "status": defense_matrix.get("matrix_status") or "unavailable",
            "case_count": defense_matrix.get("case_count"),
        },
        "formal_dp_available": bool(opacus.get("opacus_available")),
        "formal_dp_epsilon": formal_epsilon,
        "secagg_demo_residual": secure_agg.get("aggregate_residual_norm"),
        "warning": [
            "dp_noise is central/noise style and not formal DP",
            "Opacus toy is standalone and not integrated into FedVLR training",
            "secure aggregation is simulation/demo, not a production cryptographic protocol",
        ],
    }


def build_model_support_panel(matrix: Dict[str, Any]) -> Dict[str, Any]:
    entries = matrix.get("entries", []) if isinstance(matrix.get("entries"), list) else []
    by_status: Dict[str, List[str]] = {"supported": [], "partial": [], "future_adapter": [], "unsupported": []}
    direction_support: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = "{}::{}".format(entry.get("model"), entry.get("dataset"))
        status = str(entry.get("status"))
        if status in by_status and key not in by_status[status]:
            by_status[status].append(key)
        direction_support.setdefault(str(entry.get("capability")), {})[key] = status
    return {
        "summary_type": "security_artifact_v3_model_support_panel",
        "supported_models": sorted(by_status["supported"]),
        "partial_models": sorted(by_status["partial"]),
        "adapter_required_models": sorted(by_status["future_adapter"]),
        "unsupported_models": sorted(by_status["unsupported"]),
        "recommended_showcase_models": {
            "multimodal_showcase": "MMFedRAP::KU",
            "security_validation_base": "FedAvg::AMAZON_BEAUTY_POC",
        },
        "model_direction_support": direction_support,
        "status_counts": matrix.get("status_counts"),
        "warnings": matrix.get("warnings", []),
    }


def build_runtime_timeline() -> Dict[str, Any]:
    events = [
        (1, "train", "客户端本地训练", "FedAvg clients train locally on Amazon Beauty PoC.", "completed", "normal"),
        (2, "attack", "恶意客户端上传目标更新", "Target interaction injection modifies malicious clients' in-memory local batches.", "completed", "attack"),
        (3, "aggregate", "服务端聚合", "Server aggregates participant updates for the target-promotion run.", "completed", "normal"),
        (4, "audit", "目标商品排序审计", "Unmasked and masked target ranks are exported for target item auditing.", "completed", "audit"),
        (5, "export", "推荐列表与隐私审计 artifact 导出", "TopK, manipulation, MIA, leakage, and model support artifacts are exported.", "completed", "audit"),
        (6, "audit", "推荐结果读取", "Recommendation TopK files are read for membership and manipulation panels.", "completed", "audit"),
        (7, "audit", "membership labels 读取", "Membership labels are read from security sidecars when available.", "completed", "audit"),
        (8, "audit", "排名证据生成", "Checkpoint scores are preferred; unmasked rank and TopK rank proxy are used when scores are unavailable.", "completed", "audit"),
        (9, "audit", "MIA probe", "Membership inference summary reports AUC/accuracy and proxy boundaries.", "completed", "audit"),
        (10, "audit", "participant params 读取", "Real participant_params are summarized for update leakage risk.", "completed", "audit"),
        (11, "audit", "item embedding 风险分析", "Item-like update magnitudes are used for candidate reconstruction.", "completed", "audit"),
        (12, "audit", "候选商品还原", "Candidate items are enriched with metadata and cached image paths.", "completed", "audit"),
        (13, "audit", "hit@K 输出", "hit@10/hit@20/hit@50 are exported when train-interaction references exist.", "completed", "audit"),
        (14, "defense", "客户端更新读取", "Robust defense matrix reads config/result summaries only.", "configured", "defense"),
        (15, "defense", "鲁棒聚合执行", "Krum/trimmed_mean/median cases are configured; no full benchmark is implied.", "configured", "defense"),
        (16, "defense", "异常更新过滤", "Rejected client fields remain unavailable without a real defense run.", "configured", "defense"),
        (17, "defense", "Recall/NDCG/恢复率输出", "Recovery metrics remain unavailable/null when no defense result is attached.", "configured", "defense"),
    ]
    return {
        "summary_type": "security_artifact_v3_runtime_timeline",
        "events": [
            {
                "step": step,
                "type": event_type,
                "label": label,
                "description": description,
                "status": status,
                "severity": severity,
                "order": step,
            }
            for step, event_type, label, description, status, severity in events
        ],
    }


def build_frontend_summary(
    target: Dict[str, Any],
    membership: Dict[str, Any],
    leakage: Dict[str, Any],
    aggregation: Dict[str, Any],
    privacy: Dict[str, Any],
    model_support: Dict[str, Any],
) -> Dict[str, Any]:
    attack_hit = target.get("attack_topk_hit")
    rank_gain = target.get("rank_gain")
    headline = "Target rank moved strongly, while masked TopK exposure remains not hit."
    return {
        "summary_type": "security_artifact_v3_frontend_summary",
        "headline": headline,
        "key_metrics": {
            "target_rank_gain": rank_gain,
            "target_attack_topk_hit": attack_hit,
            "target_manipulation_index": target.get("target_manipulation_index"),
            "mia_auc": membership.get("auc"),
            "mia_accuracy": membership.get("accuracy"),
            "interaction_hit_at_10": leakage.get("hit_at_10"),
            "interaction_hit_at_20": leakage.get("hit_at_20"),
            "interaction_hit_at_50": leakage.get("hit_at_50"),
            "secagg_demo_residual": privacy.get("secagg_demo_residual"),
            "model_status_counts": model_support.get("status_counts"),
        },
        "direction_cards": [
            {
                "direction": "recommendation_manipulation",
                "status": target.get("status"),
                "headline_metric": target.get("target_manipulation_index"),
                "warning": "Top50 exposure hit is false when masked TopK does not include the target.",
            },
            {
                "direction": "membership_inference",
                "status": membership.get("status"),
                "headline_metric": membership.get("auc"),
                "warning": "Rank/unmasked-rank proxy is not checkpoint score." if membership.get("proxy_only") else None,
            },
            {
                "direction": "update_leakage",
                "status": leakage.get("status"),
                "headline_metric": leakage.get("hit_at_50"),
                "warning": leakage.get("limitation"),
            },
            {
                "direction": "aggregation_defense",
                "status": aggregation.get("status"),
                "headline_metric": aggregation.get("recovery_rate"),
                "warning": aggregation.get("warning"),
            },
        ],
        "next_step_suggestion": "Run bounded per-model defense smoke before presenting recovery metrics as benchmark results.",
        "warnings": [
            "Do not present unmasked rank movement as masked TopK manipulation success.",
            "DP and secure aggregation panels are demo/boundary evidence, not production privacy guarantees.",
        ],
        "available_panels": [
            "scenario_profile",
            "runtime_timeline",
            "training_curves",
            "target_manipulation_metrics",
            "membership_inference_panel",
            "update_leakage_panel",
            "aggregation_defense_panel",
            "privacy_defense_panel",
            "model_support_panel",
        ],
    }


def main() -> int:
    args = parse_args()
    sources = source_dirs(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / args.scenario_id
    dataset_root = Path(args.dataset_root)
    metadata = metadata_index(dataset_root)
    image_manifest = read_json(Path(args.image_manifest))
    images = image_index(image_manifest)

    experiment_summary = read_json(find_file(sources, patterns=["*.experiment_summary.json"]))
    model, dataset, attacks, defenses = experiment_identity(experiment_summary)
    target_rank = read_json(find_file(sources, names=["target_rank_comparison.json"]))
    manipulation = read_json(find_file(sources, names=["recommendation_manipulation_summary.json"]))
    membership_summary = read_json(find_file(sources, names=["membership_score_summary.json"]))
    pair_scores = find_file(sources, names=["membership_pair_scores.csv"])
    interaction = read_json(find_file(sources, names=["interaction_reconstruction_summary.json"]))
    update_risk = read_json(find_file(sources, names=["update_leakage_risk_summary.json"]))
    defense_matrix = read_json(find_file(sources, names=["defense_matrix_summary.json"]))
    defense_trace = read_json(find_file(sources, names=["defense_trace.json"]))
    attack_defense = read_json(find_file(sources, names=["attack_defense_summary.json"]))
    opacus = read_json(find_file(sources, names=["opacus_toy_summary.json", "opacus_toy_v25_summary.json"]))
    secure_agg = read_json(find_file(sources, names=["secure_aggregation_demo_summary.json", "secure_aggregation_demo_v25_summary.json"]))
    model_matrix = read_json(Path(args.model_matrix))
    round_csv = find_round_csv(sources)

    exported_at = utc_now()
    scenario_profile = {
        "summary_type": "security_artifact_v3_scenario_profile",
        "scenario_id": args.scenario_id,
        "display_name": args.display_name,
        "dataset": dataset,
        "model": model,
        "attack_type": attacks,
        "defense_type": defenses or None,
        "evidence_tags": [
            "real_training_summary",
            "target_rank_comparison",
            "rank_or_unmasked_rank_mia",
            "real_participant_params",
            "config_only_defense_matrix",
            "secure_aggregation_demo",
        ],
        "supported_frontend_directions": [
            "recommendation_manipulation",
            "membership_inference",
            "update_leakage",
            "aggregation_defense",
        ],
        "artifact_source": [repo_relative(path) for path in sources],
        "created_at": experiment_summary.get("output_run_id"),
        "exported_at": exported_at,
        "limitations": [
            "Masked TopK exposure is separate from unmasked target rank movement.",
            "MIA may use rank/unmasked-rank proxy evidence when checkpoint_score is unavailable.",
            "Interaction reconstruction is candidate reconstruction, not full user history.",
            "Defense matrix cases may be config-only and do not imply benchmark recovery.",
            "dp_noise is not formal DP and secure aggregation is simulation/demo only.",
        ],
    }

    training_curves = build_training_curves(round_csv, manipulation, attack_defense)
    target_metrics = build_target_metrics(target_rank, manipulation, metadata, images)
    membership_panel = build_membership_panel(membership_summary, pair_scores)
    update_panel = build_update_leakage_panel(interaction, update_risk, metadata, images)
    aggregation_panel = build_aggregation_defense_panel(defense_trace, defense_matrix, attack_defense)
    privacy_panel = build_privacy_defense_panel(opacus, secure_agg, defense_trace, defense_matrix)
    model_panel = build_model_support_panel(model_matrix)
    runtime_timeline = build_runtime_timeline()
    frontend_summary = build_frontend_summary(
        target_metrics,
        membership_panel,
        update_panel,
        aggregation_panel,
        privacy_panel,
        model_panel,
    )

    payloads = {
        "scenario_profile.json": scenario_profile,
        "runtime_timeline.json": runtime_timeline,
        "training_curves.json": training_curves,
        "target_manipulation_metrics.json": target_metrics,
        "membership_inference_panel.json": membership_panel,
        "update_leakage_panel.json": update_panel,
        "aggregation_defense_panel.json": aggregation_panel,
        "privacy_defense_panel.json": privacy_panel,
        "model_support_panel.json": model_panel,
        "frontend_summary.json": frontend_summary,
    }
    for name, payload in payloads.items():
        write_json(output_dir / name, payload)

    print(
        json.dumps(
            {
                "output_dir": repo_relative(output_dir),
                "file_count": len(payloads),
                "files": sorted(payloads),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
