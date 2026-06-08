from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_eval.model_security_adapter import (  # noqa: E402
    VALID_STATUSES,
    describe_security_capability,
    resolve_model_name,
)

CAPABILITY_PATH = ROOT / "configs" / "model_attack_defense_capabilities.json"

DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "model_security_capability_matrix"
    / "model_security_capability_matrix.json"
)
MODEL_MATRIX_V2_CONFIG_DIR = ROOT / "configs" / "experiment_smoke" / "model_matrix_v2"
SMOKE_STATUS_VALUES = ["passed", "failed", "not_run", "adapter_required"]

TARGET_MODELS: List[Tuple[str, str]] = [
    ("FedAvg", "AMAZON_BEAUTY_POC"),
    ("FedAvg", "KU"),
    ("FedRAP", "KU"),
    ("FedRAP", "AMAZON_BEAUTY_POC"),
    ("MMFedRAP", "KU"),
    ("MMFedAvg", "KU"),
]

TARGET_MODELS_V2: List[Tuple[str, str]] = [
    ("FedAvg", "AMAZON_BEAUTY_POC"),
    ("FedAvg", "KU"),
    ("FedRAP", "KU"),
    ("FedNCF", "KU"),
    ("FCF", "KU"),
    ("MGCN", "KU"),
    ("MMFedAvg", "KU"),
    ("MMFedRAP", "KU"),
    ("MMFedNCF", "KU"),
    ("MMFCF", "KU"),
    ("MMGCN", "KU"),
    ("MMMGCN", "KU"),
    ("MultiModalMGCN", "KU"),
]

CAPABILITIES = [
    "baseline_training",
    "topk_export",
    "target_rank_summary",
    "target_interaction_injection",
    "target_promotion_loss",
    "membership_score_unmasked_rank_mia",
    "interaction_reconstruction",
    "update_leakage_risk",
    "robust_aggregation",
    "dp_noise",
    "secure_aggregation_sim",
    "checkpoint_scorer",
]

CAPABILITIES_V2 = [
    "baseline_training",
    "dataset_support",
    "topk_export",
    "participant_update_capture",
    "item_embedding_access",
    "score_user_item_pairs",
    "target_rank_summary",
    "target_interaction_injection",
    "membership_inference_rank_proxy",
    "membership_inference_checkpoint_score",
    "update_leakage_probe",
    "interaction_reconstruction_probe",
    "robust_aggregation",
    "dp_noise",
    "secure_aggregation_sim",
    "v3_artifact_export",
]

KNOWN_ARTIFACTS = {
    "fedavg_amazon_v25_result": ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "AMAZON_BEAUTY_POC"
    / "AmazonBeautyPOCTargetPromotionV25Smoke",
    "fedavg_amazon_ir_v25": ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "AMAZON_BEAUTY_POC"
    / "AmazonBeautyPOCInteractionReconstructionV25Smoke"
    / "interaction_reconstruction_summary.json",
    "fedavg_ku_ir": ROOT
    / "outputs"
    / "results"
    / "FedAvg"
    / "KU"
    / "SecurityMatrixSmoke"
    / "interaction_reconstruction_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conservative multi-model FedVLR security capability matrix."
    )
    parser.add_argument(
        "--capabilities",
        default=str(CAPABILITY_PATH),
        help="Path to model_attack_defense_capabilities.json.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT),
        help="Where to write model_security_capability_matrix.json.",
    )
    parser.add_argument(
        "--matrix-version",
        choices=["v1", "v2"],
        default="v1",
        help="v1 keeps the original matrix schema; v2 emits per-model security adapter capabilities.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def newest(paths: Iterable[Path]) -> Optional[Path]:
    candidates = [path for path in paths if path.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def has_nonfinite_scalar(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return True
    return False


def has_nonfinite_json_value(value: Any) -> bool:
    if has_nonfinite_scalar(value):
        return True
    if isinstance(value, dict):
        return any(has_nonfinite_json_value(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite_json_value(item) for item in value)
    return False


def json_is_readable_and_finite(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return not has_nonfinite_json_value(payload)


def csv_is_readable_and_finite(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for value in row.values():
                    if has_nonfinite_scalar(value):
                        return False
    except Exception:
        return False
    return True


def topk_manifest_verified(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    file_count = payload.get("file_count")
    topk_files = payload.get("topk_files")
    if isinstance(file_count, int) and file_count > 0:
        return True
    if isinstance(topk_files, list) and topk_files:
        return True
    return bool(list(path.parent.glob("*.csv")))


def load_model_matrix_v2_smoke_configs() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    if not MODEL_MATRIX_V2_CONFIG_DIR.exists():
        return configs
    for path in sorted(MODEL_MATRIX_V2_CONFIG_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        model = str(payload.get("model", "")).strip()
        dataset = str(payload.get("dataset", "")).strip()
        run_type = str(payload.get("type", "")).strip()
        comment = str(payload.get("comment", "")).strip()
        if not model or not dataset or not run_type or not comment:
            continue
        key = "{}::{}".format(model, dataset)
        configs[key] = {
            "config_path": path,
            "model": model,
            "dataset": dataset,
            "type": run_type,
            "comment": comment,
        }
    return configs


def inspect_model_matrix_v2_smoke(config: Dict[str, Any], default_status: str) -> Dict[str, Any]:
    if default_status == "adapter_required":
        return {
            "smoke_status": "adapter_required",
            "smoke_result_dir": None,
            "topk_export_verified": False,
            "metrics_export_verified": False,
            "security_artifact_ready": False,
            "failure_reason": "core model status is adapter_required",
        }

    result_dir = ROOT / "outputs" / "results" / config["model"] / config["dataset"] / config["type"]
    prefix = "[{}]-[{}]-[{}.{}]-[".format(
        config["model"],
        config["dataset"],
        config["type"],
        config["comment"],
    )
    csv_path = newest(path for path in result_dir.glob("*.csv") if path.name.startswith(prefix)) if result_dir.exists() else None
    if csv_path is None:
        return {
            "smoke_status": "not_run",
            "smoke_result_dir": rel(result_dir),
            "smoke_config_path": rel(config["config_path"]),
            "topk_export_verified": False,
            "metrics_export_verified": False,
            "security_artifact_ready": False,
            "failure_reason": None,
        }

    base_name = csv_path.stem
    summary_path = csv_path.with_name("{}.experiment_summary.json".format(base_name))
    result_path = csv_path.with_name("{}.experiment_result.json".format(base_name))
    topk_dir = result_dir / "recommend_topk" / base_name
    topk_manifest = topk_dir / "recommend_topk_manifest.json"
    metrics_ok = (
        csv_is_readable_and_finite(csv_path)
        and json_is_readable_and_finite(summary_path)
        and json_is_readable_and_finite(result_path)
    )
    topk_ok = topk_manifest_verified(topk_manifest)
    if metrics_ok and topk_ok:
        smoke_status = "passed"
        failure_reason = None
    else:
        smoke_status = "failed"
        missing: List[str] = []
        if not metrics_ok:
            missing.append("metrics_csv_or_experiment_json_missing_unreadable_or_nonfinite")
        if not topk_ok:
            missing.append("topk_manifest_missing_or_empty")
        failure_reason = ";".join(missing)
    return {
        "smoke_status": smoke_status,
        "smoke_result_dir": rel(result_dir),
        "smoke_config_path": rel(config["config_path"]),
        "smoke_csv_path": rel(csv_path),
        "smoke_summary_path": rel(summary_path) if summary_path.exists() else None,
        "smoke_result_path": rel(result_path) if result_path.exists() else None,
        "recommend_topk_dir": rel(topk_dir) if topk_dir.exists() else None,
        "topk_manifest_path": rel(topk_manifest) if topk_manifest.exists() else None,
        "topk_export_verified": topk_ok,
        "metrics_export_verified": metrics_ok,
        "security_artifact_ready": bool(metrics_ok and topk_ok),
        "failure_reason": failure_reason,
    }


def verification_level_for(overall_status: str, smoke_status: str) -> str:
    if smoke_status == "passed":
        return "partial_smoke_verified" if overall_status == "partial" else "smoke_verified"
    if smoke_status == "failed":
        return "failed_smoke"
    if smoke_status == "adapter_required" or overall_status == "adapter_required":
        return "adapter_required"
    return "partial_validate" if overall_status == "partial" else "validate_only"


def model_records(capabilities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in capabilities.get("models", [])
        if item.get("name")
    }


def dataset_supported(record: Dict[str, Any], dataset: str) -> bool:
    supported = record.get("supported_datasets") or []
    return not supported or dataset in supported


def artifact_evidence(path: Path, label: str) -> str:
    if path.exists():
        return "{}: {}".format(label, rel(path))
    return "{} not found at {}".format(label, rel(path))


def fedavg_amazon_v25_evidence(capability: str) -> Optional[str]:
    result_dir = KNOWN_ARTIFACTS["fedavg_amazon_v25_result"]
    mapping = {
        "topk_export": result_dir / "recommend_topk" / "recommend_topk_manifest.json",
        "target_rank_summary": result_dir / "target_rank_comparison.json",
        "target_interaction_injection": result_dir / "target_interaction_plan.json",
        "membership_score_unmasked_rank_mia": result_dir / "membership_score_summary.json",
        "secure_aggregation_sim": result_dir / "secure_aggregation_demo_summary.json",
    }
    path = mapping.get(capability)
    if path:
        return artifact_evidence(path, "FedAvg Amazon V2.5 evidence")
    return None


def status_for(
    model: str,
    dataset: str,
    capability: str,
    records: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    record = records.get(model, {})
    if not record:
        return {
            "status": "unsupported",
            "evidence": "model is absent from configs/model_attack_defense_capabilities.json",
            "reason": "unknown model",
            "recommended_demo_usage": "do not expose as selectable until the model is registered",
        }

    if not dataset_supported(record, dataset):
        return {
            "status": "unsupported",
            "evidence": "supported_datasets={}".format(record.get("supported_datasets", [])),
            "reason": "{} is not listed for {} in the capability matrix".format(
                dataset, model
            ),
            "recommended_demo_usage": "show as unsupported dataset/model pairing",
        }

    compatibility = str(record.get("compatibility_status", "unknown"))
    if compatibility == "blocked":
        return {
            "status": "unsupported",
            "evidence": str(record.get("notes", "blocked model")),
            "reason": "model is blocked in the capability registry",
            "recommended_demo_usage": "hide from security demos",
        }

    evidence = fedavg_amazon_v25_evidence(capability)
    if model == "FedAvg" and dataset == "AMAZON_BEAUTY_POC" and evidence:
        if capability == "target_rank_summary":
            return {
                "status": "supported",
                "evidence": evidence + "; target item 0 moved from rank 170 to 3 unmasked",
                "reason": "real V2.5 Amazon smoke produced target_rank_comparison.json",
                "recommended_demo_usage": "primary target-rank demonstration; keep masked TopK hit caveat visible",
            }
        if capability == "target_interaction_injection":
            return {
                "status": "supported",
                "evidence": evidence,
                "reason": "hook_active target interaction plan was emitted in the real V2.5 run",
                "recommended_demo_usage": "primary target interaction injection smoke",
            }
        if capability == "membership_score_unmasked_rank_mia":
            return {
                "status": "partial",
                "evidence": evidence,
                "reason": "mixed unmasked-rank/rank-proxy evidence exists, but checkpoint_score was not available",
                "recommended_demo_usage": "show as score/rank MIA with proxy boundary",
            }
        if capability == "topk_export":
            return {
                "status": "supported",
                "evidence": evidence,
                "reason": "manifest-backed TopK export prevents per-user file overwrite",
                "recommended_demo_usage": "show as reusable observation infrastructure",
            }
        if capability == "secure_aggregation_sim":
            return {
                "status": "partial",
                "evidence": evidence,
                "reason": "synthetic pairwise-mask cancellation demo only",
                "recommended_demo_usage": "show as demo-only secure aggregation boundary",
            }

    if capability == "baseline_training":
        return {
            "status": "supported",
            "evidence": "model registry compatibility_status={}".format(compatibility),
            "reason": "baseline scenario is in validated_combinations for registered models",
            "recommended_demo_usage": "allow validate-only and bounded smoke runs",
        }

    if capability == "topk_export":
        return {
            "status": "supported",
            "evidence": "utils/topk_evaluator.py writes per-user filenames and recommend_topk_manifest.json",
            "reason": "TopK export sits in the shared evaluator path",
            "recommended_demo_usage": "safe to expose as model-agnostic export when save_recommended_topk=true",
        }

    if capability == "target_rank_summary":
        return {
            "status": "partial",
            "evidence": "utils/federated/trainer.py records unmasked/masked target rank when target_item_ids are configured",
            "reason": "diagnostic infrastructure is shared, but target movement is only validated on FedAvg Amazon",
            "recommended_demo_usage": "show as diagnostic; do not claim model-wide target promotion success",
        }

    if capability == "target_interaction_injection":
        return {
            "status": "partial",
            "evidence": "attacks/target_interaction_injection.py modifies only in-memory malicious-client loaders",
            "reason": "hook is model-agnostic, but effectiveness is model/dataset-specific",
            "recommended_demo_usage": "enable only in smoke configs and label as experimental",
        }

    if capability == "target_promotion_loss":
        return {
            "status": "future_adapter",
            "evidence": "target_interaction_injection summary marks target_promotion_loss status=feasibility_only",
            "reason": "no stable model-specific local target-score loss hook is implemented",
            "recommended_demo_usage": "show as future adapter, not an implemented attack",
        }

    if capability == "membership_score_unmasked_rank_mia":
        if model in {"FedAvg", "FedRAP"}:
            return {
                "status": "partial",
                "evidence": "privacy_eval/export_membership_pair_scores.py supports checkpoint_score for FedAvg/FedRAP-style checkpoints, then unmasked_rank, then rank_proxy",
                "reason": "score/rank export exists, but checkpoint availability depends on saved parameter format",
                "recommended_demo_usage": "show with score_source and proxy_only fields",
            }
        return {
            "status": "partial",
            "evidence": "unmasked_rank and rank_proxy paths are model-independent; full checkpoint scorer needs adapter",
            "reason": "MM model checkpoint_score path is a future adapter",
            "recommended_demo_usage": "show rank-based MIA only unless a model scorer is added",
        }

    if capability == "interaction_reconstruction":
        if model == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return {
                "status": "supported",
                "evidence": artifact_evidence(
                    KNOWN_ARTIFACTS["fedavg_amazon_ir_v25"],
                    "FedAvg Amazon interaction reconstruction V2.5",
                )
                + "; hit@10/20/50 = 0.20/0.30/0.36 when available",
                "reason": "real participant_params smoke produced per-client candidates",
                "recommended_demo_usage": "show as candidate reconstruction only, not full history recovery",
            }
        if model == "FedAvg" and dataset == "KU":
            return {
                "status": "partial",
                "evidence": artifact_evidence(
                    KNOWN_ARTIFACTS["fedavg_ku_ir"], "FedAvg KU interaction reconstruction"
                ),
                "reason": "candidate reconstruction probe exists; V2.5 headline result is Amazon FedAvg",
                "recommended_demo_usage": "show as probe when participant_params are present",
            }
        return {
            "status": "partial",
            "evidence": "privacy_eval/interaction_reconstruction_probe.py reads real participant_params and item-like updates",
            "reason": "participant_params are shared, but per-model item-like parameter naming may need validation",
            "recommended_demo_usage": "validate before front-end effect claims",
        }

    if capability == "update_leakage_risk":
        return {
            "status": "partial",
            "evidence": "privacy_eval/update_leakage_risk_probe.py summarizes real participant_params without saving raw updates",
            "reason": "shared hook can observe updates, but risk magnitude is run-specific",
            "recommended_demo_usage": "show as update-risk summary, not reconstruction",
        }

    if capability == "robust_aggregation":
        if model in {"FedRAP", "MMFedRAP"}:
            return {
                "status": "supported",
                "evidence": "registry lists robust_defense and model has validated trimmed_mean-style defenses",
                "reason": "robust aggregation is the recommended defense track for RAP-style showcase models",
                "recommended_demo_usage": "safe for validate-only and bounded KU smoke matrix",
            }
        return {
            "status": "partial",
            "evidence": "defenses/robust_defense.py, trimmed_mean, median, krum, multi_krum, bulyan are available",
            "reason": "infrastructure is shared, but per-model target-promotion defense effect is not fully validated",
            "recommended_demo_usage": "show config-level support unless a matching result exists",
        }

    if capability == "dp_noise":
        return {
            "status": "partial",
            "evidence": "defenses/dp_noise_defense.py is central DP-style clipping plus Gaussian noise; Opacus toy remains standalone",
            "reason": "formal_accountant=false for FedVLR training",
            "recommended_demo_usage": "label as DP-style, not formal DP",
        }

    if capability == "secure_aggregation_sim":
        return {
            "status": "partial",
            "evidence": "defenses/secure_aggregation_sim.py and privacy_eval/run_secure_aggregation_demo.py are simulation/demo paths",
            "reason": "not a production cryptographic protocol and not equivalent to robust filtering",
            "recommended_demo_usage": "show as simulation-only boundary",
        }

    if capability == "checkpoint_scorer":
        if model in {"FedAvg", "FedRAP"}:
            return {
                "status": "partial",
                "evidence": "privacy_eval/export_membership_pair_scores.py can score FedAvg/FedRAP-style parameter pickles when item_commonality and client_models exist",
                "reason": "checkpoint_score depends on saved checkpoint format; unsupported checkpoints are not guessed",
                "recommended_demo_usage": "show as available when checkpoint_available=true",
            }
        return {
            "status": "future_adapter",
            "evidence": "full model reconstruction adapter is required for multimodal checkpoints",
            "reason": "MMFedRAP/MMFedAvg checkpoint scoring is not wired through Config + get_model + load_state_dict",
            "recommended_demo_usage": "show as future adapter; use unmasked_rank/rank_proxy meanwhile",
        }

    return {
        "status": "not_tested",
        "evidence": "no rule",
        "reason": "capability rule missing",
        "recommended_demo_usage": "hide until reviewed",
    }


def build_matrix(capabilities: Dict[str, Any]) -> Dict[str, Any]:
    records = model_records(capabilities)
    entries: List[Dict[str, Any]] = []
    for model, dataset in TARGET_MODELS:
        for capability in CAPABILITIES:
            result = status_for(model, dataset, capability, records)
            entries.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "capability": capability,
                    "status": result["status"],
                    "evidence": result["evidence"],
                    "reason": result["reason"],
                    "recommended_demo_usage": result["recommended_demo_usage"],
                }
            )

    counts: Dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1

    supported_demos = [
        {
            "model": item["model"],
            "dataset": item["dataset"],
            "capability": item["capability"],
            "status": item["status"],
            "recommended_demo_usage": item["recommended_demo_usage"],
        }
        for item in entries
        if item["status"] in {"supported", "partial"}
    ]
    unsupported_reasons = [
        {
            "model": item["model"],
            "dataset": item["dataset"],
            "capability": item["capability"],
            "status": item["status"],
            "reason": item["reason"],
        }
        for item in entries
        if item["status"] in {"unsupported", "future_adapter", "not_tested"}
    ]

    matrix_by_model: Dict[str, Dict[str, Dict[str, str]]] = {}
    for item in entries:
        key = "{}::{}".format(item["model"], item["dataset"])
        matrix_by_model.setdefault(key, {})[item["capability"]] = {
            "status": item["status"],
            "reason": item["reason"],
        }

    return {
        "summary_type": "model_security_capability_matrix",
        "generated_at": utc_now(),
        "scope": {
            "repository": "FedVLR",
            "no_api_or_frontend_change": True,
            "ordinary_outputs_not_for_git": True,
        },
        "models": [{"model": model, "dataset": dataset} for model, dataset in TARGET_MODELS],
        "capabilities": CAPABILITIES,
        "status_counts": counts,
        "entries": entries,
        "matrix_by_model": matrix_by_model,
        "supported_demos": supported_demos,
        "unsupported_reasons": unsupported_reasons,
        "recommended_frontend_labels": {
            "supported": "validated / displayable with evidence",
            "partial": "partial support / display with boundary note",
            "unsupported": "unsupported pairing",
            "future_adapter": "future adapter required",
            "not_tested": "not tested",
            "model_note": "MMFedRAP is the multimodal showcase model; FedAvg is the strongest attack/defense validation base.",
            "target_promotion_note": "FedAvg Amazon rank movement must not be generalized to every model, and masked TopK exposure may remain zero.",
        },
        "warnings": [
            "target_promotion_loss is feasibility-only until a model-specific local score loss hook is validated",
            "dp_noise is central DP-style noise without a formal accountant",
            "secure_aggregation_sim is simulation/demo only",
            "rank-only MIA evidence is proxy evidence",
        ],
    }


def model_inventory(capabilities: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = model_records(capabilities)
    inventory: List[Dict[str, Any]] = []
    for model, _dataset in TARGET_MODELS_V2:
        resolved = resolve_model_name(model)
        canonical = resolved["canonical_model"]
        record = records.get(canonical, {})
        baseline = describe_security_capability(model, _dataset, "baseline_training", capabilities)
        inventory.append(
            {
                "requested_model": model,
                "canonical_model": canonical,
                "is_alias": resolved["is_alias"],
                "registered": bool(record),
                "compatibility_status": record.get("compatibility_status"),
                "security_status": record.get("security_status"),
                "model_file": "models/{}.py".format(canonical.lower()),
                "baseline_status": baseline["status"],
                "baseline_reason": baseline["reason"],
            }
        )
    return inventory


def build_matrix_v2(capabilities: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for model, dataset in TARGET_MODELS_V2:
        for capability in CAPABILITIES_V2:
            result = describe_security_capability(model, dataset, capability, capabilities)
            entries.append(
                {
                    "model": model,
                    "canonical_model": result["canonical_model"],
                    "dataset": dataset,
                    "capability": capability,
                    "status": result["status"],
                    "reason": result["reason"],
                    "evidence_file": result["evidence_file"],
                }
            )

    status_counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for entry in entries:
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1

    model_records_map: Dict[str, Dict[str, Any]] = {}
    model_dataset_support: Dict[str, Dict[str, Any]] = {}
    model_direction_support: Dict[str, Dict[str, str]] = {}
    attack_defense_support_by_model: Dict[str, Dict[str, Any]] = {}
    by_status: Dict[str, set[str]] = {
        "supported": set(),
        "partial": set(),
        "adapter_required": set(),
        "unsupported": set(),
        "not_tested": set(),
    }

    for entry in entries:
        model_key = "{}::{}".format(entry["model"], entry["dataset"])
        model_records_map.setdefault(
            model_key,
            {
                "model": entry["model"],
                "canonical_model": entry["canonical_model"],
                "dataset": entry["dataset"],
                "capabilities": {},
            },
        )
        model_records_map[model_key]["capabilities"][entry["capability"]] = {
            "status": entry["status"],
            "reason": entry["reason"],
            "evidence_file": entry["evidence_file"],
        }
        by_status.setdefault(entry["status"], set()).add(model_key)
        model_direction_support.setdefault(entry["capability"], {})[model_key] = entry["status"]

        if entry["capability"] == "dataset_support":
            model_dataset_support[model_key] = {
                "status": entry["status"],
                "reason": entry["reason"],
                "evidence_file": entry["evidence_file"],
            }
        if entry["capability"] in {
            "target_interaction_injection",
            "robust_aggregation",
            "dp_noise",
            "secure_aggregation_sim",
        }:
            attack_defense_support_by_model.setdefault(model_key, {})[entry["capability"]] = {
                "status": entry["status"],
                "reason": entry["reason"],
            }

    models_by_capability_status = {status: sorted(values) for status, values in by_status.items()}
    overall_model_status: Dict[str, Dict[str, Any]] = {}
    for model_key, record in model_records_map.items():
        caps = record["capabilities"]
        core_statuses = [
            caps.get("baseline_training", {}).get("status"),
            caps.get("dataset_support", {}).get("status"),
            caps.get("topk_export", {}).get("status"),
            caps.get("participant_update_capture", {}).get("status"),
        ]
        if "unsupported" in core_statuses:
            overall = "unsupported"
            reason = "dataset or core launcher support is unsupported"
        elif "adapter_required" in core_statuses:
            overall = "adapter_required"
            reason = "core model/trainer/update capture path needs an adapter"
        elif "partial" in core_statuses:
            overall = "partial"
            reason = "core path exists but registry validation is partial/not_validated"
        else:
            overall = "supported"
            reason = "baseline, dataset, TopK, and participant update capture paths are supported"
        overall_model_status[model_key] = {
            "status": overall,
            "reason": reason,
            "model": record["model"],
            "canonical_model": record["canonical_model"],
            "dataset": record["dataset"],
        }

    overall_groups: Dict[str, List[str]] = {
        "supported": [],
        "partial": [],
        "adapter_required": [],
        "unsupported": [],
        "not_tested": [],
    }
    for model_key, payload in overall_model_status.items():
        overall_groups.setdefault(payload["status"], []).append(model_key)
    overall_groups = {key: sorted(value) for key, value in overall_groups.items()}

    smoke_configs = load_model_matrix_v2_smoke_configs()
    model_smoke_evidence: Dict[str, Dict[str, Any]] = {}
    smoke_verified_models: List[str] = []
    partial_smoke_verified_models: List[str] = []
    validate_only_models: List[str] = []
    failed_smoke_models: List[str] = []
    for model_key, payload in overall_model_status.items():
        config = smoke_configs.get(model_key)
        if config:
            smoke = inspect_model_matrix_v2_smoke(config, payload["status"])
        elif payload["status"] == "adapter_required":
            smoke = {
                "smoke_status": "adapter_required",
                "smoke_result_dir": None,
                "topk_export_verified": False,
                "metrics_export_verified": False,
                "security_artifact_ready": False,
                "failure_reason": "no runnable smoke config because model requires an adapter",
            }
        else:
            smoke = {
                "smoke_status": "not_run",
                "smoke_result_dir": None,
                "topk_export_verified": False,
                "metrics_export_verified": False,
                "security_artifact_ready": False,
                "failure_reason": None,
            }
        smoke["verification_level"] = verification_level_for(payload["status"], smoke["smoke_status"])
        payload.update(smoke)
        if model_key in model_records_map:
            model_records_map[model_key].update(smoke)
        model_smoke_evidence[model_key] = smoke

        if smoke["smoke_status"] == "passed":
            smoke_verified_models.append(model_key)
            if payload["status"] == "partial":
                partial_smoke_verified_models.append(model_key)
        elif smoke["smoke_status"] == "failed":
            failed_smoke_models.append(model_key)
        elif smoke["smoke_status"] == "not_run" and payload["status"] != "adapter_required":
            validate_only_models.append(model_key)

    return {
        "summary_type": "model_security_capability_matrix_v2",
        "matrix_version": "v2",
        "generated_at": utc_now(),
        "scope": {
            "repository": "FedVLR",
            "no_api_or_frontend_change": True,
            "ordinary_outputs_not_for_git": True,
            "does_not_run_training": True,
        },
        "status_values": sorted(VALID_STATUSES),
        "models": [{"model": model, "dataset": dataset} for model, dataset in TARGET_MODELS_V2],
        "model_inventory": model_inventory(capabilities),
        "capabilities": CAPABILITIES_V2,
        "smoke_status_values": SMOKE_STATUS_VALUES,
        "status_counts": status_counts,
        "entries": entries,
        "model_records": list(model_records_map.values()),
        "overall_model_status": overall_model_status,
        "model_smoke_evidence": model_smoke_evidence,
        "smoke_verified_models": sorted(smoke_verified_models),
        "partial_smoke_verified_models": sorted(partial_smoke_verified_models),
        "validate_only_models": sorted(validate_only_models),
        "failed_smoke_models": sorted(failed_smoke_models),
        "models_by_capability_status": models_by_capability_status,
        "supported_models": overall_groups["supported"],
        "partial_models": overall_groups["partial"],
        "adapter_required_models": overall_groups["adapter_required"],
        "unsupported_models": overall_groups["unsupported"],
        "not_tested_models": overall_groups["not_tested"],
        "recommended_showcase_models": {
            "security_validation_base": "FedAvg::AMAZON_BEAUTY_POC",
            "multimodal_showcase": "MMFedRAP::KU",
            "extension_targets": ["FedRAP::KU", "FedNCF::KU", "FCF::KU", "MMFedNCF::KU", "MMFCF::KU"],
        },
        "model_direction_support": model_direction_support,
        "model_dataset_support": model_dataset_support,
        "attack_defense_support_by_model": attack_defense_support_by_model,
        "warnings": [
            "FedAvg Amazon target-rank movement must not be generalized to other models.",
            "adapter_required means the model exists only partially, lacks required trainer/dependencies, or needs a scorer/security hook adapter.",
            "partial means the shared path exists but model-specific security evidence is not complete.",
            "secure_aggregation_sim remains simulation/demo only.",
        ],
    }


def main() -> int:
    args = parse_args()
    capabilities = load_json(Path(args.capabilities))
    matrix = build_matrix_v2(capabilities) if args.matrix_version == "v2" else build_matrix(capabilities)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": rel(output_path), "status_counts": matrix["status_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
