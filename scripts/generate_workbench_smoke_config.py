from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "workbench_experiment_schema.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "workbench_jobs"
AMAZON_DATASET = "AMAZON_BEAUTY_POC"

DIRECTION_ALIASES = {
    "target_poisoning_play": "recommendation_manipulation",
    "recommendation_manipulation": "recommendation_manipulation",
    "membership_privacy_play": "membership_inference",
    "membership_inference": "membership_inference",
    "update_leakage_play": "update_leakage",
    "update_leakage": "update_leakage",
    "robust_defense_play": "aggregation_defense",
    "aggregation_defense": "aggregation_defense",
}

ATTACKS_BY_DIRECTION = {
    "recommendation_manipulation": ["poisoning_attack"],
    "membership_inference": [],
    "update_leakage": [],
    "aggregation_defense": ["poisoning_attack"],
}

PRIVACY_BY_DIRECTION = {
    "recommendation_manipulation": ["membership_inference", "interaction_reconstruction"],
    "membership_inference": ["membership_inference"],
    "update_leakage": ["interaction_reconstruction"],
    "aggregation_defense": [],
}


def load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def short_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"workbench_{stamp}_{uuid.uuid4().hex[:8]}"


def safe_job_id(value: str | None) -> str:
    if not value:
        return short_job_id()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise ValueError("invalid_job_id")
    return value


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def schema() -> Dict[str, Any]:
    return load_json(SCHEMA_PATH, {})


def as_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return parsed


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    if isinstance(value, (int, float)):
        return value != 0
    return default


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return []


def clamp_number(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def get_target_item_options(limit: int = 60) -> List[Dict[str, Any]]:
    metadata = load_json(ROOT / "datasets" / AMAZON_DATASET / "item_metadata.json", {})
    sidecar = load_json(ROOT / "outputs" / "security_sidecars" / AMAZON_DATASET / "target_items.json", {})
    manifest = load_json(ROOT / "datasets" / AMAZON_DATASET / "item_image_manifest.json", {})

    items = metadata.get("items", []) if isinstance(metadata, dict) else []
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("itemID", item.get("item_id", ""))).strip()
        if item_id:
            by_id[item_id] = item

    image_by_id: Dict[str, Dict[str, Any]] = {}
    for item in manifest.get("items", []) if isinstance(manifest, dict) else []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id", item.get("itemID", ""))).strip()
        if item_id:
            image_by_id[item_id] = item

    ordered_ids: List[str] = []
    for item_id in sidecar.get("target_items", []) if isinstance(sidecar, dict) else []:
        key = str(item_id).strip()
        if key and key not in ordered_ids:
            ordered_ids.append(key)
    for key in by_id:
        if len(ordered_ids) >= limit:
            break
        if key not in ordered_ids:
            ordered_ids.append(key)

    options: List[Dict[str, Any]] = []
    for item_id in ordered_ids[:limit]:
        meta = by_id.get(item_id, {})
        image = image_by_id.get(item_id, {})
        title = str(meta.get("title") or image.get("title") or f"item {item_id}")
        short_title = title if len(title) <= 72 else f"{title[:69]}..."
        options.append(
            {
                "item_id": item_id,
                "raw_item_id": meta.get("raw_item_id") or image.get("raw_item_id"),
                "title": title,
                "short_title": short_title,
                "category": meta.get("main_category") or image.get("category"),
                "image_url": meta.get("image_url") or image.get("image_url"),
                "is_target_sidecar": item_id in set(str(x) for x in (sidecar.get("target_items", []) if isinstance(sidecar, dict) else [])),
            }
        )
    return options


def get_workbench_options() -> Dict[str, Any]:
    data = schema()
    return {
        "schema_version": data.get("version"),
        "directions": data.get("directions", []),
        "datasets": data.get("datasets", []),
        "models": data.get("models", []),
        "adapter_required_models": data.get("adapter_required_models", []),
        "aggregation_visibility_modes": data.get("aggregation_visibility_modes", []),
        "robust_aggregators": data.get("robust_aggregators", []),
        "bounds": data.get("bounds", {}),
        "defaults": data.get("defaults", {}),
        "target_items": get_target_item_options(),
        "notes": [
            "Workbench jobs are bounded smoke/demo configs.",
            "This generator writes job artifacts only and does not start training.",
            "MGCN family models require adapters and are not launchable from this workbench.",
        ],
    }


def normalize_direction(value: Any) -> str:
    key = str(value or "recommendation_manipulation").strip()
    if key not in DIRECTION_ALIASES:
        raise ValueError(f"unknown_direction:{key}")
    return DIRECTION_ALIASES[key]


def normalize_workbench_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    data = schema()
    bounds = data.get("bounds", {})
    defaults = data.get("defaults", {})
    warnings: List[str] = []
    errors: List[str] = []

    direction = normalize_direction(payload.get("direction") or payload.get("play_id") or payload.get("playId"))
    direction_record = next((item for item in data.get("directions", []) if item.get("id") == direction), {})

    models = {item.get("id"): item for item in data.get("models", [])}
    adapter_models = {item.get("id"): item for item in data.get("adapter_required_models", [])}
    datasets = {item.get("id"): item for item in data.get("datasets", [])}

    model = str(payload.get("model") or direction_record.get("default_model") or "FedAvg")
    dataset = str(payload.get("dataset") or direction_record.get("default_dataset") or "AMAZON_BEAUTY_POC")
    if model in adapter_models:
        errors.append(f"adapter_required_model:{model}")
    if model not in models:
        errors.append(f"unknown_model:{model}")
    if dataset not in datasets:
        errors.append(f"unknown_dataset:{dataset}")
    if model in models and dataset not in models[model].get("datasets", []):
        warnings.append(f"dataset_not_smoke_verified_for_model:{model}:{dataset}")

    total_rounds = as_int(payload.get("total_rounds", payload.get("totalRounds")), bounds.get("default_total_rounds", 10))
    total_rounds = int(clamp_number(total_rounds, 1, bounds.get("max_total_rounds", 10)))
    epochs = as_int(payload.get("epochs", total_rounds), total_rounds)
    epochs = int(clamp_number(epochs, 1, bounds.get("max_epochs", 10)))
    local_epochs = as_int(payload.get("local_epochs", payload.get("localEpochs")), bounds.get("default_local_epochs", 5))
    local_epochs = int(clamp_number(local_epochs, 1, bounds.get("max_local_epochs", 5)))

    explicit_large_ratio = as_bool(payload.get("allow_large_client_ratio") or payload.get("explicit_large_client_ratio"), False)
    ratio_max = bounds.get("max_client_sampling_ratio_explicit" if explicit_large_ratio else "max_client_sampling_ratio_default", 0.2)
    client_ratio = as_float(payload.get("client_sampling_ratio", payload.get("clientSamplingRate")), bounds.get("default_client_sampling_ratio", 0.2))
    if client_ratio > ratio_max and not explicit_large_ratio:
        warnings.append("client_sampling_ratio_clamped_to_bounded_smoke_default")
    client_ratio = clamp_number(client_ratio, 0.01, ratio_max)

    malicious_ratio = as_float(payload.get("malicious_client_ratio", payload.get("poisoningRatio")), bounds.get("default_malicious_client_ratio", 0.2))
    malicious_ratio = clamp_number(
        malicious_ratio,
        bounds.get("min_malicious_client_ratio", 0.0),
        bounds.get("max_malicious_client_ratio", 0.6),
    )

    aggregation_mode = str(payload.get("aggregation_mode", payload.get("aggregationMode")) or "plain_updates")
    if aggregation_mode not in {"plain_updates", "secure_aggregation"}:
        errors.append(f"unknown_aggregation_mode:{aggregation_mode}")
    robust_aggregators = [item for item in as_list(payload.get("robust_aggregators", payload.get("robustAggregators"))) if item != "none"]
    allowed_robust = set(data.get("robust_aggregators", []))
    for item in robust_aggregators:
        if item not in allowed_robust:
            errors.append(f"unknown_robust_aggregator:{item}")
    if aggregation_mode == "secure_aggregation" and robust_aggregators:
        errors.append("secure_aggregation_conflicts_with_robust_aggregation")

    dp_noise_enabled = as_bool(payload.get("dp_noise_enabled", payload.get("dpNoiseEnabled")), False)
    target_item_id = str(payload.get("target_item_id", payload.get("targetItemId")) or "0")
    target_title = payload.get("target_item_title", payload.get("targetItemTitle"))
    target_options = get_target_item_options()
    target_record = next((item for item in target_options if str(item.get("item_id")) == target_item_id), None)
    if dataset == AMAZON_DATASET and not target_record:
        warnings.append(f"target_item_not_in_target_options:{target_item_id}")

    candidate_k = as_int(payload.get("candidate_k", payload.get("candidateK")), 50)
    if candidate_k not in set(bounds.get("candidate_k_values", [10, 20, 50])):
        warnings.append("candidate_k_clamped_to_50")
        candidate_k = 50

    enabled_defenses: List[str] = []
    if dp_noise_enabled:
        enabled_defenses.append("dp_noise")
    if aggregation_mode == "secure_aggregation":
        enabled_defenses.append("secure_aggregation_sim")
    if robust_aggregators:
        enabled_defenses.append("robust_aggregation")

    enabled_attacks = ATTACKS_BY_DIRECTION[direction]
    enabled_privacy = PRIVACY_BY_DIRECTION[direction]
    if direction == "membership_inference":
        scenario = "privacy_observation"
    elif direction == "update_leakage":
        scenario = "privacy_observation"
    elif enabled_attacks and enabled_defenses:
        scenario = "attack_and_defense"
    elif enabled_attacks:
        scenario = "attack_only"
    elif enabled_defenses:
        scenario = "defense_only"
    else:
        scenario = "baseline"

    normalized = {
        "direction": direction,
        "scenario_id": payload.get("scenario_id") or payload.get("scenarioId") or direction_record.get("default_scenario_id"),
        "model": model,
        "dataset": dataset,
        "aggregation_mode": aggregation_mode,
        "robust_aggregators": robust_aggregators,
        "dp_noise_enabled": dp_noise_enabled,
        "training": {
            "epochs": epochs,
            "total_rounds": total_rounds,
            "local_epochs": local_epochs,
            "client_sampling_ratio": client_ratio,
            "learning_rate": as_float(payload.get("learning_rate", payload.get("learningRate")), defaults.get("learning_rate", 0.001)),
            "weight_decay": as_float(payload.get("weight_decay", payload.get("weightDecay")), defaults.get("weight_decay", 0.0)),
            "gradient_clip": as_float(payload.get("gradient_clip", payload.get("gradientClip")), defaults.get("gradient_clip", 5.0)),
            "save_topk": as_bool(payload.get("save_topk", payload.get("saveTopK")), defaults.get("save_topk", True)),
            "export_artifact": as_bool(payload.get("export_artifact", payload.get("exportArtifact")), defaults.get("export_artifact", True)),
        },
        "attack": {
            "malicious_client_ratio": malicious_ratio,
            "attack_strength": str(payload.get("attack_strength", payload.get("attackStrength")) or defaults.get("attack_strength", "strong")),
            "target_item_id": target_item_id,
            "target_item_title": target_title or (target_record or {}).get("title"),
            "injection_ratio": as_float(payload.get("injection_ratio", payload.get("injectionRatio")), defaults.get("injection_ratio", 0.2)),
            "max_injections_per_client": as_int(payload.get("max_injections_per_client", payload.get("maxInjectionsPerClient")), defaults.get("max_injections_per_client", 10)),
        },
        "privacy": {
            "mia_evidence_source": str(payload.get("mia_evidence_source", payload.get("evidenceSource")) or "auto"),
            "membership_sample_count": int(
                clamp_number(
                    as_int(payload.get("membership_sample_count", payload.get("membershipSampleCount")), bounds.get("default_membership_sample_count", 200)),
                    1,
                    bounds.get("max_membership_sample_count", 5000),
                )
            ),
            "candidate_k": candidate_k,
            "risk_modality": str(payload.get("risk_modality", payload.get("riskModality")) or "item embedding"),
        },
        "defense": {
            "trim_ratio": as_float(payload.get("trim_ratio", payload.get("trimRatio")), defaults.get("trim_ratio", 0.2)),
            "krum_f": as_int(payload.get("krum_f", payload.get("krumF")), defaults.get("krum_f", 1)),
            "bulyan_f": as_int(payload.get("bulyan_f", payload.get("bulyanF")), defaults.get("bulyan_f", 1)),
            "dp_noise_std": as_float(payload.get("dp_noise_std", payload.get("dpNoiseStd")), defaults.get("dp_noise_std", 0.15)),
        },
        "unified_experiment_config": {
            "model": model,
            "dataset": dataset,
            "scenario": scenario,
            "type": "WorkbenchSmoke",
            "comment": f"workbench:{direction}",
            "enabled_attacks": enabled_attacks,
            "enabled_defenses": enabled_defenses,
            "enabled_privacy_metrics": enabled_privacy,
            "malicious_client_config": {
                "enabled": malicious_ratio > 0,
                "mode": "ratio" if malicious_ratio > 0 else "none",
                "ratio": malicious_ratio,
                "client_ids": [],
            },
            "training_params": {
                "epochs": epochs,
                "total_rounds": total_rounds,
                "local_epochs": local_epochs,
                "client_sampling_ratio": client_ratio,
                "lr": as_float(payload.get("learning_rate", payload.get("learningRate")), defaults.get("learning_rate", 0.001)),
                "learning_rate": as_float(payload.get("learning_rate", payload.get("learningRate")), defaults.get("learning_rate", 0.001)),
                "weight_decay": as_float(payload.get("weight_decay", payload.get("weightDecay")), defaults.get("weight_decay", 0.0)),
                "gradient_clip": as_float(payload.get("gradient_clip", payload.get("gradientClip")), defaults.get("gradient_clip", 5.0)),
                "save_recommended_topk": as_bool(payload.get("save_topk", payload.get("saveTopK")), defaults.get("save_topk", True)),
                "recommendation_topk": 50,
                "export_security_artifact": as_bool(payload.get("export_artifact", payload.get("exportArtifact")), defaults.get("export_artifact", True)),
            },
            "attack_params": {
                "target_interaction_injection": {
                    "enabled": direction == "recommendation_manipulation",
                    "target_item_ids": [target_item_id],
                    "target_item_title": target_title or (target_record or {}).get("title"),
                    "malicious_client_ratio": malicious_ratio,
                    "injection_ratio": as_float(payload.get("injection_ratio", payload.get("injectionRatio")), defaults.get("injection_ratio", 0.2)),
                    "max_injections_per_client": as_int(payload.get("max_injections_per_client", payload.get("maxInjectionsPerClient")), defaults.get("max_injections_per_client", 10)),
                    "planner_only": False,
                }
            },
            "defense_params": {
                "robust_aggregation": {
                    "enabled": bool(robust_aggregators),
                    "algorithms": robust_aggregators,
                    "trim_ratio": as_float(payload.get("trim_ratio", payload.get("trimRatio")), defaults.get("trim_ratio", 0.2)),
                    "krum_f": as_int(payload.get("krum_f", payload.get("krumF")), defaults.get("krum_f", 1)),
                    "bulyan_f": as_int(payload.get("bulyan_f", payload.get("bulyanF")), defaults.get("bulyan_f", 1)),
                },
                "secure_aggregation_sim": {
                    "enabled": aggregation_mode == "secure_aggregation",
                    "simulation_only": True,
                },
                "dp_noise": {
                    "enabled": dp_noise_enabled,
                    "noise_std": as_float(payload.get("dp_noise_std", payload.get("dpNoiseStd")), defaults.get("dp_noise_std", 0.15)),
                    "formal_accountant": False,
                },
            },
            "privacy_params": {
                "membership_inference": {
                    "enabled": "membership_inference" in enabled_privacy,
                    "evidence_source": str(payload.get("mia_evidence_source", payload.get("evidenceSource")) or "auto"),
                    "sample_count": int(
                        clamp_number(
                            as_int(payload.get("membership_sample_count", payload.get("membershipSampleCount")), bounds.get("default_membership_sample_count", 200)),
                            1,
                            bounds.get("max_membership_sample_count", 5000),
                        )
                    ),
                },
                "interaction_reconstruction": {
                    "enabled": "interaction_reconstruction" in enabled_privacy,
                    "candidate_k": candidate_k,
                    "risk_modality": str(payload.get("risk_modality", payload.get("riskModality")) or "item embedding"),
                },
            },
        },
    }
    return normalized, warnings, errors


def validation_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized, warnings, errors = normalize_workbench_payload(payload)
    return {
        "valid": not errors,
        "status": "validated" if not errors else "invalid",
        "warnings": warnings,
        "errors": errors,
        "normalized_config": normalized,
        "expected_outputs": [
            "config.json",
            "status.json",
            "run.log",
            "result_pointer.json",
            "metrics_summary.json",
        ],
        "disabled_reason": "The workbench generator validates and records bounded smoke configs; it does not start training.",
    }


def write_workbench_job(payload: Dict[str, Any], output_root: Path = DEFAULT_OUTPUT_ROOT, job_id: str | None = None) -> Dict[str, Any]:
    response = validation_response(payload)
    safe_id = safe_job_id(job_id)
    output_root = output_root.resolve()
    job_dir = (output_root / safe_id).resolve()
    if output_root not in [job_dir, *job_dir.parents]:
        raise ValueError("job_path_outside_output_root")
    job_dir.mkdir(parents=True, exist_ok=True)

    created_at = utc_now()
    config_payload = response.get("normalized_config", {})
    status = "disabled" if response["valid"] else "invalid"
    status_payload = {
        "job_id": safe_id,
        "status": status,
        "valid": response["valid"],
        "created_at": created_at,
        "updated_at": created_at,
        "direction": config_payload.get("direction"),
        "scenario_id": config_payload.get("scenario_id"),
        "message": "训练任务启动未接入；已生成受限 smoke 配置和 job 档案。" if response["valid"] else "配置未通过校验。",
        "disabled_reason": response["disabled_reason"],
        "warnings": response["warnings"],
        "errors": response["errors"],
    }
    metrics_summary = {
        "job_id": safe_id,
        "status": status,
        "direction": config_payload.get("direction"),
        "metrics": {},
        "message": "No runtime metrics are generated because this helper does not start training.",
    }
    pointer = {
        "job_id": safe_id,
        "status": status,
        "config": "config.json",
        "status_file": "status.json",
        "metrics_summary": "metrics_summary.json",
        "log": "run.log",
        "result_dir": None,
        "showcase_scenario_id": config_payload.get("scenario_id"),
    }

    write_json(job_dir / "config.json", config_payload)
    write_json(job_dir / "status.json", status_payload)
    write_json(job_dir / "metrics_summary.json", metrics_summary)
    write_json(job_dir / "result_pointer.json", pointer)
    log_lines = [
        f"[{created_at}] Workbench job {safe_id} recorded.",
        f"[validate] status={status}",
        "[boundary] This path does not start real training.",
        "[next] Use existing showcase artifacts or connect a bounded launcher explicitly.",
    ]
    (job_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    response.update(
        {
            "job_id": safe_id,
            "job_status": status,
            "job_dir": repo_relative(job_dir),
            "files": {
                "config": "config.json",
                "status": "status.json",
                "log": "run.log",
                "result_pointer": "result_pointer.json",
                "metrics_summary": "metrics_summary.json",
            },
        }
    )
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a bounded FedVLR workbench smoke config/job artifact.")
    parser.add_argument("--payload", help="Workbench payload as JSON string.")
    parser.add_argument("--payload-file", help="Workbench payload JSON file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for job artifacts.")
    parser.add_argument("--job-id", help="Optional safe job id.")
    parser.add_argument("--options", action="store_true", help="Print workbench options.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate and print the normalized config; do not write a job.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.options:
        print(json.dumps(get_workbench_options(), ensure_ascii=False, indent=2))
        return 0
    if args.payload_file:
        payload = load_json(Path(args.payload_file), {})
    elif args.payload:
        payload = json.loads(args.payload)
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise SystemExit("payload must be a JSON object")
    response = validation_response(payload) if args.validate_only else write_workbench_job(payload, Path(args.output_dir), args.job_id)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response.get("valid", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
