from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "workbench_experiment_schema.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "workbench_jobs"
AMAZON_DATASET = "AMAZON_BEAUTY_POC"
CANONICAL_DATASET_IDS = {"AMAZON_BEAUTY_POC", "KU"}
LAUNCHABLE_MODEL_IDS = {"FedAvg", "FedRAP", "FedNCF", "FCF", "MMFedAvg", "MMFedRAP", "MMFedNCF", "MMFCF"}
EXECUTION_MODE_ALIASES = {
    "full_train": "full_train",
}
TARGET_ZH_RULES = [
    ("Empty Amber Glass Spray Bottles", "琥珀玻璃喷雾瓶套装", "琥珀喷雾瓶"),
    ("Bouquet Garni Body Shower White Musk", "白麝香香氛沐浴露", "白麝香沐浴露"),
    ("Bioré J-Beauty Makeup Removing Moisturizing Cleansing Jelly", "Bioré 卸妆保湿洁面啫喱", "Bioré 卸妆啫喱"),
    ("Bioré Makeup Removing Cleansing Jelly", "Bioré 卸妆保湿洁面啫喱", "Bioré 卸妆啫喱"),
    ("Vitamin C", "维C亮肤精华液", "维C精华"),
    ("Gel Nail Polish Set", "闪粉凝胶甲油套装", "凝胶甲油套装"),
]
CATEGORY_ZH = {
    "All Beauty": "美妆护理",
    "Beauty": "美妆护理",
}

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
    "recommendation_manipulation": ["target_interaction_injection"],
    "membership_inference": [],
    "update_leakage": [],
    "aggregation_defense": ["poisoning_attack"],
}

PRIVACY_BY_DIRECTION = {
    "recommendation_manipulation": [],
    "membership_inference": ["membership_inference_probe"],
    "update_leakage": ["interaction_reconstruction_probe"],
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


def target_zh_names(title: str) -> Tuple[str, str]:
    for needle, display_name, short_name in TARGET_ZH_RULES:
        if needle.lower() in title.lower():
            return display_name, short_name
    compact = title.strip()
    if len(compact) > 18:
        compact = compact[:18].rstrip() + "…"
    return compact or "目标商品", compact or "目标商品"


def canonical_datasets(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in data.get("datasets", []) if item.get("id") in CANONICAL_DATASET_IDS]


def launchable_models(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in data.get("models", []) if item.get("id") in LAUNCHABLE_MODEL_IDS]


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
        display_name_zh, short_name_zh = target_zh_names(title)
        category = meta.get("main_category") or image.get("category")
        short_title = title if len(title) <= 72 else f"{title[:69]}..."
        options.append(
            {
                "item_id": item_id,
                "raw_item_id": meta.get("raw_item_id") or image.get("raw_item_id"),
                "raw_title": title,
                "title": title,
                "short_title": short_title,
                "display_name_zh": display_name_zh,
                "short_name_zh": short_name_zh,
                "category": category,
                "category_zh": CATEGORY_ZH.get(str(category), str(category or "美妆护理")),
                "image_url": meta.get("image_url") or image.get("image_url"),
                "thumbnail_url": f"/showcase/images/{AMAZON_DATASET}/{item_id}?size=thumb" if image.get("thumbnail_path") else None,
                "is_target_sidecar": item_id in set(str(x) for x in (sidecar.get("target_items", []) if isinstance(sidecar, dict) else [])),
            }
        )
    return options


def get_workbench_options() -> Dict[str, Any]:
    data = schema()
    return {
        "schema_version": data.get("version"),
        "directions": data.get("directions", []),
        "datasets": canonical_datasets(data),
        "models": launchable_models(data),
        "adapter_required_models": data.get("adapter_required_models", []),
        "aggregation_visibility_modes": data.get("aggregation_visibility_modes", []),
        "robust_aggregators": data.get("robust_aggregators", []),
        "common_parameters": data.get("common_parameters", []),
        "fixed_parameters": data.get("fixed_parameters", {}),
        "direction_parameters": data.get("direction_parameters", {}),
        "defense_parameters": data.get("defense_parameters", {}),
        "compatibility_matrix": data.get("compatibility_matrix", {}),
        "model_dataset_execution": data.get(
            "model_dataset_code_capabilities",
            data.get("model_dataset_execution", {}),
        ),
        "parameter_descriptors": data.get("parameter_descriptors", {}),
        "bounds": data.get("bounds", {}),
        "defaults": data.get("defaults", {}),
        "target_items": get_target_item_options(),
        "notes": [
            "Workbench jobs always request full training.",
            "Unsupported direction, dataset, and model combinations fail validation instead of falling back to another execution path.",
            "MGCN family models require adapters and are not launchable from this workbench.",
        ],
    }


def normalize_direction(value: Any) -> str:
    key = str(value or "recommendation_manipulation").strip()
    if key not in DIRECTION_ALIASES:
        raise ValueError(f"unknown_direction:{key}")
    return DIRECTION_ALIASES[key]


def normalize_execution_mode(value: Any, default: str = "full_train") -> str:
    key = str(value or default).strip()
    if key not in EXECUTION_MODE_ALIASES:
        raise ValueError(f"unknown_execution_mode:{key}")
    return EXECUTION_MODE_ALIASES[key]


def supports_full_train(
    data: Dict[str, Any], direction: str, dataset: str, model: str
) -> bool:
    capability = execution_capability(data, dataset, model)
    allowed_modes = set(capability.get("allowed_execution_modes", []))
    supported_directions = set(
        capability.get(
            "supported_directions",
            [item.get("id") for item in data.get("directions", [])],
        )
    )
    return "full_train" in allowed_modes and direction in supported_directions


def execution_capability(data: Dict[str, Any], dataset: str, model: str) -> Dict[str, Any]:
    matrix = data.get(
        "model_dataset_code_capabilities",
        data.get("model_dataset_execution", {}),
    )
    dataset_record = matrix.get(dataset, {}) if isinstance(matrix, dict) else {}
    record = dataset_record.get(model, {}) if isinstance(dataset_record, dict) else {}
    if isinstance(record, dict) and record:
        return dict(record)
    return {
        "status": "unsupported",
        "allowed_execution_modes": [],
        "message": "当前组合不支持工作台真实全量训练。",
    }


def normalize_workbench_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    data = schema()
    bounds = data.get("bounds", {})
    defaults = data.get("defaults", {})
    descriptors = data.get("parameter_descriptors", {})
    fixed_parameters = data.get("fixed_parameters", {})
    warnings: List[str] = []
    errors: List[str] = []

    direction = normalize_direction(payload.get("direction") or payload.get("play_id") or payload.get("playId"))
    direction_record = next((item for item in data.get("directions", []) if item.get("id") == direction), {})

    models = {item.get("id"): item for item in launchable_models(data)}
    adapter_models = {item.get("id"): item for item in data.get("adapter_required_models", [])}
    datasets = {item.get("id"): item for item in canonical_datasets(data)}
    compatibility = data.get("compatibility_matrix", {})
    field_errors: Dict[str, List[str]] = {}

    def add_error(field: str, code: str) -> None:
        errors.append(code)
        field_errors.setdefault(field, []).append(code)

    def descriptor(field: str) -> Dict[str, Any]:
        record = descriptors.get(field, {}) if isinstance(descriptors, dict) else {}
        return record if isinstance(record, dict) else {}

    def raw_value(*names: str) -> Any:
        for name in names:
            if name in payload:
                return payload[name]
        return None

    def numeric_parameter(
        field: str,
        names: Tuple[str, ...],
        *,
        integer: bool = False,
        fallback: float | int | None = None,
        validate: bool = True,
    ) -> float | int:
        record = descriptor(field)
        default = record.get("default", fallback if fallback is not None else 0)
        raw = raw_value(*names)
        value = as_int(raw, int(default)) if integer else as_float(raw, float(default))
        minimum = record.get("min")
        maximum = record.get("max")
        step = record.get("step")
        if validate and integer and raw is not None:
            raw_number = as_float(raw, float(value))
            if not raw_number.is_integer():
                add_error(field, f"{field}_must_be_integer")
        if validate and minimum is not None and value < minimum:
            add_error(field, f"{field}_below_min:{minimum}")
        if validate and maximum is not None and value > maximum:
            add_error(field, f"{field}_above_max:{maximum}")
        if validate and step and step > 0:
            origin = minimum if minimum is not None else 0
            step_count = (float(value) - float(origin)) / float(step)
            if abs(step_count - round(step_count)) > 1e-8:
                add_error(field, f"{field}_invalid_step:{step}")
        return value

    def enum_parameter(field: str, names: Tuple[str, ...], *, fallback: str = "", validate: bool = True) -> str:
        record = descriptor(field)
        value = str(raw_value(*names) or record.get("default", fallback))
        options = record.get("options", [])
        if validate and options and value not in options:
            add_error(field, f"{field}_invalid_option:{value}")
        return value

    model = str(payload.get("model") or direction_record.get("default_model") or "FedAvg")
    dataset = str(payload.get("dataset") or direction_record.get("default_dataset") or "AMAZON_BEAUTY_POC")
    try:
        requested_execution_mode = normalize_execution_mode(payload.get("execution_mode", payload.get("executionMode")), defaults.get("execution_mode", "full_train"))
    except ValueError as exc:
        requested_execution_mode = defaults.get("execution_mode", "full_train")
        add_error("execution_mode", str(exc))
    execution_mode = requested_execution_mode
    if model in adapter_models:
        add_error("model", f"adapter_required_model:{model}")
    if model not in models:
        add_error("model", f"unknown_model:{model}")
    if dataset not in datasets:
        add_error("dataset", f"unknown_dataset:{dataset}")
    selectable_for_dataset = set(compatibility.get(dataset, LAUNCHABLE_MODEL_IDS))
    if model in models and dataset in datasets and model not in selectable_for_dataset:
        warnings.append(f"model_dataset_outside_selectable_matrix:{model}:{dataset}")
    capability = execution_capability(data, dataset, model)
    if requested_execution_mode != "full_train" or not supports_full_train(
        data, direction, dataset, model
    ):
        add_error("execution_mode", f"full_train_not_available:{direction}:{model}:{dataset}")
    verification_status = str(capability.get("verification_status", "not_verified"))
    verified_directions = set(capability.get("verified_directions", []))
    if (
        verification_status not in {"full_train_verified", "smoke_verified"}
        and direction not in verified_directions
    ):
        warnings.append(
            f"full_train_code_path_not_yet_verified:{direction}:{model}:{dataset}"
        )

    total_rounds = as_int(payload.get("total_rounds", payload.get("totalRounds")), bounds.get("default_total_rounds", 10))
    total_rounds = int(clamp_number(total_rounds, 1, bounds.get("max_total_rounds", 10)))
    epochs = as_int(payload.get("epochs", total_rounds), total_rounds)
    epochs = int(clamp_number(epochs, 1, bounds.get("max_epochs", 10)))
    local_epochs = as_int(payload.get("local_epochs", payload.get("localEpochs")), bounds.get("default_local_epochs", 5))
    local_epochs = int(clamp_number(local_epochs, 1, bounds.get("max_local_epochs", 5)))

    client_ratio = float(numeric_parameter("client_sampling_ratio", ("client_sampling_ratio", "clientSamplingRate")))
    total_client_count = max(1, as_int(payload.get("total_client_count", payload.get("totalClientCount")), descriptor("client_count").get("default", 100)))
    sampled_client_count = max(1, math.ceil(total_client_count * client_ratio))
    batch_size = int(numeric_parameter("batch_size", ("batch_size", "batchSize"), integer=True))
    batch_options = descriptor("batch_size").get("options", [])
    if batch_options and batch_size not in batch_options:
        add_error("batch_size", f"batch_size_invalid_option:{batch_size}")
    seed = int(numeric_parameter("seed", ("seed",), integer=True))
    gradient_clip = float(numeric_parameter("gradient_clip", ("gradient_clip", "gradientClip")))

    recommendation_direction = direction == "recommendation_manipulation"
    recommendation_malicious_ratio = float(
        numeric_parameter(
            "malicious_client_ratio",
            ("malicious_client_ratio", "poisoningRatio"),
            validate=recommendation_direction,
        )
    )
    anomaly_client_ratio = as_float(
        payload.get("anomaly_client_ratio", payload.get("anomalyClientRatio")),
        bounds.get("default_malicious_client_ratio", 0.2),
    )
    if direction == "aggregation_defense" and not 0 <= anomaly_client_ratio <= 0.6:
        add_error("anomaly_client_ratio", "anomaly_client_ratio_out_of_range:0:0.6")
    malicious_ratio = recommendation_malicious_ratio if recommendation_direction else anomaly_client_ratio if direction == "aggregation_defense" else 0.0

    aggregation_mode = str(payload.get("aggregation_mode", payload.get("aggregationMode")) or "plain_updates")
    if aggregation_mode not in {"plain_updates", "secure_aggregation"}:
        errors.append(f"unknown_aggregation_mode:{aggregation_mode}")
    robust_aggregators = [item for item in as_list(payload.get("robust_aggregators", payload.get("robustAggregators"))) if item != "none"]
    allowed_robust = set(data.get("robust_aggregators", []))
    for item in robust_aggregators:
        if item not in allowed_robust:
            add_error("robust_aggregators", f"unknown_robust_aggregator:{item}")
    if len(robust_aggregators) > 1:
        add_error("robust_aggregators", "multiple_robust_aggregators_not_supported")
    if aggregation_mode == "secure_aggregation" and robust_aggregators:
        add_error("secure_aggregation_enabled", "secure_aggregation_conflicts_with_robust_aggregation")
        add_error("robust_aggregators", "secure_aggregation_conflicts_with_robust_aggregation")

    dp_noise_enabled = as_bool(payload.get("dp_noise_enabled", payload.get("dpNoiseEnabled")), False)
    krum_max = math.floor((sampled_client_count - 3) / 2)
    bulyan_max = math.floor((sampled_client_count - 3) / 4)
    krum_default = min(1, max(0, krum_max))
    bulyan_default = min(1, max(0, bulyan_max))
    trim_ratio = float(numeric_parameter("trim_ratio", ("trim_ratio", "trimRatio"), validate="TrimmedMean" in robust_aggregators))
    trim_min_keep = int(numeric_parameter("trim_min_keep", ("trim_min_keep", "trimMinKeep"), integer=True, validate=False))
    raw_trim_min_keep = raw_value("trim_min_keep", "trimMinKeep")
    raw_krum_f = raw_value("krum_f", "krumF")
    raw_bulyan_f = raw_value("bulyan_f", "bulyanF")
    krum_f = as_int(raw_krum_f, krum_default)
    bulyan_f = as_int(raw_bulyan_f, bulyan_default)
    multi_krum_enabled = as_bool(raw_value("multi_krum_enabled", "multiKrumEnabled"), False)
    distance_metric = enum_parameter("distance_metric", ("distance_metric", "distanceMetric"), validate="Krum" in robust_aggregators)
    defense_gradient_clip_norm = float(
        numeric_parameter(
            "gradient_clip_norm",
            ("gradient_clip_norm", "gradientClipNorm"),
            validate=bool({"Krum", "Median"}.intersection(robust_aggregators)),
        )
    )
    outlier_strategy = enum_parameter(
        "outlier_strategy",
        ("outlier_strategy", "outlierStrategy"),
        validate="Median" in robust_aggregators,
    )
    bulyan_selection_ratio = float(
        numeric_parameter(
            "bulyan_selection_ratio",
            ("bulyan_selection_ratio", "bulyanSelectionRatio"),
            validate="Bulyan" in robust_aggregators,
        )
    )
    if "Krum" in robust_aggregators:
        if raw_krum_f is not None and not as_float(raw_krum_f, float(krum_f)).is_integer():
            add_error("krum_f", "krum_f_must_be_integer")
        if sampled_client_count < 3:
            add_error("krum_f", f"krum_requires_at_least_3_sampled_clients:{sampled_client_count}")
        elif krum_f < 0 or krum_f > krum_max:
            add_error("krum_f", f"krum_f_out_of_range:0:{krum_max}")
    if "TrimmedMean" in robust_aggregators:
        if raw_trim_min_keep is not None and not as_float(raw_trim_min_keep, float(trim_min_keep)).is_integer():
            add_error("trim_min_keep", "trim_min_keep_must_be_integer")
        if sampled_client_count < 2:
            add_error("trim_min_keep", f"trimmed_mean_requires_at_least_2_sampled_clients:{sampled_client_count}")
        elif trim_min_keep < 2 or trim_min_keep > sampled_client_count:
            add_error("trim_min_keep", f"trim_min_keep_out_of_range:2:{sampled_client_count}")
    if "Bulyan" in robust_aggregators:
        if raw_bulyan_f is not None and not as_float(raw_bulyan_f, float(bulyan_f)).is_integer():
            add_error("bulyan_f", "bulyan_f_must_be_integer")
        if sampled_client_count < 3:
            add_error("bulyan_f", f"bulyan_requires_at_least_3_sampled_clients:{sampled_client_count}")
        elif bulyan_f < 0 or bulyan_f > bulyan_max:
            add_error("bulyan_f", f"bulyan_f_out_of_range:0:{bulyan_max}")

    noise_multiplier = float(numeric_parameter("noise_multiplier", ("noise_multiplier", "noiseMultiplier"), validate=dp_noise_enabled))
    max_grad_norm = float(numeric_parameter("max_grad_norm", ("max_grad_norm", "maxGradNorm"), validate=dp_noise_enabled))
    target_delta = float(numeric_parameter("target_delta", ("target_delta", "targetDelta"), validate=False))
    if dp_noise_enabled and target_delta not in descriptor("target_delta").get("options", []):
        add_error("target_delta", f"target_delta_invalid_option:{target_delta}")
    dp_seed = int(numeric_parameter("dp_seed", ("dp_seed", "dpSeed"), integer=True, validate=dp_noise_enabled))

    attack_strength = enum_parameter("attack_strength", ("attack_strength", "attackStrength"), validate=recommendation_direction)
    attack_strength_multiplier = float(descriptor("attack_strength").get("value_map", {}).get(attack_strength, 1.0))
    injection_ratio = float(numeric_parameter("injection_ratio", ("injection_ratio", "injectionRatio"), validate=recommendation_direction))
    max_injections_per_client = int(
        numeric_parameter(
            "max_injections_per_client",
            ("max_injections_per_client", "maxInjectionsPerClient"),
            integer=True,
            validate=recommendation_direction,
        )
    )
    target_loss_weight = float(numeric_parameter("target_loss_weight", ("target_loss_weight", "targetLossWeight"), validate=recommendation_direction))
    target_rank_selector = enum_parameter("target_rank_selector", ("target_rank_selector", "targetRankSelector"), validate=recommendation_direction)
    fixed_top_k = int(fixed_parameters.get("top_k", 50))
    if raw_value("top_k", "topK") not in (None, fixed_top_k, str(fixed_top_k)):
        warnings.append(f"top_k_fixed_to_{fixed_top_k}")
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
    mia_evidence_source = str(payload.get("mia_evidence_source", payload.get("evidenceSource")) or "auto")
    if mia_evidence_source not in {"rank", "unmasked rank", "checkpoint score", "auto"}:
        add_error("mia_evidence_source", f"mia_evidence_source_invalid_option:{mia_evidence_source}")
    label_source = str(payload.get("label_source", payload.get("labelSource")) or defaults.get("label_source", "membership_labels"))
    if label_source not in {"membership labels", "membership_labels", "scenario labels", "scenario_labels", "auto labels", "auto_labels", "auto"}:
        add_error("label_source", f"label_source_invalid_option:{label_source}")
    threshold_strategy = str(payload.get("threshold_strategy", payload.get("thresholdStrategy")) or defaults.get("threshold_strategy", "auto"))
    if threshold_strategy not in {"auto", "median", "fixed"}:
        add_error("threshold_strategy", f"threshold_strategy_invalid_option:{threshold_strategy}")
    mia_model = str(payload.get("mia_model", payload.get("miaModel")) or "rank_proxy")
    if mia_model not in {"threshold", "logistic_probe", "rank_proxy"}:
        add_error("mia_model", f"mia_model_invalid_option:{mia_model}")

    robust_defense_names = {
        "Krum": "krum",
        "Median": "median",
        "TrimmedMean": "trimmed_mean",
        "Bulyan": "bulyan",
    }
    enabled_defenses: List[str] = [robust_defense_names[item] for item in robust_aggregators if item in robust_defense_names]
    if dp_noise_enabled:
        enabled_defenses.append("dp_noise")
    if aggregation_mode == "secure_aggregation":
        enabled_defenses.append("secure_aggregation_sim")

    base_attack = str(payload.get("base_attack", payload.get("baseAttack")) or defaults.get("base_attack", "none"))
    if direction == "aggregation_defense" and base_attack not in {"none", "malicious_update"}:
        add_error("base_attack", f"aggregation_defense_invalid_base_attack:{base_attack}")
    perturbation_type = str(payload.get("perturbation_type", payload.get("perturbationType")) or "sign_flip")
    if direction == "aggregation_defense" and base_attack == "malicious_update" and perturbation_type != "sign_flip":
        add_error("perturbation_type", f"full_train_perturbation_not_supported:{perturbation_type}")
    enabled_attacks = [] if direction == "aggregation_defense" and base_attack == "none" else ATTACKS_BY_DIRECTION[direction]
    malicious_clients_enabled = malicious_ratio > 0 and not (direction == "aggregation_defense" and base_attack == "none")
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
        "requested_execution_mode": requested_execution_mode,
        "execution_mode": execution_mode,
        "execution_capability": capability,
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
            "gradient_clip": gradient_clip,
            "batch_size": batch_size,
            "seed": seed,
            "total_client_count": total_client_count,
            "sampled_client_count": sampled_client_count,
            "top_k": fixed_top_k,
            "save_topk": as_bool(payload.get("save_topk", payload.get("saveTopK")), defaults.get("save_topk", True)),
            "export_artifact": as_bool(payload.get("export_artifact", payload.get("exportArtifact")), defaults.get("export_artifact", True)),
        },
        "attack": {
            "malicious_client_ratio": malicious_ratio,
            "attack_strength": attack_strength,
            "attack_strength_multiplier": attack_strength_multiplier,
            "target_item_id": target_item_id,
            "target_item_title": target_title or (target_record or {}).get("title"),
            "injection_ratio": injection_ratio,
            "max_injections_per_client": max_injections_per_client,
            "target_loss_weight": target_loss_weight,
            "target_rank_selector": target_rank_selector,
        },
        "privacy": {
            "mia_evidence_source": mia_evidence_source,
            "label_source": label_source,
            "threshold_strategy": threshold_strategy,
            "mia_model": mia_model,
            "member_nonmember_ratio": as_float(payload.get("member_nonmember_ratio", payload.get("memberNonmemberRatio")), 1.0),
            "export_pair_scores": as_bool(payload.get("export_pair_scores", payload.get("exportPairScores")), defaults.get("export_pair_scores", True)),
            "membership_sample_count": int(
                clamp_number(
                    as_int(payload.get("membership_sample_count", payload.get("membershipSampleCount")), bounds.get("default_membership_sample_count", 200)),
                    1,
                    bounds.get("max_membership_sample_count", 5000),
                )
            ),
            "candidate_k": candidate_k,
            "hit_k": as_int(payload.get("hit_k", payload.get("hitK")), candidate_k),
            "client_count": as_int(payload.get("client_count", payload.get("clientCount")), 100),
            "export_reconstruction": as_bool(payload.get("export_reconstruction", payload.get("exportReconstruction")), defaults.get("export_reconstruction", True)),
            "risk_modality": str(payload.get("risk_modality", payload.get("riskModality")) or "item embedding"),
            "update_input_source": str(payload.get("update_input_source", payload.get("updateInputSource")) or "client_update"),
            "candidate_pool_size": as_int(payload.get("candidate_pool_size", payload.get("candidatePoolSize")), 500),
            "similarity_method": str(payload.get("similarity_method", payload.get("similarityMethod")) or "cosine"),
            "show_candidate_images": as_bool(payload.get("show_candidate_images", payload.get("showCandidateImages")), True),
        },
        "defense": {
            "base_attack": base_attack,
            "gradient_clip_norm": defense_gradient_clip_norm,
            "outlier_strategy": outlier_strategy,
            "trim_ratio": trim_ratio,
            "trim_min_keep": trim_min_keep,
            "krum_f": krum_f,
            "multi_krum_enabled": multi_krum_enabled,
            "distance_metric": distance_metric,
            "bulyan_f": bulyan_f,
            "bulyan_selection_ratio": bulyan_selection_ratio,
            "dp_noise_std": noise_multiplier,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "target_delta": target_delta,
            "dp_seed": dp_seed,
        },
        "field_errors": field_errors,
        "unified_experiment_config": {
            "model": model,
            "dataset": dataset,
            "scenario": scenario,
            "type": "WorkbenchFullTrain",
            "comment": f"workbench:{direction}",
            "execution_mode": execution_mode,
            "requested_execution_mode": requested_execution_mode,
            "execution_capability": capability,
            "enabled_attacks": enabled_attacks,
            "enabled_defenses": enabled_defenses,
            "enabled_privacy_metrics": enabled_privacy,
            "malicious_client_config": {
                "enabled": malicious_clients_enabled,
                "mode": "ratio" if malicious_clients_enabled else "none",
                "ratio": malicious_ratio,
                "client_ids": [],
            },
            "training_params": {
                "epochs": epochs,
                "total_rounds": total_rounds,
                "local_epochs": local_epochs,
                "client_sampling_ratio": client_ratio,
                "clients_sample_ratio": client_ratio,
                "lr": as_float(payload.get("learning_rate", payload.get("learningRate")), defaults.get("learning_rate", 0.001)),
                "learning_rate": as_float(payload.get("learning_rate", payload.get("learningRate")), defaults.get("learning_rate", 0.001)),
                "weight_decay": as_float(payload.get("weight_decay", payload.get("weightDecay")), defaults.get("weight_decay", 0.0)),
                "gradient_clip": gradient_clip,
                "batch_size": batch_size,
                "seed": seed,
                "save_recommended_topk": as_bool(payload.get("save_topk", payload.get("saveTopK")), defaults.get("save_topk", True)),
                "top_k": fixed_top_k,
                "topk": [fixed_top_k],
                "recommendation_topk": fixed_top_k,
                "export_security_artifact": as_bool(payload.get("export_artifact", payload.get("exportArtifact")), defaults.get("export_artifact", True)),
            },
            "attack_params": {
                "target_interaction_injection": {
                    "enabled": direction == "recommendation_manipulation",
                    "target_item_ids": [target_item_id],
                    "target_item_title": target_title or (target_record or {}).get("title"),
                    "malicious_client_ratio": malicious_ratio,
                    "injection_ratio": injection_ratio,
                    "max_injections_per_client": max_injections_per_client,
                    "target_loss_weight": target_loss_weight,
                    "attack_strength": attack_strength_multiplier,
                    "target_rank_selector": target_rank_selector,
                    "planner_only": False,
                },
                "poisoning_attack": {
                    "base_attack": base_attack,
                    "poisoning_enabled_substrategies": [perturbation_type],
                    "perturbation_type": perturbation_type,
                    "perturbation_strength": as_float(payload.get("perturbation_strength", payload.get("perturbationStrength")), 1.5),
                    "poisoning_attack_scale": as_float(payload.get("perturbation_strength", payload.get("perturbationStrength")), 1.5),
                    "poisoning_sign_flip_scale": as_float(payload.get("perturbation_strength", payload.get("perturbationStrength")), 1.5),
                }
            },
            "defense_params": {
                "krum": {
                    "krum_f": krum_f,
                    "multi_krum_enabled": multi_krum_enabled,
                    "distance_metric": distance_metric,
                    "gradient_clip_norm": defense_gradient_clip_norm,
                },
                "median": {
                    "enabled": "Median" in robust_aggregators,
                    "gradient_clip_norm": defense_gradient_clip_norm,
                    "outlier_strategy": outlier_strategy,
                },
                "trimmed_mean": {
                    "trim_ratio": trim_ratio,
                    "trim_min_keep": trim_min_keep,
                },
                "bulyan": {
                    "bulyan_f": bulyan_f,
                    "bulyan_selection_ratio": bulyan_selection_ratio,
                },
                "secure_aggregation_sim": {
                    "enabled": aggregation_mode == "secure_aggregation",
                    "simulation_only": True,
                },
                "dp_noise": {
                    "enabled": dp_noise_enabled,
                    "noise_std": noise_multiplier,
                    "noise_multiplier": noise_multiplier,
                    "max_grad_norm": max_grad_norm,
                    "target_delta": target_delta,
                    "seed": dp_seed,
                    "formal_accountant": False,
                },
            },
            "privacy_params": {
                "membership_inference_probe": {
                    "enabled": "membership_inference_probe" in enabled_privacy,
                    "evidence_source": mia_evidence_source,
                    "label_source": label_source,
                    "threshold_strategy": threshold_strategy,
                    "mia_model": mia_model,
                    "member_nonmember_ratio": as_float(payload.get("member_nonmember_ratio", payload.get("memberNonmemberRatio")), 1.0),
                    "export_pair_scores": as_bool(payload.get("export_pair_scores", payload.get("exportPairScores")), defaults.get("export_pair_scores", True)),
                    "sample_count": int(
                        clamp_number(
                            as_int(payload.get("membership_sample_count", payload.get("membershipSampleCount")), bounds.get("default_membership_sample_count", 200)),
                            1,
                            bounds.get("max_membership_sample_count", 5000),
                        )
                    ),
                },
                "interaction_reconstruction_probe": {
                    "enabled": "interaction_reconstruction_probe" in enabled_privacy,
                    "candidate_k": candidate_k,
                    "candidate_pool_size": as_int(payload.get("candidate_pool_size", payload.get("candidatePoolSize")), 500),
                    "risk_modality": str(payload.get("risk_modality", payload.get("riskModality")) or "item embedding"),
                    "update_input_source": str(payload.get("update_input_source", payload.get("updateInputSource")) or "client_update"),
                    "similarity_method": str(payload.get("similarity_method", payload.get("similarityMethod")) or "cosine"),
                    "audit_client_count": as_int(payload.get("client_count", payload.get("clientCount")), 5),
                    "export_candidate_evidence": as_bool(payload.get("export_reconstruction", payload.get("exportReconstruction")), defaults.get("export_reconstruction", True)),
                },
            },
        },
    }
    return normalized, warnings, errors


def validation_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized, warnings, errors = normalize_workbench_payload(payload)
    field_errors = normalized.get("field_errors", {}) if isinstance(normalized.get("field_errors"), dict) else {}
    field_messages = []
    for field, messages in field_errors.items():
        if isinstance(messages, list):
            field_messages.extend(f"{field}: {message}" for message in messages)
    error_parts = []
    for item in [*field_messages, *(str(item) for item in errors)]:
        if item not in error_parts:
            error_parts.append(item)
    return {
        "valid": not errors,
        "status": "validated" if not errors else "invalid",
        "warnings": warnings,
        "errors": errors,
        "field_errors": field_errors,
        "error_message": "；".join(error_parts) if errors else None,
        "normalized_config": normalized,
        "expected_outputs": [
            "config.json",
            "launcher_config.json",
            "status.json",
            "run.log",
            "result_pointer.json",
            "metrics_summary.json",
        ],
        "disabled_reason": None,
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
        "message": "已生成全量训练配置和 job 档案；此辅助命令本身不启动训练。" if response["valid"] else "配置未通过校验。",
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
        "[next] Submit the validated config through the workbench API to start training.",
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
    parser = argparse.ArgumentParser(description="Generate a FedVLR workbench full-training config/job artifact.")
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
