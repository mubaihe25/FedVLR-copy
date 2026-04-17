from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.configurator import Config
from utils.logger import init_logger
from utils.quick_start import _prepare_data
from utils.utils import get_model, get_trainer, init_seed, save_experiment_results


CAPABILITY_PATH = ROOT / "configs" / "model_attack_defense_capabilities.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a FedVLR experiment from the unified config object."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a unified experiment config JSON file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the mapped FedVLR config without running training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias of --validate-only.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Treat unverified combinations as errors instead of warnings.",
    )
    parser.add_argument(
        "--allow-blocked-model",
        action="store_true",
        help="Allow blocked models to pass validation. Use only for local debugging.",
    )
    return parser


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_training_params(training_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize unified training params into legacy keys consumed by trainers.

    The project still has two naming layers: unified configs use concise fields
    such as ``lr`` and ``l2_reg``, while the common optimizer path consumes
    ``learning_rate`` and ``weight_decay``. Keep both aliases present so model-
    specific trainers and generic trainers observe the same intended values.
    """
    normalized = dict(training_params or {})

    if "lr" in normalized and "learning_rate" not in normalized:
        normalized["learning_rate"] = normalized["lr"]
    elif "learning_rate" in normalized and "lr" not in normalized:
        normalized["lr"] = normalized["learning_rate"]

    if "l2_reg" in normalized and "weight_decay" not in normalized:
        normalized["weight_decay"] = normalized["l2_reg"]
    elif "weight_decay" in normalized and "l2_reg" not in normalized:
        normalized["l2_reg"] = normalized["weight_decay"]

    if "learner" in normalized and "optimizer" not in normalized:
        normalized["optimizer"] = normalized["learner"]
    elif "optimizer" in normalized and "learner" not in normalized:
        normalized["learner"] = normalized["optimizer"]

    return normalized


def get_model_record(capabilities: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    for item in capabilities.get("models", []):
        if str(item.get("name")) == model_name:
            return item
    return None


def module_names(capabilities: Dict[str, Any], section: str) -> set[str]:
    names: set[str] = set()
    for item in capabilities.get(section, []):
        name = str(item.get("name", "")).strip()
        if name:
            names.add(name)
        for alias in item.get("aliases", []):
            alias_name = str(alias).strip()
            if alias_name:
                names.add(alias_name)
    return names


def matches_validated_combination(
    capabilities: Dict[str, Any],
    model_name: str,
    scenario: str,
    attacks: List[str],
    defenses: List[str],
    privacy_metrics: List[str],
) -> Tuple[bool, Optional[str]]:
    for item in capabilities.get("validated_combinations", []):
        if item.get("scenario") != scenario:
            continue
        if list(item.get("attacks", [])) != attacks:
            continue
        if list(item.get("defenses", [])) != defenses:
            continue
        if list(item.get("privacy_metrics", [])) != privacy_metrics:
            continue
        if model_name not in item.get("validated_models", []):
            return False, "combination_known_but_model_not_validated:{}".format(
                item.get("name")
            )
        return True, str(item.get("name"))
    return False, "combination_not_in_validated_matrix"


def validate_config(
    config: Dict[str, Any],
    capabilities: Dict[str, Any],
    strict_validation: bool = False,
    allow_blocked_model: bool = False,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    model_name = str(config.get("model", "")).strip()
    dataset_name = str(config.get("dataset", "KU")).strip()
    scenario = str(config.get("scenario", "custom")).strip()
    attacks = normalize_list(config.get("enabled_attacks", []))
    defenses = normalize_list(config.get("enabled_defenses", []))
    privacy_metrics = normalize_list(config.get("enabled_privacy_metrics", []))

    if not model_name:
        errors.append("missing_model")
        return errors, warnings

    model_record = get_model_record(capabilities, model_name)
    if not model_record:
        errors.append("unknown_model:{}".format(model_name))
        return errors, warnings

    status = str(model_record.get("compatibility_status", "unknown"))
    if status == "blocked" and not allow_blocked_model:
        errors.append("blocked_model:{}:{}".format(model_name, model_record.get("notes", "")))

    supported_datasets = model_record.get("supported_datasets", [])
    if supported_datasets and dataset_name not in supported_datasets:
        warnings.append(
            "dataset_not_listed_for_model:{}:{} supported={}".format(
                model_name, dataset_name, supported_datasets
            )
        )

    known_attacks = module_names(capabilities, "attacks")
    known_defenses = module_names(capabilities, "defenses")
    known_privacy = module_names(capabilities, "privacy_metrics")
    for name in attacks:
        if name not in known_attacks:
            errors.append("unknown_attack:{}".format(name))
    for name in defenses:
        if name not in known_defenses:
            errors.append("unknown_defense:{}".format(name))
    for name in privacy_metrics:
        if name not in known_privacy:
            errors.append("unknown_privacy_metric:{}".format(name))

    if len(attacks) > int(capabilities.get("max_enabled_attacks", 2)):
        errors.append("too_many_attacks:{}>{}".format(len(attacks), capabilities.get("max_enabled_attacks")))
    recommended_defense_limit = capabilities.get("recommended_max_enabled_defenses")
    if recommended_defense_limit is not None and len(defenses) > int(recommended_defense_limit):
        warnings.append(
            "defense_chain_exceeds_recommended_length:{}>{}".format(
                len(defenses), recommended_defense_limit
            )
        )
    if len(privacy_metrics) > int(capabilities.get("max_enabled_privacy_metrics", 3)):
        errors.append(
            "too_many_privacy_metrics:{}>{}".format(
                len(privacy_metrics), capabilities.get("max_enabled_privacy_metrics")
            )
        )

    if scenario == "baseline" and (attacks or defenses or privacy_metrics):
        warnings.append("baseline_with_enabled_modules")
    if scenario == "attack_only" and (not attacks or defenses):
        warnings.append("attack_only_scenario_does_not_match_enabled_modules")
    if scenario == "defense_only" and (attacks or not defenses):
        warnings.append("defense_only_scenario_does_not_match_enabled_modules")
    if scenario == "attack_and_defense" and (not attacks or not defenses):
        warnings.append("attack_and_defense_scenario_does_not_match_enabled_modules")

    is_validated, reason = matches_validated_combination(
        capabilities, model_name, scenario, attacks, defenses, privacy_metrics
    )
    if not is_validated:
        message = "unvalidated_combination:{}".format(reason)
        if strict_validation:
            errors.append(message)
        else:
            warnings.append(message)

    return errors, warnings


def merge_module_params(
    flat_config: Dict[str, Any],
    enabled_modules: List[str],
    grouped_params: Dict[str, Any],
) -> None:
    for module_name in enabled_modules:
        module_params = grouped_params.get(module_name, {})
        if not isinstance(module_params, dict):
            raise ValueError("module params for {} must be an object".format(module_name))
        flat_config.update(module_params)


def build_fedvlr_config(
    unified_config: Dict[str, Any],
    capabilities: Dict[str, Any],
) -> Dict[str, Any]:
    model_name = str(unified_config["model"])
    model_record = get_model_record(capabilities, model_name) or {}
    attacks = normalize_list(unified_config.get("enabled_attacks", []))
    defenses = normalize_list(unified_config.get("enabled_defenses", []))
    privacy_metrics = normalize_list(unified_config.get("enabled_privacy_metrics", []))
    malicious_config = unified_config.get("malicious_client_config") or {}
    training_params = normalize_training_params(unified_config.get("training_params") or {})

    flat_config: Dict[str, Any] = {
        "use_gpu": False,
        "seed": 42,
        "type": unified_config.get("type", "LauncherExperiment"),
        "comment": unified_config.get("comment", "launcher_experiment"),
        "hyper_parameters": [],
        "epochs": 1,
        "local_epochs": 1,
        "clients_sample_ratio": 1.0,
        "eval_step": 1,
        "collect_round_metrics": True,
        "enable_experiment_hooks": False,
    }
    flat_config.update(model_record.get("recommended_training_overrides", {}))
    flat_config.update(training_params)
    flat_config.update(
        {
            "enabled_attacks": attacks,
            "enabled_defenses": defenses,
            "enabled_privacy_metrics": privacy_metrics,
            "enable_malicious_clients": bool(
                malicious_config.get("enabled", bool(attacks))
            ),
            "malicious_client_mode": malicious_config.get(
                "mode", "ratio" if attacks else "none"
            ),
            "malicious_client_ratio": float(malicious_config.get("ratio", 0.2 if attacks else 0.0)),
            "malicious_client_ids": malicious_config.get("client_ids", []),
        }
    )

    merge_module_params(flat_config, attacks, unified_config.get("attack_params") or {})
    merge_module_params(flat_config, defenses, unified_config.get("defense_params") or {})
    merge_module_params(
        flat_config, privacy_metrics, unified_config.get("privacy_params") or {}
    )
    flat_config.update(unified_config.get("extra_config") or {})
    return flat_config


def reset_logging() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def derive_output_paths(config: Config) -> Dict[str, str]:
    result_file_name = Path(config["result_file_name"])
    base_name = result_file_name.stem
    return {
        "csv": str(result_file_name),
        "experiment_result_json": str(
            result_file_name.with_name("{}.experiment_result.json".format(base_name))
        ),
        "experiment_summary_json": str(
            result_file_name.with_name("{}.experiment_summary.json".format(base_name))
        ),
    }


def build_csv_params(unified_config: Dict[str, Any], fedvlr_config: Dict[str, Any]) -> Dict[str, Any]:
    params = {
        "scenario": unified_config.get("scenario"),
        "model": unified_config.get("model"),
        "dataset": unified_config.get("dataset"),
    }
    params.update(fedvlr_config)
    params["enabled_attacks"] = ",".join(fedvlr_config.get("enabled_attacks", []))
    params["enabled_defenses"] = ",".join(fedvlr_config.get("enabled_defenses", []))
    params["enabled_privacy_metrics"] = ",".join(
        fedvlr_config.get("enabled_privacy_metrics", [])
    )
    return params


def run_experiment(unified_config: Dict[str, Any], fedvlr_config: Dict[str, Any]) -> Dict[str, Any]:
    config = Config(
        model=unified_config["model"],
        dataset=unified_config["dataset"],
        config_dict=fedvlr_config,
        mg=False,
    )
    reset_logging()
    init_logger(config)
    init_seed(config["seed"])

    train_data, valid_data, test_data = _prepare_data(config)
    model = get_model(config["model"])(config, train_data).to(config["device"])
    trainer = get_trainer(config["model"], config["is_federated"])(config, model, False)
    _, _, best_test_upon_valid = trainer.fit(
        train_data,
        valid_data=valid_data,
        test_data=test_data,
        saved=False,
    )

    try:
        csv_params = build_csv_params(unified_config, fedvlr_config)
        csv_params["result_file_name"] = config["result_file_name"]
        save_experiment_results(
            csv_params,
            best_test_upon_valid or {},
            config["result_file_name"],
        )
    except Exception as exc:  # noqa: BLE001 - CSV should not break the training output path.
        logging.getLogger().warning("CSV result export skipped: %s", exc)

    output_paths = derive_output_paths(config)
    return {
        "experiment_id": trainer.experiment_summary_dict.get("experiment_id"),
        "model": unified_config["model"],
        "dataset": unified_config["dataset"],
        "scenario": unified_config.get("scenario"),
        "result_dir": str(Path(config["result_file_name"]).parent),
        "summary_path": output_paths["experiment_summary_json"],
        "result_path": output_paths["experiment_result_json"],
        "csv_path": output_paths["csv"],
        "experiment_mode": trainer.experiment_summary_dict.get("experiment_mode"),
        "scenario_tags": trainer.experiment_summary_dict.get("scenario_tags", []),
        "active_attacks": trainer.experiment_summary_dict.get("active_attacks", []),
        "active_defenses": trainer.experiment_summary_dict.get("active_defenses", []),
        "active_privacy_metrics": trainer.experiment_summary_dict.get(
            "active_privacy_metrics", []
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    unified_config = load_json(config_path)
    capabilities = load_json(CAPABILITY_PATH)
    errors, warnings = validate_config(
        unified_config,
        capabilities,
        strict_validation=args.strict_validation,
        allow_blocked_model=args.allow_blocked_model,
    )
    if errors:
        print(json.dumps({"ok": False, "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
        return 2

    fedvlr_config = build_fedvlr_config(unified_config, capabilities)
    validation_payload = {
        "ok": True,
        "warnings": warnings,
        "model": unified_config.get("model"),
        "dataset": unified_config.get("dataset"),
        "scenario": unified_config.get("scenario"),
        "mapped_config": fedvlr_config,
    }

    if args.validate_only or args.dry_run:
        print(json.dumps(validation_payload, ensure_ascii=False, indent=2, default=str))
        return 0

    run_payload = run_experiment(unified_config, fedvlr_config)
    print(
        json.dumps(
            {
                "ok": True,
                "warnings": warnings,
                "experiment": run_payload,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
