from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = ROOT / "configs" / "model_attack_defense_capabilities.json"

STATUS_SUPPORTED = "supported"
STATUS_PARTIAL = "partial"
STATUS_UNSUPPORTED = "unsupported"
STATUS_ADAPTER_REQUIRED = "adapter_required"
STATUS_NOT_TESTED = "not_tested"

VALID_STATUSES = {
    STATUS_SUPPORTED,
    STATUS_PARTIAL,
    STATUS_UNSUPPORTED,
    STATUS_ADAPTER_REQUIRED,
    STATUS_NOT_TESTED,
}

MODEL_ALIASES = {
    "MMMGCN": "MMGCN",
    "MultiModalMGCN": "MMGCN",
    "MGCN-multimodal": "MMGCN",
    "MGCN_multimodal": "MMGCN",
    "MGCNMultimodal": "MMGCN",
}

CAPABILITY_EVIDENCE_FILES = {
    "baseline_training": "models/{model_file}.py",
    "dataset_support": "configs/model_attack_defense_capabilities.json",
    "topk_export": "utils/topk_evaluator.py",
    "participant_update_capture": "utils/federated/trainer.py",
    "item_embedding_access": "models/{model_file}.py",
    "score_user_item_pairs": "privacy_eval/export_membership_pair_scores.py",
    "target_rank_summary": "utils/federated/trainer.py",
    "target_interaction_injection": "attacks/target_interaction_injection.py",
    "membership_inference_rank_proxy": "privacy_eval/export_membership_pair_scores.py",
    "membership_inference_checkpoint_score": "privacy_eval/export_membership_pair_scores.py",
    "update_leakage_probe": "privacy_eval/update_leakage_risk_probe.py",
    "interaction_reconstruction_probe": "privacy_eval/interaction_reconstruction_probe.py",
    "robust_aggregation": "defenses/robust_defense.py",
    "dp_noise": "defenses/dp_noise_defense.py",
    "secure_aggregation_sim": "defenses/secure_aggregation_sim.py",
    "v3_artifact_export": "scripts/export_security_artifact_v3.py",
}

FEDAVG_AMAZON_EVIDENCE = {
    "target_rank_summary": "outputs/results/FedAvg/AMAZON_BEAUTY_POC/AmazonBeautyPOCTargetPromotionV25Smoke/target_rank_comparison.json",
    "target_interaction_injection": "outputs/results/FedAvg/AMAZON_BEAUTY_POC/AmazonBeautyPOCTargetPromotionV25Smoke/target_interaction_plan.json",
    "membership_inference_rank_proxy": "outputs/results/FedAvg/AMAZON_BEAUTY_POC/AmazonBeautyPOCTargetPromotionV25Smoke/membership_score_summary.json",
    "update_leakage_probe": "outputs/results/FedAvg/AMAZON_BEAUTY_POC/AmazonBeautyPOCSecuritySmoke/update_leakage_risk_summary.json",
    "interaction_reconstruction_probe": "outputs/results/FedAvg/AMAZON_BEAUTY_POC/AmazonBeautyPOCInteractionReconstructionV25Smoke/interaction_reconstruction_summary.json",
    "v3_artifact_export": "outputs/showcase_artifacts/amazon_beauty_poc_security_v3/frontend_summary.json",
}


def _load_capabilities(path: Path = CAPABILITY_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _records(capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    payload = capabilities if capabilities is not None else _load_capabilities()
    return {
        str(item.get("name")): item
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }


def resolve_model_name(model_name: str) -> Dict[str, Any]:
    requested = str(model_name)
    canonical = MODEL_ALIASES.get(requested, requested)
    return {
        "requested_model": requested,
        "canonical_model": canonical,
        "is_alias": requested != canonical,
        "alias_target": canonical if requested != canonical else None,
    }


def _model_file(canonical_model: str) -> str:
    return canonical_model.lower()


def _import_status(canonical_model: str) -> Dict[str, Any]:
    model_file = _model_file(canonical_model)
    module_path = "models.{}".format(model_file)
    status: Dict[str, Any] = {
        "model_file": model_file,
        "model_module": module_path,
        "model_import_ok": False,
        "trainer_import_ok": False,
        "model_error": None,
        "trainer_error": None,
    }
    try:
        module = importlib.import_module(module_path)
        getattr(module, canonical_model)
        status["model_import_ok"] = True
    except Exception as exc:  # noqa: BLE001 - status probe only.
        status["model_error"] = str(exc)
        return status

    try:
        module = importlib.import_module(module_path)
        getattr(module, "{}Trainer".format(canonical_model))
        status["trainer_import_ok"] = True
    except Exception as exc:  # noqa: BLE001 - status probe only.
        status["trainer_error"] = str(exc)
    return status


def _registry_record(canonical_model: str, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _records(capabilities).get(canonical_model, {})


def _dataset_supported(record: Dict[str, Any], dataset: str) -> bool:
    supported = record.get("supported_datasets") or []
    return bool(dataset in supported) if supported else False


def _base_context(model_name: str, dataset: str, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved = resolve_model_name(model_name)
    canonical = resolved["canonical_model"]
    record = _registry_record(canonical, capabilities)
    imports = _import_status(canonical)
    context = {
        **resolved,
        **imports,
        "dataset": dataset,
        "registry_record": record,
        "registry_status": record.get("compatibility_status"),
        "security_status": record.get("security_status"),
        "dataset_supported": _dataset_supported(record, dataset),
        "supports_multi_defense": bool(record.get("supports_multi_defense")),
    }
    return context


def _evidence_file(capability: str, canonical_model: str, dataset: str) -> str:
    if canonical_model == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
        if capability in FEDAVG_AMAZON_EVIDENCE:
            return FEDAVG_AMAZON_EVIDENCE[capability]
    template = CAPABILITY_EVIDENCE_FILES.get(capability, "configs/model_attack_defense_capabilities.json")
    return template.format(model_file=_model_file(canonical_model))


def _result(status: str, reason: str, evidence_file: str, context: Dict[str, Any]) -> Dict[str, Any]:
    if status not in VALID_STATUSES:
        status = STATUS_NOT_TESTED
    return {
        "status": status,
        "reason": reason,
        "evidence_file": evidence_file,
        "canonical_model": context["canonical_model"],
        "requested_model": context["requested_model"],
        "dataset": context["dataset"],
    }


def _adapter_needed_reason(context: Dict[str, Any]) -> Optional[str]:
    if context.get("is_alias") and not context.get("model_import_ok"):
        return "alias {} maps to {}, but the canonical model is not importable".format(
            context["requested_model"], context["canonical_model"]
        )
    if not context.get("registry_record"):
        return "model is not registered in model_attack_defense_capabilities.json"
    if context.get("registry_status") in {"blocked", "adapter_required"}:
        return "model registry marks this model as {}".format(context.get("registry_status"))
    if not context.get("model_import_ok"):
        return "model import failed: {}".format(context.get("model_error"))
    if not context.get("trainer_import_ok"):
        return "federated trainer import failed: {}".format(context.get("trainer_error"))
    return None


def _baseline_level(context: Dict[str, Any]) -> str:
    status = str(context.get("registry_status"))
    if status in {"validated", "showcase_ready"}:
        return STATUS_SUPPORTED
    if status in {"not_validated", "experimental"}:
        return STATUS_PARTIAL
    return STATUS_NOT_TESTED


def describe_security_capability(
    model_name: str,
    dataset: str,
    capability: str,
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = _base_context(model_name, dataset, capabilities)
    evidence = _evidence_file(capability, context["canonical_model"], dataset)
    adapter_reason = _adapter_needed_reason(context)

    if capability == "secure_aggregation_sim":
        return _result(
            STATUS_PARTIAL,
            "secure aggregation is a simulation/demo path and not a production cryptographic protocol",
            evidence,
            context,
        )

    if adapter_reason:
        return _result(STATUS_ADAPTER_REQUIRED, adapter_reason, evidence, context)

    if capability == "dataset_support":
        if context["dataset_supported"]:
            return _result(
                _baseline_level(context),
                "dataset is listed for this model in the capability registry",
                evidence,
                context,
            )
        return _result(
            STATUS_UNSUPPORTED,
            "dataset is not listed for this model in the capability registry",
            evidence,
            context,
        )

    if not context["dataset_supported"]:
        return _result(
            STATUS_UNSUPPORTED,
            "dataset is not listed for this model in the capability registry",
            evidence,
            context,
        )

    if capability == "baseline_training":
        return _result(
            _baseline_level(context),
            "model and federated trainer import successfully; validation level comes from registry",
            evidence,
            context,
        )

    if capability in {"topk_export", "participant_update_capture", "membership_inference_rank_proxy"}:
        return _result(
            _baseline_level(context),
            "shared federated trainer/evaluator path is available for this model",
            evidence,
            context,
        )

    if capability == "item_embedding_access":
        if context["canonical_model"] in {"FedAvg", "FedRAP", "MMFedAvg", "MMFedRAP"}:
            return _result(
                _baseline_level(context),
                "item-like parameters are known from existing FedAvg/RAP family probes",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "model imports through the federated path, but item-like parameter names need per-model validation",
            evidence,
            context,
        )

    if capability == "score_user_item_pairs":
        if context["canonical_model"] == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return _result(
                STATUS_SUPPORTED,
                "unmasked target rank/score evidence exists for FedAvg Amazon V2.5",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "pair score export can use unmasked rank or TopK rank proxy; exact checkpoint scoring is model-specific",
            evidence,
            context,
        )

    if capability == "target_rank_summary":
        if context["canonical_model"] == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return _result(
                STATUS_SUPPORTED,
                "target_rank_comparison exists for FedAvg Amazon V2.5; do not generalize effect to other models",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "target-rank diagnostics are implemented in the shared federated trainer, but effect is not validated for this model",
            evidence,
            context,
        )

    if capability == "target_interaction_injection":
        if context["canonical_model"] == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return _result(
                STATUS_SUPPORTED,
                "hook_active target interaction plan exists for the FedAvg Amazon V2.5 run",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "in-memory hook is configurable, but target promotion effectiveness is model/dataset-specific",
            evidence,
            context,
        )

    if capability == "membership_inference_checkpoint_score":
        if context["canonical_model"] in {"FedAvg", "FedRAP"}:
            return _result(
                STATUS_PARTIAL,
                "checkpoint scoring supports FedAvg/FedRAP-style parameter pickles when the saved format matches",
                evidence,
                context,
            )
        return _result(
            STATUS_ADAPTER_REQUIRED,
            "checkpoint score export needs a model-specific scorer adapter for this model",
            evidence,
            context,
        )

    if capability == "update_leakage_probe":
        if context["canonical_model"] == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return _result(
                STATUS_SUPPORTED,
                "real participant_params update leakage summary exists for Amazon security smoke",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "participant updates can be captured, but risk magnitude is run-specific",
            evidence,
            context,
        )

    if capability == "interaction_reconstruction_probe":
        if context["canonical_model"] == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return _result(
                STATUS_SUPPORTED,
                "real participant_params reconstruction summary exists for FedAvg Amazon V2.5",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "candidate reconstruction depends on per-model item-like parameter names",
            evidence,
            context,
        )

    if capability == "robust_aggregation":
        if context["supports_multi_defense"] and _baseline_level(context) == STATUS_SUPPORTED:
            return _result(
                STATUS_SUPPORTED,
                "model registry marks robust/multi-defense path as available and validated/showcase-ready",
                evidence,
                context,
            )
        if context["supports_multi_defense"]:
            return _result(
                STATUS_PARTIAL,
                "robust defense path is configurable but not validated for this model",
                evidence,
                context,
            )
        return _result(
            STATUS_ADAPTER_REQUIRED,
            "model does not expose the federated update path required by robust aggregation",
            evidence,
            context,
        )

    if capability == "dp_noise":
        return _result(
            STATUS_PARTIAL,
            "central DP-style noise is configurable on update tensors, but formal_accountant=false",
            evidence,
            context,
        )

    if capability == "v3_artifact_export":
        if context["canonical_model"] == "FedAvg" and dataset == "AMAZON_BEAUTY_POC":
            return _result(
                STATUS_SUPPORTED,
                "V3 artifact bundle exists for FedAvg Amazon",
                evidence,
                context,
            )
        return _result(
            STATUS_PARTIAL,
            "V3 exporter can summarize capability status, but model-specific evidence is not necessarily available",
            evidence,
            context,
        )

    return _result(STATUS_NOT_TESTED, "capability has no adapter rule yet", evidence, context)


def supports_topk_export(model_name: str, dataset: str = "KU") -> Dict[str, Any]:
    return describe_security_capability(model_name, dataset, "topk_export")


def supports_participant_updates(model_name: str, dataset: str = "KU") -> Dict[str, Any]:
    return describe_security_capability(model_name, dataset, "participant_update_capture")


def supports_item_embeddings(model_name: str, dataset: str = "KU") -> Dict[str, Any]:
    return describe_security_capability(model_name, dataset, "item_embedding_access")


def supports_score_pairs(model_name: str, dataset: str = "KU") -> Dict[str, Any]:
    return describe_security_capability(model_name, dataset, "score_user_item_pairs")


def supports_target_rank(model_name: str, dataset: str = "KU") -> Dict[str, Any]:
    return describe_security_capability(model_name, dataset, "target_rank_summary")


def supports_privacy_probe(model_name: str, probe_name: str, dataset: str = "KU") -> Dict[str, Any]:
    probe_to_capability = {
        "membership_inference_rank_proxy": "membership_inference_rank_proxy",
        "membership_inference_checkpoint_score": "membership_inference_checkpoint_score",
        "update_leakage_probe": "update_leakage_probe",
        "interaction_reconstruction_probe": "interaction_reconstruction_probe",
    }
    return describe_security_capability(model_name, dataset, probe_to_capability.get(probe_name, probe_name))


def supports_robust_defense(model_name: str, dataset: str = "KU") -> Dict[str, Any]:
    return describe_security_capability(model_name, dataset, "robust_aggregation")
