"""Gradient leakage risk probe for image or multimodal updates.

This module provides a lightweight risk summary over gradient-like tensors. It
does not reconstruct original FedVLR images and does not implement a full
DLG/InvertingGrad pipeline. Future work can feed real image encoder gradients
or add a controlled optimizer demo on top of this probe.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


class GradientLeakageProbe(BasePrivacyMetric):
    """Summarize reversible-risk signals from gradient-like tensors."""

    def __init__(
        self,
        name: str = "gradient_leakage_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.sensitive_modality = str(
            self._config_get(config, "sensitive_modality", "unknown")
        ).lower()
        self.norm_reference = float(self._config_get(config, "norm_reference", 10.0))
        self.energy_reference = float(self._config_get(config, "energy_reference", 100.0))
        self.zero_threshold = float(self._config_get(config, "zero_threshold", 1e-12))
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _config_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                try:
                    value = getter(key)
                except Exception:
                    return default
                return default if value is None else value
        return getattr(config, key, default)

    def _candidate_sources(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> List[Any]:
        sources: List[Any] = []
        for key in (
            "gradient_tensors",
            "synthetic_gradient_tensors",
            "image_gradient_tensors",
        ):
            value = self._config_get(self.config, key, None)
            if value is not None:
                sources.append(value)

        for source in (round_state, aggregation_result):
            if not isinstance(source, dict):
                continue
            for key in (
                "gradient_tensors",
                "synthetic_gradient_tensors",
                "image_gradient_tensors",
                "gradient_leakage",
                "gradient_leakage_probe",
            ):
                value = source.get(key)
                if value is not None:
                    sources.append(value)
            for nested_key in ("privacy_probe_inputs", "privacy_metric_inputs"):
                nested = source.get(nested_key)
                if not isinstance(nested, dict):
                    continue
                for probe_key in ("gradient_leakage", "gradient_leakage_probe"):
                    value = nested.get(probe_key)
                    if value is not None:
                        sources.append(value)

        if not sources and participant_params:
            sources.append(participant_params)
        return sources

    def _walk_tensors(self, value: Any, path: str = "root") -> Tuple[List[Tuple[str, torch.Tensor]], int]:
        if value is None:
            return [], 1

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            if not torch.is_floating_point(tensor):
                return [], 1
            if tensor.numel() == 0:
                return [], 1
            return [(path, tensor.float())], 0

        if isinstance(value, dict):
            tensors: List[Tuple[str, torch.Tensor]] = []
            skipped = 0
            for key in sorted(value.keys(), key=lambda item: str(item)):
                child_tensors, child_skipped = self._walk_tensors(
                    value[key], "{}.{}".format(path, key)
                )
                tensors.extend(child_tensors)
                skipped += child_skipped
            return tensors, skipped

        if isinstance(value, (list, tuple)):
            tensors = []
            skipped = 0
            for index, item in enumerate(value):
                child_tensors, child_skipped = self._walk_tensors(
                    item, "{}[{}]".format(path, index)
                )
                tensors.extend(child_tensors)
                skipped += child_skipped
            return tensors, skipped

        return [], 1

    def _infer_modality(self, tensor_paths: List[str]) -> str:
        if self.sensitive_modality in {"image", "multimodal", "unknown"}:
            if self.sensitive_modality != "unknown":
                return self.sensitive_modality
        joined_paths = " ".join(path.lower() for path in tensor_paths)
        if any(token in joined_paths for token in ("image", "img", "visual", "vision")):
            return "image"
        if any(token in joined_paths for token in ("fusion", "multimodal", "multi_modal")):
            return "multimodal"
        return "unknown"

    def _risk_level(self, leakage_risk_score: float) -> str:
        if leakage_risk_score >= 0.66:
            return "high"
        if leakage_risk_score >= 0.33:
            return "medium"
        return "low"

    def evaluate_gradients(
        self,
        gradient_source: Any,
        sensitive_modality: Optional[str] = None,
    ) -> Dict[str, Any]:
        tensors, skipped_value_count = self._walk_tensors(gradient_source)
        tensor_paths = [path for path, _ in tensors]
        tensor_count = len(tensors)

        if tensor_count == 0:
            result = self._empty_result(
                sensitive_modality=sensitive_modality or self.sensitive_modality,
                fallback_reason="no_supported_gradient_tensors",
                skipped_value_count=skipped_value_count,
            )
            return result

        total_energy = 0.0
        total_numel = 0
        zero_like_count = 0
        max_abs = 0.0

        for _, tensor in tensors:
            tensor = tensor.detach().float()
            abs_tensor = torch.abs(tensor)
            total_energy += float(torch.sum(tensor * tensor).item())
            total_numel += int(tensor.numel())
            zero_like_count += int(torch.sum(abs_tensor <= self.zero_threshold).item())
            max_abs = max(max_abs, float(torch.max(abs_tensor).item()))

        gradient_norm = float(math.sqrt(max(total_energy, 0.0)))
        gradient_energy = float(total_energy)
        sparsity = float(zero_like_count / total_numel) if total_numel else 1.0
        density = 1.0 - sparsity
        modality = sensitive_modality or self._infer_modality(tensor_paths)
        modality_boost = 0.10 if modality in {"image", "multimodal"} else 0.0
        norm_component = min(
            1.0,
            math.log1p(gradient_norm) / max(math.log1p(max(self.norm_reference, 1e-12)), 1e-12),
        )
        energy_component = min(
            1.0,
            math.log1p(gradient_energy) / max(math.log1p(max(self.energy_reference, 1e-12)), 1e-12),
        )
        leakage_risk_score = min(
            1.0,
            0.40 * norm_component + 0.35 * density + 0.15 * energy_component + modality_boost,
        )
        risk_level = self._risk_level(leakage_risk_score)

        result = {
            "probe_type": "gradient_leakage",
            "gradient_norm": gradient_norm,
            "gradient_energy": gradient_energy,
            "tensor_count": tensor_count,
            "total_numel": total_numel,
            "gradient_sparsity": sparsity,
            "max_abs_gradient": max_abs,
            "sensitive_modality": modality,
            "leakage_risk_score": float(leakage_risk_score),
            "risk_level": risk_level,
            "skipped_value_count": skipped_value_count,
            "note": "demo probe, not full FedVLR image reconstruction",
        }
        result["gradient_leakage"] = {
            key: result[key]
            for key in (
                "gradient_norm",
                "gradient_energy",
                "tensor_count",
                "sensitive_modality",
                "leakage_risk_score",
                "risk_level",
            )
        }
        result["privacy_risk_summary"] = {
            "gradient_leakage": {
                "risk_level": risk_level,
                "leakage_risk_score": float(leakage_risk_score),
                "sensitive_modality": modality,
            }
        }
        result["privacy_attack_summaries"] = {
            "gradient_leakage": dict(result["gradient_leakage"])
        }
        return result

    def _empty_result(
        self,
        sensitive_modality: str = "unknown",
        fallback_reason: str = "no_supported_gradient_tensors",
        skipped_value_count: int = 0,
    ) -> Dict[str, Any]:
        result = {
            "probe_type": "gradient_leakage",
            "gradient_norm": None,
            "gradient_energy": None,
            "tensor_count": 0,
            "total_numel": 0,
            "gradient_sparsity": None,
            "max_abs_gradient": None,
            "sensitive_modality": sensitive_modality,
            "leakage_risk_score": 0.0,
            "risk_level": "low",
            "skipped_value_count": skipped_value_count,
            "fallback_reason": fallback_reason,
            "note": "demo probe, not full FedVLR image reconstruction",
        }
        result["gradient_leakage"] = {
            "gradient_norm": None,
            "gradient_energy": None,
            "tensor_count": 0,
            "sensitive_modality": sensitive_modality,
            "leakage_risk_score": 0.0,
            "risk_level": "low",
        }
        result["privacy_risk_summary"] = {
            "gradient_leakage": {
                "risk_level": "low",
                "leakage_risk_score": 0.0,
                "sensitive_modality": sensitive_modality,
            }
        }
        result["privacy_attack_summaries"] = {
            "gradient_leakage": dict(result["gradient_leakage"])
        }
        return result

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        sources = self._candidate_sources(round_state, participant_params, aggregation_result)
        if not sources:
            round_result = self._empty_result(fallback_reason="missing_gradient_inputs")
        else:
            if len(sources) == 1:
                source = sources[0]
            else:
                source = {"source_{}".format(index): value for index, value in enumerate(sources)}
            round_result = self.evaluate_gradients(source)
        self.history.append(round_result)
        return round_result

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            summary = self._empty_result(fallback_reason="no_probe_history")
            summary["num_rounds"] = 0
            return summary

        latest = dict(self.history[-1])
        latest["num_rounds"] = len(self.history)
        return latest


def run_synthetic_smoke() -> Dict[str, Any]:
    """Run a tiny synthetic gradient leakage risk check."""
    synthetic_gradient = torch.linspace(0.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 8, 8)
    probe = GradientLeakageProbe(
        config={
            "sensitive_modality": "image",
            "norm_reference": 4.0,
            "energy_reference": 16.0,
        }
    )
    result = probe.evaluate_gradients({"image_encoder.grad": synthetic_gradient})
    assert result["tensor_count"] == 1
    assert result["gradient_norm"] is not None
    assert result["gradient_energy"] is not None
    assert result["risk_level"] in {"low", "medium", "high"}
    return result


if __name__ == "__main__":
    print(json.dumps(run_synthetic_smoke(), indent=2, sort_keys=True))


register_privacy_metric("gradient_leakage_probe", GradientLeakageProbe)
register_privacy_metric("gradient_leakage", GradientLeakageProbe)
register_privacy_metric("image_leakage_probe", GradientLeakageProbe)
register_privacy_metric("gradientleakageprobe", GradientLeakageProbe)
