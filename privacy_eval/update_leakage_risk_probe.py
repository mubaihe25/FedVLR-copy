"""Read-only update leakage risk probe for real participant_params.

This probe summarizes server-visible client updates. It does not save raw
participant updates and does not reconstruct images or training samples.
"""

from __future__ import annotations

import json
import math
import argparse
from statistics import mean, pstdev
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


MODALITY_TOKENS = {
    "image": ("image", "img", "visual", "vision", "v_feat", "vfeat"),
    "text": ("text", "txt", "textual", "language", "t_feat", "tfeat"),
    "id": ("embedding", "embed", "item", "user", "id"),
    "fusion": ("fusion", "router", "gate", "attention", "multi_modal", "multimodal"),
}


class UpdateLeakageRiskProbe(BasePrivacyMetric):
    """Summarize leakage-risk signals from uploaded client updates."""

    def __init__(
        self,
        name: str = "update_leakage_risk_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.norm_reference = float(self._config_get(config, "norm_reference", 10.0))
        self.energy_reference = float(self._config_get(config, "energy_reference", 100.0))
        self.zero_threshold = float(self._config_get(config, "zero_threshold", 1e-12))
        self.max_flatten_elements = int(self._config_get(config, "max_flatten_elements", 100000))
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

    def _infer_modality(self, path: str) -> str:
        normalized = path.lower()
        for modality, tokens in MODALITY_TOKENS.items():
            if any(token in normalized for token in tokens):
                return modality
        return "unknown"

    def _walk_tensors(self, value: Any, path: str = "root") -> Tuple[List[Tuple[str, torch.Tensor]], int]:
        if value is None:
            return [], 1
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            if not torch.is_floating_point(tensor) or tensor.numel() == 0:
                return [], 1
            return [(path, tensor.float().cpu())], 0
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

    def _client_stats(self, client_update: Any) -> Dict[str, Any]:
        tensors, skipped = self._walk_tensors(client_update)
        total_energy = 0.0
        total_params = 0
        zero_like = 0
        modality_energy: Dict[str, float] = {}
        modality_params: Dict[str, int] = {}
        flat_parts: List[torch.Tensor] = []
        remaining = self.max_flatten_elements

        for path, tensor in tensors:
            tensor = tensor.float()
            energy = float(torch.sum(tensor * tensor).item())
            numel = int(tensor.numel())
            zeros = int(torch.sum(torch.abs(tensor) <= self.zero_threshold).item())
            modality = self._infer_modality(path)
            total_energy += energy
            total_params += numel
            zero_like += zeros
            modality_energy[modality] = modality_energy.get(modality, 0.0) + energy
            modality_params[modality] = modality_params.get(modality, 0) + numel
            if remaining > 0:
                flat = tensor.reshape(-1)
                flat_parts.append(flat[:remaining])
                remaining -= min(remaining, flat.numel())

        norm = float(math.sqrt(max(total_energy, 0.0))) if total_params else None
        sparsity = float(zero_like / total_params) if total_params else None
        flat_vector = torch.cat(flat_parts) if flat_parts else torch.empty(0)
        return {
            "tensor_count": len(tensors),
            "skipped_value_count": skipped,
            "parameter_count": total_params,
            "energy": total_energy,
            "norm": norm,
            "sparsity": sparsity,
            "modality_energy": modality_energy,
            "modality_params": modality_params,
            "flat_vector": flat_vector,
        }

    @staticmethod
    def _cosine_diversity(vectors: List[torch.Tensor]) -> Optional[float]:
        usable = [vector.float() for vector in vectors if vector.numel() > 0]
        if len(usable) < 2:
            return None
        min_len = min(vector.numel() for vector in usable)
        if min_len <= 0:
            return None
        trimmed = [vector[:min_len] for vector in usable]
        distances: List[float] = []
        for i in range(len(trimmed)):
            for j in range(i + 1, len(trimmed)):
                left = trimmed[i]
                right = trimmed[j]
                denom = float(torch.norm(left).item() * torch.norm(right).item())
                if denom <= 0:
                    continue
                cosine = float(torch.dot(left, right).item() / denom)
                distances.append(1.0 - max(-1.0, min(1.0, cosine)))
        if not distances:
            return None
        return float(sum(distances) / len(distances))

    def _risk_level(self, score: float) -> str:
        if score >= 0.66:
            return "high"
        if score >= 0.33:
            return "medium"
        return "low"

    def evaluate_updates(self, participant_params: Any, source: str = "real_participant_params") -> Dict[str, Any]:
        if not isinstance(participant_params, dict) or not participant_params:
            return {
                "probe_type": "update_leakage_risk",
                "source": "not_available",
                "client_count": 0,
                "tensor_count": 0,
                "total_parameter_count": 0,
                "update_norm_mean": None,
                "update_norm_max": None,
                "update_norm_std": None,
                "sparsity_mean": None,
                "energy_mean": None,
                "client_update_diversity": None,
                "leakage_risk_score": 0.0,
                "risk_level": "not_available",
                "modality_risk_breakdown": {},
                "highest_risk_modality": None,
                "warnings": ["participant_params missing or unsupported"],
                "note": "risk probe only, not reconstruction",
            }

        client_stats = {
            str(client_id): self._client_stats(update)
            for client_id, update in participant_params.items()
        }
        valid_stats = [stats for stats in client_stats.values() if stats["parameter_count"] > 0]
        if not valid_stats:
            result = self.evaluate_updates({}, source="not_available")
            result["warnings"] = ["no tensor updates found in participant_params"]
            return result

        norms = [float(stats["norm"]) for stats in valid_stats if stats["norm"] is not None]
        sparsities = [float(stats["sparsity"]) for stats in valid_stats if stats["sparsity"] is not None]
        energies = [float(stats["energy"]) for stats in valid_stats]
        vectors = [stats["flat_vector"] for stats in valid_stats]
        diversity = self._cosine_diversity(vectors)

        modality_energy: Dict[str, float] = {}
        modality_params: Dict[str, int] = {}
        for stats in valid_stats:
            for modality, energy in stats["modality_energy"].items():
                modality_energy[modality] = modality_energy.get(modality, 0.0) + float(energy)
            for modality, count in stats["modality_params"].items():
                modality_params[modality] = modality_params.get(modality, 0) + int(count)

        total_energy = sum(energies)
        modality_breakdown: Dict[str, Dict[str, Any]] = {}
        for modality in sorted(set(modality_energy) | set(modality_params)):
            energy = modality_energy.get(modality, 0.0)
            params = modality_params.get(modality, 0)
            modality_score = min(
                1.0,
                math.log1p(energy) / max(math.log1p(max(self.energy_reference, 1e-12)), 1e-12),
            )
            if modality in {"image", "text", "fusion"}:
                modality_score = min(1.0, modality_score + 0.10)
            modality_breakdown[modality] = {
                "tensor_parameter_count": params,
                "energy": float(energy),
                "energy_share": float(energy / total_energy) if total_energy > 0 else None,
                "risk_score": float(modality_score),
                "risk_level": self._risk_level(modality_score),
            }

        highest_risk_modality = None
        if modality_breakdown:
            highest_risk_modality = max(
                modality_breakdown.items(),
                key=lambda item: item[1].get("risk_score", 0.0),
            )[0]

        norm_mean = float(mean(norms)) if norms else None
        norm_max = float(max(norms)) if norms else None
        norm_std = float(pstdev(norms)) if len(norms) > 1 else 0.0 if norms else None
        sparsity_mean = float(mean(sparsities)) if sparsities else None
        energy_mean = float(mean(energies)) if energies else None
        density = 1.0 - sparsity_mean if sparsity_mean is not None else 0.0
        norm_component = (
            min(1.0, math.log1p(norm_mean) / max(math.log1p(max(self.norm_reference, 1e-12)), 1e-12))
            if norm_mean is not None
            else 0.0
        )
        energy_component = (
            min(1.0, math.log1p(energy_mean) / max(math.log1p(max(self.energy_reference, 1e-12)), 1e-12))
            if energy_mean is not None
            else 0.0
        )
        diversity_component = min(1.0, diversity) if diversity is not None else 0.0
        modality_component = max(
            [entry["risk_score"] for entry in modality_breakdown.values()],
            default=0.0,
        )
        risk_score = min(
            1.0,
            0.35 * norm_component
            + 0.25 * density
            + 0.20 * energy_component
            + 0.10 * diversity_component
            + 0.10 * modality_component,
        )

        return {
            "probe_type": "update_leakage_risk",
            "source": source,
            "client_count": len(valid_stats),
            "tensor_count": int(sum(stats["tensor_count"] for stats in valid_stats)),
            "total_parameter_count": int(sum(stats["parameter_count"] for stats in valid_stats)),
            "update_norm_mean": norm_mean,
            "update_norm_max": norm_max,
            "update_norm_std": norm_std,
            "sparsity_mean": sparsity_mean,
            "energy_mean": energy_mean,
            "client_update_diversity": diversity,
            "leakage_risk_score": float(risk_score),
            "risk_level": self._risk_level(risk_score),
            "modality_risk_breakdown": modality_breakdown,
            "highest_risk_modality": highest_risk_modality,
            "warnings": [],
            "note": "risk probe only, not reconstruction",
        }

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        del round_state, aggregation_result
        summary = self.evaluate_updates(participant_params, source="real_participant_params")
        self.history.append(summary)
        return summary

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        del experiment_metadata
        valid = [entry for entry in self.history if entry.get("source") != "not_available"]
        if not valid:
            summary = self.evaluate_updates({}, source="not_available")
            summary["num_rounds"] = 0
            return summary
        latest = dict(valid[-1])
        latest["num_rounds"] = len(valid)
        latest["source"] = "real_participant_params"
        return latest


def run_synthetic_smoke() -> Dict[str, Any]:
    participant_params = {
        "client_1": {
            "image_encoder.weight": torch.linspace(0, 1, steps=16).reshape(4, 4),
            "item_embedding.weight": torch.ones(8),
        },
        "client_2": {
            "image_encoder.weight": torch.linspace(1, 0, steps=16).reshape(4, 4),
            "fusion.gate": torch.tensor([0.1, 0.2, 0.3]),
        },
    }
    probe = UpdateLeakageRiskProbe(
        config={"norm_reference": 4.0, "energy_reference": 16.0}
    )
    result = probe.evaluate_updates(participant_params, source="synthetic")
    assert result["client_count"] == 2
    assert result["tensor_count"] >= 2
    assert result["risk_level"] in {"low", "medium", "high"}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synthetic update leakage risk smoke. Real runs use the privacy metric hook."
    )
    parser.add_argument("--smoke", action="store_true", help="Run the synthetic smoke.")
    parser.add_argument("--output-json", help="Optional output summary path.")
    return parser


def write_json(path: str, payload: Dict[str, Any]) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    summary = run_synthetic_smoke()
    if args.output_json:
        write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


register_privacy_metric("update_leakage_risk_probe", UpdateLeakageRiskProbe)
register_privacy_metric("update_leakage_risk", UpdateLeakageRiskProbe)
register_privacy_metric("updateleakageriskprobe", UpdateLeakageRiskProbe)
