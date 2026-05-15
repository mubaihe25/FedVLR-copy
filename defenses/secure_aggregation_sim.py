"""Simulation-only secure aggregation mask accounting.

This module demonstrates the basic pairwise-mask cancellation idea used in
secure aggregation protocols. It does not implement authentication, key
agreement, dropout handling, cryptographic commitments, or a privacy proof.
The defense is read-only and returns participant updates unchanged.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class SecureAggregationSimDefense(BaseDefense):
    """Read-only simulation of pairwise mask cancellation before aggregation."""

    def __init__(
        self,
        name: str = "secure_aggregation_sim",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        config = config or {}
        self.seed = int(config.get("secure_agg_seed", config.get("seed", 42)))
        self.mask_std = float(config.get("secure_agg_mask_std", 1.0))
        self.max_elements = int(config.get("secure_agg_max_elements", 10000))
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _flatten_value(self, value: Any, chunks: List[torch.Tensor]) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            tensor = tensor.float().reshape(-1).cpu()
            if tensor.numel() > 0:
                chunks.append(tensor)
            return
        if isinstance(value, Number) and not isinstance(value, bool):
            chunks.append(torch.tensor([float(value)], dtype=torch.float32))
            return
        if isinstance(value, dict):
            for key in sorted(value.keys(), key=str):
                self._flatten_value(value[key], chunks)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._flatten_value(item, chunks)

    def _flatten_client_update(self, value: Any) -> Optional[torch.Tensor]:
        chunks: List[torch.Tensor] = []
        try:
            self._flatten_value(value, chunks)
        except Exception:
            return None
        if not chunks:
            return None
        vector = torch.cat(chunks)
        if vector.numel() == 0:
            return None
        return vector

    def _simulate_masks(self, vectors: List[torch.Tensor]) -> Dict[str, Any]:
        participant_count = len(vectors)
        if participant_count < 2:
            return {
                "mask_pair_count": 0,
                "simulated_tensor_element_count": int(vectors[0].numel()) if vectors else 0,
                "aggregate_mask_residual_norm": 0.0,
                "max_pair_mask_norm": 0.0,
                "warning": "need at least two compatible clients for pairwise mask simulation",
            }

        min_elements = min(int(vector.numel()) for vector in vectors)
        truncated = min_elements > self.max_elements
        element_count = min(min_elements, self.max_elements)
        residual_masks = [
            torch.zeros(element_count, dtype=torch.float32) for _ in range(participant_count)
        ]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        max_pair_mask_norm = 0.0
        mask_pair_count = 0

        for left_index in range(participant_count):
            for right_index in range(left_index + 1, participant_count):
                pair_mask = torch.normal(
                    mean=0.0,
                    std=self.mask_std,
                    size=(element_count,),
                    generator=generator,
                )
                residual_masks[left_index] += pair_mask
                residual_masks[right_index] -= pair_mask
                pair_norm = float(torch.linalg.vector_norm(pair_mask).item())
                max_pair_mask_norm = max(max_pair_mask_norm, pair_norm)
                mask_pair_count += 1

        aggregate_residual = torch.stack(residual_masks, dim=0).sum(dim=0)
        residual_norm = float(torch.linalg.vector_norm(aggregate_residual).item())
        return {
            "mask_pair_count": int(mask_pair_count),
            "simulated_tensor_element_count": int(element_count),
            "aggregate_mask_residual_norm": residual_norm,
            "max_pair_mask_norm": max_pair_mask_norm,
            "warning": (
                "simulation truncated to secure_agg_max_elements"
                if truncated
                else None
            ),
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        vectors: List[torch.Tensor] = []
        skipped_client_ids: List[str] = []
        participant_count = len(participant_params) if isinstance(participant_params, dict) else 0

        if isinstance(participant_params, dict):
            for client_id, client_update in participant_params.items():
                vector = self._flatten_client_update(client_update)
                if vector is None:
                    skipped_client_ids.append(str(client_id))
                else:
                    vectors.append(vector)

        simulation = self._simulate_masks(vectors) if vectors else {
            "mask_pair_count": 0,
            "simulated_tensor_element_count": 0,
            "aggregate_mask_residual_norm": None,
            "max_pair_mask_norm": None,
            "warning": "no compatible tensor-like client updates found",
        }
        warnings = [simulation["warning"]] if simulation.get("warning") else []

        summary = {
            "defense_type": "secure_aggregation_sim",
            "simulation_only": True,
            "is_read_only": True,
            "mutates_participant_params": False,
            "participant_count": int(participant_count),
            "compatible_client_count": int(len(vectors)),
            "skipped_client_count": int(len(skipped_client_ids)),
            "skipped_client_ids": skipped_client_ids,
            "mask_pair_count": simulation.get("mask_pair_count"),
            "simulated_tensor_element_count": simulation.get("simulated_tensor_element_count"),
            "aggregate_mask_residual_norm": simulation.get("aggregate_mask_residual_norm"),
            "max_pair_mask_norm": simulation.get("max_pair_mask_norm"),
            "secure_agg_mask_std": self.mask_std,
            "secure_agg_seed": self.seed,
            "note": "simulation only; not a production cryptographic secure aggregation protocol",
            "warnings": warnings,
        }
        if summary["aggregate_mask_residual_norm"] is not None:
            summary["aggregate_mask_cancelled"] = (
                float(summary["aggregate_mask_residual_norm"]) <= 1e-6
                or math.isclose(float(summary["aggregate_mask_residual_norm"]), 0.0, abs_tol=1e-6)
            )
        else:
            summary["aggregate_mask_cancelled"] = False

        self.last_round_output = summary
        self.history.append(summary)
        round_state.setdefault("defense_outputs", {})[self.name] = summary
        return participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "defense_type": "secure_aggregation_sim",
                "simulation_only": True,
                "num_rounds": 0,
                "note": "simulation only; not a production cryptographic secure aggregation protocol",
            }
        latest = dict(self.history[-1])
        latest["num_rounds"] = len(self.history)
        latest["rounds_with_compatible_clients"] = sum(
            1 for item in self.history if int(item.get("compatible_client_count", 0)) > 0
        )
        latest["total_mask_pair_count"] = int(
            sum(int(item.get("mask_pair_count") or 0) for item in self.history)
        )
        return latest


register_defense("secure_aggregation_sim", SecureAggregationSimDefense)
register_defense("secure_agg_sim", SecureAggregationSimDefense)
register_defense("secureaggregationsim", SecureAggregationSimDefense)
