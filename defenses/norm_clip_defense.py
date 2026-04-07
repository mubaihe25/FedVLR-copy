"""Lightweight active defense that clips client update norms before aggregation.

This module is intentionally simple:
- it observes all participant updates before aggregation
- it rescales only updates whose norm exceeds a configured threshold
- it does not replace the aggregation algorithm
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class NormClipDefense(BaseDefense):
    """Clip oversized client updates to a fixed norm threshold."""

    def __init__(
        self,
        name: str = "norm_clip",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.defense_clip_norm = float((config or {}).get("defense_clip_norm", 5.0))
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _sum_squared_norm(self, value: Any) -> float:
        if value is None:
            return 0.0

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            tensor = tensor.float()
            if tensor.numel() == 0:
                return 0.0
            return float(torch.sum(tensor * tensor).item())

        if isinstance(value, Number) and not isinstance(value, bool):
            scalar = float(value)
            return scalar * scalar

        if isinstance(value, dict):
            return sum(self._sum_squared_norm(item) for item in value.values())

        if isinstance(value, (list, tuple, set)):
            return sum(self._sum_squared_norm(item) for item in value)

        return 0.0

    def _safe_norm(self, value: Any) -> Optional[float]:
        try:
            squared_norm = self._sum_squared_norm(value)
        except Exception:
            return None
        if squared_norm <= 0:
            return None
        return float(math.sqrt(squared_norm))

    def _scale_value(self, value: Any, scale_factor: float) -> Tuple[Any, int]:
        if value is None:
            return value, 0

        if torch.is_tensor(value):
            return value * scale_factor, 1

        if isinstance(value, Number) and not isinstance(value, bool):
            return float(value) * scale_factor, 1

        if isinstance(value, dict):
            scaled_dict: Dict[Any, Any] = {}
            touched_count = 0
            for key, item in value.items():
                scaled_item, item_touched = self._scale_value(item, scale_factor)
                scaled_dict[key] = scaled_item
                touched_count += item_touched
            return scaled_dict, touched_count

        if isinstance(value, list):
            scaled_list: List[Any] = []
            touched_count = 0
            for item in value:
                scaled_item, item_touched = self._scale_value(item, scale_factor)
                scaled_list.append(scaled_item)
                touched_count += item_touched
            return scaled_list, touched_count

        if isinstance(value, tuple):
            scaled_items: List[Any] = []
            touched_count = 0
            for item in value:
                scaled_item, item_touched = self._scale_value(item, scale_factor)
                scaled_items.append(scaled_item)
                touched_count += item_touched
            return tuple(scaled_items), touched_count

        return value, 0

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict):
            defense_result = {
                "clipped_clients": [],
                "clipped_client_count": 0,
                "defense_clip_norm": self.defense_clip_norm,
                "norms_before": {},
                "norms_after": {},
            }
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        updated_participant_params = dict(participant_params)
        clipped_clients: List[str] = []
        norms_before: Dict[str, float] = {}
        norms_after: Dict[str, float] = {}

        for client_id, client_update in participant_params.items():
            client_id_str = str(client_id)
            norm_before = self._safe_norm(client_update)
            if norm_before is None:
                continue
            norms_before[client_id_str] = norm_before

            if norm_before <= self.defense_clip_norm:
                norms_after[client_id_str] = norm_before
                continue

            scale_factor = self.defense_clip_norm / max(norm_before, 1e-12)
            scaled_update, _ = self._scale_value(client_update, scale_factor)
            updated_participant_params[client_id] = scaled_update
            clipped_clients.append(client_id_str)
            norm_after = self._safe_norm(scaled_update)
            norms_after[client_id_str] = (
                norm_after if norm_after is not None else self.defense_clip_norm
            )

        defense_result = {
            "clipped_clients": clipped_clients,
            "clipped_client_count": len(clipped_clients),
            "defense_clip_norm": self.defense_clip_norm,
            "norms_before": norms_before,
            "norms_after": norms_after,
        }
        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("clipped_clients", {})[self.name] = clipped_clients
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "num_rounds": 0,
                "rounds_with_clipping": 0,
                "total_clipped_clients": 0,
                "max_clipped_client_count": 0,
                "defense_clip_norm": self.defense_clip_norm,
                "avg_norm_before": None,
                "avg_norm_after": None,
            }

        clipped_counts = [int(item.get("clipped_client_count", 0)) for item in self.history]
        before_values = [
            float(value)
            for item in self.history
            for value in item.get("norms_before", {}).values()
        ]
        after_values = [
            float(value)
            for item in self.history
            for value in item.get("norms_after", {}).values()
        ]
        return {
            "num_rounds": len(self.history),
            "rounds_with_clipping": sum(1 for count in clipped_counts if count > 0),
            "total_clipped_clients": int(sum(clipped_counts)),
            "max_clipped_client_count": int(max(clipped_counts)),
            "defense_clip_norm": self.defense_clip_norm,
            "avg_norm_before": (
                float(sum(before_values) / len(before_values)) if before_values else None
            ),
            "avg_norm_after": (
                float(sum(after_values) / len(after_values)) if after_values else None
            ),
        }


register_defense("norm_clip", NormClipDefense)
register_defense("normclipdefense", NormClipDefense)
