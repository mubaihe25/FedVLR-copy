"""Round-level privacy metric that observes client update norms.

This metric is intentionally read-only:
- it only inspects uploaded client updates before aggregation
- it never mutates participant_params
- it never changes training, aggregation, or optimizer behavior
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional

import torch

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


class ClientUpdateNormMetric(BasePrivacyMetric):
    """Collect simple norm statistics for uploaded client updates.

    Expected participant_params shape:
    - FedRAP: {client_id: {"item_commonality.weight": tensor}}
    - MMFedRAP: {client_id: {"fusion.*": grad_tensor, "item_commonality.weight": tensor}}

    The implementation is deliberately conservative and recursively walks
    nested dict/list/tuple structures. Unsupported values are ignored.
    """

    def __init__(
        self,
        name: str = "client_update_norm",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.history: List[Dict[str, Any]] = []

    def _sum_squared_norm(self, value: Any) -> float:
        """Return the summed squared norm of supported numeric leaves."""
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
            return float(value) * float(value)

        if isinstance(value, dict):
            return sum(self._sum_squared_norm(item) for item in value.values())

        if isinstance(value, (list, tuple, set)):
            return sum(self._sum_squared_norm(item) for item in value)

        return 0.0

    def _safe_client_norm(self, client_update: Any) -> Optional[float]:
        """Return a client's update norm or None when nothing is parseable."""
        try:
            squared_norm = self._sum_squared_norm(client_update)
        except Exception:
            return None

        if squared_norm <= 0:
            return None
        return float(math.sqrt(squared_norm))

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        client_norms: Dict[str, float] = {}

        if isinstance(participant_params, dict):
            for client_id, client_update in participant_params.items():
                client_norm = self._safe_client_norm(client_update)
                if client_norm is not None:
                    client_norms[str(client_id)] = client_norm

        if not client_norms:
            round_stats = {
                "avg_update_norm": None,
                "max_update_norm": None,
                "min_update_norm": None,
                "num_clients": 0,
            }
            self.history.append(round_stats)
            return round_stats

        norm_values = list(client_norms.values())
        round_stats = {
            "avg_update_norm": float(sum(norm_values) / len(norm_values)),
            "max_update_norm": float(max(norm_values)),
            "min_update_norm": float(min(norm_values)),
            "num_clients": int(len(norm_values)),
            "client_update_norms": client_norms,
        }
        self.history.append(round_stats)
        return round_stats

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        valid_entries = [
            entry
            for entry in self.history
            if entry.get("num_clients", 0) > 0
            and entry.get("avg_update_norm") is not None
        ]
        if not valid_entries:
            return {
                "num_rounds": 0,
                "avg_of_avg_update_norm": None,
                "max_observed_update_norm": None,
                "min_observed_update_norm": None,
            }

        avg_values = [float(entry["avg_update_norm"]) for entry in valid_entries]
        max_values = [float(entry["max_update_norm"]) for entry in valid_entries]
        min_values = [float(entry["min_update_norm"]) for entry in valid_entries]
        return {
            "num_rounds": len(valid_entries),
            "avg_of_avg_update_norm": float(sum(avg_values) / len(avg_values)),
            "max_observed_update_norm": float(max(max_values)),
            "min_observed_update_norm": float(min(min_values)),
        }


register_privacy_metric("client_update_norm", ClientUpdateNormMetric)
register_privacy_metric("clientupdatenormmetric", ClientUpdateNormMetric)
