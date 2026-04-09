"""Lightweight active defense that filters suspicious client updates.

This module is intentionally simple and explainable:
- it observes all participant updates before aggregation
- it scores clients with a norm-based rule
- it removes suspicious updates from the aggregation input
- it does not replace the aggregation algorithm itself
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class UpdateFilterDefense(BaseDefense):
    """Filter suspicious client uploads before aggregation."""

    def __init__(
        self,
        name: str = "update_filter",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        cfg = config or {}
        self.filter_rule = str(
            cfg.get("filter_rule", "update_norm > mean + filter_std_factor * std")
        )
        self.filter_std_factor = float(cfg.get("filter_std_factor", 2.0))
        self.max_filtered_ratio = float(cfg.get("max_filtered_ratio", 0.5))
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

    def _safe_client_score(self, client_update: Any) -> Optional[float]:
        try:
            squared_norm = self._sum_squared_norm(client_update)
        except Exception:
            return None

        if squared_norm <= 0:
            return None
        return float(math.sqrt(squared_norm))

    def _build_empty_result(self) -> Dict[str, Any]:
        return {
            "filtered_clients": [],
            "filtered_client_count": 0,
            "retained_client_count": 0,
            "filter_rule": self.filter_rule,
            "filter_std_factor": self.filter_std_factor,
            "max_filtered_ratio": self.max_filtered_ratio,
            "client_scores": {},
            "filter_threshold": None,
            "num_clients": 0,
            "avg_update_norm": None,
            "std_update_norm": None,
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict):
            defense_result = self._build_empty_result()
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        client_scores: Dict[str, float] = {}
        for client_id, client_update in participant_params.items():
            client_score = self._safe_client_score(client_update)
            if client_score is not None:
                client_scores[str(client_id)] = client_score

        if not client_scores:
            defense_result = self._build_empty_result()
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        score_values = list(client_scores.values())
        avg_score = float(sum(score_values) / len(score_values))
        variance = float(
            sum((score - avg_score) ** 2 for score in score_values) / len(score_values)
        )
        std_score = float(math.sqrt(variance))
        threshold = float(avg_score + self.filter_std_factor * std_score)

        suspicious_clients = [
            client_id
            for client_id, score in client_scores.items()
            if score > threshold
        ]

        num_clients = len(client_scores)
        if num_clients <= 1:
            filtered_clients: List[str] = []
        else:
            max_allowed = min(
                num_clients - 1,
                max(0, int(math.floor(num_clients * self.max_filtered_ratio))),
            )
            if max_allowed <= 0:
                filtered_clients = []
            else:
                suspicious_sorted = sorted(
                    suspicious_clients,
                    key=lambda client_id: client_scores.get(client_id, 0.0),
                    reverse=True,
                )
                filtered_clients = suspicious_sorted[:max_allowed]

        filtered_set = set(filtered_clients)
        updated_participant_params = {
            client_id: client_update
            for client_id, client_update in participant_params.items()
            if str(client_id) not in filtered_set
        }

        if not updated_participant_params and participant_params:
            retained_client_id = max(
                participant_params.keys(),
                key=lambda client_id: client_scores.get(str(client_id), float("-inf")),
            )
            updated_participant_params = {retained_client_id: participant_params[retained_client_id]}
            filtered_clients = [
                str(client_id)
                for client_id in participant_params.keys()
                if client_id != retained_client_id
            ]
            filtered_set = set(filtered_clients)

        defense_result = {
            "filtered_clients": filtered_clients,
            "filtered_client_count": len(filtered_clients),
            "retained_client_count": len(updated_participant_params),
            "filter_rule": self.filter_rule,
            "filter_std_factor": self.filter_std_factor,
            "max_filtered_ratio": self.max_filtered_ratio,
            "client_scores": client_scores,
            "filter_threshold": threshold,
            "num_clients": num_clients,
            "avg_update_norm": avg_score,
            "std_update_norm": std_score,
        }

        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("filtered_clients", {})[self.name] = filtered_clients
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "num_rounds": 0,
                "rounds_with_filtering": 0,
                "total_filtered_clients": 0,
                "max_filtered_client_count": 0,
                "filter_rule": self.filter_rule,
                "filter_std_factor": self.filter_std_factor,
                "max_filtered_ratio": self.max_filtered_ratio,
                "avg_retained_client_count": None,
            }

        filtered_counts = [
            int(item.get("filtered_client_count", 0)) for item in self.history
        ]
        retained_counts = [
            int(item.get("retained_client_count", 0)) for item in self.history
        ]
        return {
            "num_rounds": len(self.history),
            "rounds_with_filtering": sum(1 for count in filtered_counts if count > 0),
            "total_filtered_clients": int(sum(filtered_counts)),
            "max_filtered_client_count": int(max(filtered_counts)),
            "filter_rule": self.filter_rule,
            "filter_std_factor": self.filter_std_factor,
            "max_filtered_ratio": self.max_filtered_ratio,
            "avg_retained_client_count": (
                float(sum(retained_counts) / len(retained_counts))
                if retained_counts
                else None
            ),
        }


register_defense("update_filter", UpdateFilterDefense)
register_defense("updatefilterdefense", UpdateFilterDefense)
