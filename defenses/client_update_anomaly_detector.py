"""Read-only defense detector for client update anomalies.

This module only observes uploaded client updates before aggregation:
- it computes simple norm-based anomaly scores
- it records suspicious clients using an explainable threshold rule
- it never filters clients or mutates participant_params
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class ClientUpdateAnomalyDetector(BaseDefense):
    """Detect suspicious client updates using a norm threshold rule."""

    def __init__(
        self,
        name: str = "client_update_anomaly",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.std_factor = float((config or {}).get("defense_anomaly_std_factor", 2.0))
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

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        client_scores: Dict[str, float] = {}

        if isinstance(participant_params, dict):
            for client_id, client_update in participant_params.items():
                client_score = self._safe_client_score(client_update)
                if client_score is not None:
                    client_scores[str(client_id)] = client_score

        if not client_scores:
            detection_result = {
                "suspicious_clients": [],
                "suspicious_client_count": 0,
                "anomaly_threshold": None,
                "detection_rule": "update_norm > mean + std_factor * std",
                "client_scores": {},
                "num_clients": 0,
                "avg_update_norm": None,
                "std_update_norm": None,
            }
        else:
            score_values = list(client_scores.values())
            avg_score = float(sum(score_values) / len(score_values))
            variance = float(
                sum((score - avg_score) ** 2 for score in score_values) / len(score_values)
            )
            std_score = float(math.sqrt(variance))
            threshold = float(avg_score + self.std_factor * std_score)
            suspicious_clients = [
                client_id
                for client_id, score in client_scores.items()
                if score > threshold
            ]
            detection_result = {
                "suspicious_clients": suspicious_clients,
                "suspicious_client_count": len(suspicious_clients),
                "anomaly_threshold": threshold,
                "detection_rule": "update_norm > mean + std_factor * std",
                "client_scores": client_scores,
                "num_clients": len(client_scores),
                "avg_update_norm": avg_score,
                "std_update_norm": std_score,
                "std_factor": self.std_factor,
            }

        self.last_round_output = detection_result
        self.history.append(detection_result)
        round_state.setdefault("defense_outputs", {})[self.name] = detection_result
        round_state.setdefault("suspicious_clients", {})[self.name] = list(
            detection_result.get("suspicious_clients", [])
        )
        return participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "num_rounds": 0,
                "rounds_with_suspicious_clients": 0,
                "total_suspicious_detections": 0,
                "max_suspicious_client_count": 0,
                "detection_rule": "update_norm > mean + std_factor * std",
                "std_factor": self.std_factor,
            }

        suspicious_counts = [
            int(item.get("suspicious_client_count", 0)) for item in self.history
        ]
        rounds_with_suspicious = sum(1 for count in suspicious_counts if count > 0)
        return {
            "num_rounds": len(self.history),
            "rounds_with_suspicious_clients": rounds_with_suspicious,
            "total_suspicious_detections": int(sum(suspicious_counts)),
            "max_suspicious_client_count": int(max(suspicious_counts)),
            "detection_rule": "update_norm > mean + std_factor * std",
            "std_factor": self.std_factor,
        }


register_defense("client_update_anomaly", ClientUpdateAnomalyDetector)
register_defense("clientupdateanomalydetector", ClientUpdateAnomalyDetector)
