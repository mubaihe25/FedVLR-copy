"""FSHA-inspired preference leakage probe for federated recommendation.

This module is intentionally read-only:
- it does not implement a real attack or reconstructor
- it does not mutate participant_params
- it only observes uploaded client updates before aggregation

The idea is inspired by FSHA-style privacy analysis:
- in split learning, server-observable intermediates may leak client privacy
- here, we use observable client updates in federated recommendation as a
  simplified probe to estimate preference leakage risk
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional

import torch

from attacks.base_attack import BaseAttack
from attacks.registry import register_attack


class ClientPreferenceLeakageProbe(BaseAttack):
    """Read-only probe that scores potential preference leakage risk."""

    def __init__(
        self,
        name: str = "client_preference_leakage_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.topk_ratio = float((config or {}).get("attack_probe_topk_ratio", 0.1))
        self.std_factor = float((config or {}).get("attack_probe_std_factor", 1.5))
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _collect_abs_values(self, value: Any) -> List[float]:
        if value is None:
            return []

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            tensor = tensor.float().reshape(-1)
            if tensor.numel() == 0:
                return []
            return [float(item) for item in torch.abs(tensor).cpu().tolist()]

        if isinstance(value, Number) and not isinstance(value, bool):
            return [abs(float(value))]

        if isinstance(value, dict):
            collected: List[float] = []
            for item in value.values():
                collected.extend(self._collect_abs_values(item))
            return collected

        if isinstance(value, (list, tuple, set)):
            collected: List[float] = []
            for item in value:
                collected.extend(self._collect_abs_values(item))
            return collected

        return []

    def _safe_leakage_features(self, client_update: Any) -> Optional[Dict[str, float]]:
        try:
            abs_values = self._collect_abs_values(client_update)
        except Exception:
            return None

        if not abs_values:
            return None

        num_values = len(abs_values)
        sum_squares = float(sum(value * value for value in abs_values))
        update_norm = float(math.sqrt(sum_squares))
        nonzero_count = sum(1 for value in abs_values if value > 0)
        nonzero_ratio = float(nonzero_count / max(1, num_values))

        sorted_values = sorted(abs_values, reverse=True)
        topk_count = max(1, int(math.ceil(num_values * self.topk_ratio)))
        topk_strength = float(sum(sorted_values[:topk_count]))
        total_strength = float(sum(sorted_values))
        topk_concentration = (
            float(topk_strength / total_strength) if total_strength > 0 else 0.0
        )

        leakage_score = float(update_norm * topk_concentration * nonzero_ratio)
        return {
            "update_norm": update_norm,
            "nonzero_ratio": nonzero_ratio,
            "topk_concentration": topk_concentration,
            "leakage_score": leakage_score,
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        client_features: Dict[str, Dict[str, float]] = {}
        leakage_scores: Dict[str, float] = {}

        if isinstance(participant_params, dict):
            for client_id, client_update in participant_params.items():
                features = self._safe_leakage_features(client_update)
                if features is None:
                    continue
                client_id_str = str(client_id)
                client_features[client_id_str] = features
                leakage_scores[client_id_str] = float(features["leakage_score"])

        if not leakage_scores:
            probe_result = {
                "leakage_scores": {},
                "high_risk_clients": [],
                "high_risk_client_count": 0,
                "risk_rule": "leakage_score > mean + std_factor * std",
                "num_clients": 0,
                "avg_leakage_score": None,
                "max_leakage_score": None,
                "risk_threshold": None,
                "client_features": {},
            }
        else:
            score_values = list(leakage_scores.values())
            avg_score = float(sum(score_values) / len(score_values))
            variance = float(
                sum((score - avg_score) ** 2 for score in score_values) / len(score_values)
            )
            std_score = float(math.sqrt(variance))
            risk_threshold = float(avg_score + self.std_factor * std_score)
            high_risk_clients = [
                client_id
                for client_id, score in leakage_scores.items()
                if score > risk_threshold
            ]
            probe_result = {
                "leakage_scores": leakage_scores,
                "high_risk_clients": high_risk_clients,
                "high_risk_client_count": len(high_risk_clients),
                "risk_rule": "leakage_score > mean + std_factor * std",
                "num_clients": len(leakage_scores),
                "avg_leakage_score": avg_score,
                "max_leakage_score": float(max(score_values)),
                "risk_threshold": risk_threshold,
                "client_features": client_features,
                "topk_ratio": self.topk_ratio,
                "std_factor": self.std_factor,
            }

        self.last_round_output = probe_result
        self.history.append(probe_result)
        round_state.setdefault("attack_outputs", {})[self.name] = probe_result
        round_state.setdefault("high_risk_clients", {})[self.name] = list(
            probe_result.get("high_risk_clients", [])
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
                "rounds_with_high_risk_clients": 0,
                "total_high_risk_detections": 0,
                "max_high_risk_client_count": 0,
                "risk_rule": "leakage_score > mean + std_factor * std",
                "topk_ratio": self.topk_ratio,
                "std_factor": self.std_factor,
            }

        risk_counts = [int(item.get("high_risk_client_count", 0)) for item in self.history]
        return {
            "num_rounds": len(self.history),
            "rounds_with_high_risk_clients": sum(1 for count in risk_counts if count > 0),
            "total_high_risk_detections": int(sum(risk_counts)),
            "max_high_risk_client_count": int(max(risk_counts)),
            "risk_rule": "leakage_score > mean + std_factor * std",
            "topk_ratio": self.topk_ratio,
            "std_factor": self.std_factor,
        }


register_attack("client_preference_leakage_probe", ClientPreferenceLeakageProbe)
register_attack("clientpreferenceleakageprobe", ClientPreferenceLeakageProbe)
