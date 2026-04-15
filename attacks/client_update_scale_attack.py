"""Lightweight active attack that scales malicious client updates.

This is intentionally simple and explainable:
- it only acts on clients listed in the current round's malicious_clients
- it only rescales uploaded client updates before aggregation
- it does not modify the model structure, loss, or aggregation algorithm
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch

from attacks.base_attack import BaseAttack
from attacks.registry import register_attack


class ClientUpdateScaleAttack(BaseAttack):
    """Scale malicious client uploads by a constant factor."""

    attack_family = "poisoning"
    attack_category = "poisoning"
    attack_strategy = "scale"
    attack_display_category = "投毒攻击"
    mutates_participant_params = True
    is_read_only = False

    def __init__(
        self,
        name: str = "client_update_scale",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.attack_scale = float((config or {}).get("attack_scale", 2.0))
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

    def _scale_value(self, value: Any) -> Tuple[Any, int]:
        if value is None:
            return value, 0

        if torch.is_tensor(value):
            return value * self.attack_scale, 1

        if isinstance(value, Number) and not isinstance(value, bool):
            return float(value) * self.attack_scale, 1

        if isinstance(value, dict):
            scaled_dict: Dict[Any, Any] = {}
            touched_count = 0
            for key, item in value.items():
                scaled_item, item_touched = self._scale_value(item)
                scaled_dict[key] = scaled_item
                touched_count += item_touched
            return scaled_dict, touched_count

        if isinstance(value, list):
            scaled_list: List[Any] = []
            touched_count = 0
            for item in value:
                scaled_item, item_touched = self._scale_value(item)
                scaled_list.append(scaled_item)
                touched_count += item_touched
            return scaled_list, touched_count

        if isinstance(value, tuple):
            scaled_items: List[Any] = []
            touched_count = 0
            for item in value:
                scaled_item, item_touched = self._scale_value(item)
                scaled_items.append(scaled_item)
                touched_count += item_touched
            return tuple(scaled_items), touched_count

        return value, 0

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        malicious_clients = {
            str(client_id) for client_id in round_state.get("malicious_clients", [])
        }

        if not isinstance(participant_params, dict) or not malicious_clients:
            attack_result = {
                **self.semantic_metadata(),
                "attacked_clients": [],
                "attacked_client_count": 0,
                "attack_scale": self.attack_scale,
                "touched_update_count": 0,
                "attacked_client_norms_before": {},
                "attacked_client_norms_after": {},
                "avg_attacked_norm_before": None,
                "avg_attacked_norm_after": None,
            }
            self.last_round_output = attack_result
            self.history.append(attack_result)
            round_state.setdefault("attack_outputs", {})[self.name] = attack_result
            return participant_params

        updated_participant_params = dict(participant_params)
        attacked_clients: List[str] = []
        attacked_client_norms_before: Dict[str, float] = {}
        attacked_client_norms_after: Dict[str, float] = {}
        touched_update_count = 0

        for client_id, client_update in participant_params.items():
            client_id_str = str(client_id)
            if client_id_str not in malicious_clients:
                continue

            scaled_update, touched_count = self._scale_value(client_update)
            updated_participant_params[client_id] = scaled_update
            attacked_clients.append(client_id_str)
            touched_update_count += touched_count

            norm_before = self._safe_norm(client_update)
            norm_after = self._safe_norm(scaled_update)
            if norm_before is not None:
                attacked_client_norms_before[client_id_str] = norm_before
            if norm_after is not None:
                attacked_client_norms_after[client_id_str] = norm_after

        before_values = list(attacked_client_norms_before.values())
        after_values = list(attacked_client_norms_after.values())
        attack_result = {
            **self.semantic_metadata(),
            "attacked_clients": attacked_clients,
            "attacked_client_count": len(attacked_clients),
            "attack_scale": self.attack_scale,
            "touched_update_count": touched_update_count,
            "attacked_client_norms_before": attacked_client_norms_before,
            "attacked_client_norms_after": attacked_client_norms_after,
            "avg_attacked_norm_before": (
                float(sum(before_values) / len(before_values)) if before_values else None
            ),
            "avg_attacked_norm_after": (
                float(sum(after_values) / len(after_values)) if after_values else None
            ),
        }

        self.last_round_output = attack_result
        self.history.append(attack_result)
        round_state.setdefault("attack_outputs", {})[self.name] = attack_result
        round_state.setdefault("attacked_clients", {})[self.name] = attacked_clients
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                **self.semantic_metadata(),
                "num_rounds": 0,
                "rounds_with_attacks": 0,
                "total_attacked_clients": 0,
                "max_attacked_client_count": 0,
                "total_touched_update_count": 0,
                "attack_scale": self.attack_scale,
                "avg_attacked_norm_before": None,
                "avg_attacked_norm_after": None,
            }

        attacked_counts = [
            int(item.get("attacked_client_count", 0)) for item in self.history
        ]
        norm_before_values = [
            float(item["avg_attacked_norm_before"])
            for item in self.history
            if item.get("avg_attacked_norm_before") is not None
        ]
        norm_after_values = [
            float(item["avg_attacked_norm_after"])
            for item in self.history
            if item.get("avg_attacked_norm_after") is not None
        ]
        return {
            **self.semantic_metadata(),
            "num_rounds": len(self.history),
            "rounds_with_attacks": sum(1 for count in attacked_counts if count > 0),
            "total_attacked_clients": int(sum(attacked_counts)),
            "max_attacked_client_count": int(max(attacked_counts)),
            "total_touched_update_count": int(
                sum(int(item.get("touched_update_count", 0)) for item in self.history)
            ),
            "attack_scale": self.attack_scale,
            "avg_attacked_norm_before": (
                float(sum(norm_before_values) / len(norm_before_values))
                if norm_before_values
                else None
            ),
            "avg_attacked_norm_after": (
                float(sum(norm_after_values) / len(norm_after_values))
                if norm_after_values
                else None
            ),
        }


register_attack("client_update_scale", ClientUpdateScaleAttack)
register_attack("clientupdatescaleattack", ClientUpdateScaleAttack)
