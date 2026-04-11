"""Minimal model-replacement-like active attack.

This is not a full paper-level model replacement or backdoor implementation.
It stays inside the existing FedVLR engineering hook:
- it only acts on clients listed in the current round's malicious_clients
- it only mutates participant_params before aggregation
- it does not access global model internals, add triggers, or change local loss

The default rule, ``aligned_mean``, builds a shared malicious update direction
from the current round's malicious clients and replaces each malicious upload
with ``replacement_scale * shared_direction``. Compared with a per-client scale
attack, this makes malicious clients push the aggregate in a coordinated,
replacement-like direction while keeping the implementation small and safe.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from attacks.base_attack import BaseAttack
from attacks.registry import register_attack


class ModelReplacementAttack(BaseAttack):
    """Apply a minimal replacement-like transform to malicious client uploads."""

    def __init__(
        self,
        name: str = "model_replacement",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.replacement_scale = float((config or {}).get("replacement_scale", 5.0))
        self.replacement_rule = str(
            (config or {}).get("replacement_rule", "aligned_mean")
        ).lower()
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
            return value * self.replacement_scale, 1

        if isinstance(value, Number) and not isinstance(value, bool):
            return float(value) * self.replacement_scale, 1

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

    def _average_tensor_values(self, values: Sequence[Any]) -> Tuple[Any, int, bool]:
        if not values or not all(torch.is_tensor(value) for value in values):
            return None, 0, False
        if any(value.is_sparse for value in values):
            return None, 0, False
        first = values[0]
        if any(value.shape != first.shape for value in values):
            return None, 0, False

        try:
            total = torch.zeros_like(first)
            for value in values:
                total = total + value
            return total / len(values), 1, True
        except Exception:
            return None, 0, False

    def _average_values(self, values: Sequence[Any]) -> Tuple[Any, int, bool]:
        values = [value for value in values if value is not None]
        if not values:
            return None, 0, False

        tensor_average, tensor_touched, tensor_ok = self._average_tensor_values(values)
        if tensor_ok:
            return tensor_average, tensor_touched, True

        if all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
            return float(sum(float(value) for value in values) / len(values)), 1, True

        if all(isinstance(value, dict) for value in values):
            first = values[0]
            averaged_dict: Dict[Any, Any] = {}
            touched_count = 0
            any_averaged = False
            for key in first.keys():
                if not all(key in value for value in values):
                    averaged_dict[key] = first[key]
                    continue
                averaged_item, item_touched, item_ok = self._average_values(
                    [value[key] for value in values]
                )
                averaged_dict[key] = averaged_item if item_ok else first[key]
                touched_count += item_touched
                any_averaged = any_averaged or item_ok
            return averaged_dict, touched_count, any_averaged

        if all(isinstance(value, list) for value in values):
            length = len(values[0])
            if any(len(value) != length for value in values):
                return None, 0, False
            averaged_list: List[Any] = []
            touched_count = 0
            any_averaged = False
            for index in range(length):
                averaged_item, item_touched, item_ok = self._average_values(
                    [value[index] for value in values]
                )
                averaged_list.append(averaged_item if item_ok else values[0][index])
                touched_count += item_touched
                any_averaged = any_averaged or item_ok
            return averaged_list, touched_count, any_averaged

        if all(isinstance(value, tuple) for value in values):
            length = len(values[0])
            if any(len(value) != length for value in values):
                return None, 0, False
            averaged_items: List[Any] = []
            touched_count = 0
            any_averaged = False
            for index in range(length):
                averaged_item, item_touched, item_ok = self._average_values(
                    [value[index] for value in values]
                )
                averaged_items.append(averaged_item if item_ok else values[0][index])
                touched_count += item_touched
                any_averaged = any_averaged or item_ok
            return tuple(averaged_items), touched_count, any_averaged

        return None, 0, False

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "attacked_clients": [],
            "attacked_client_count": 0,
            "replacement_scale": self.replacement_scale,
            "replacement_rule": self.replacement_rule,
            "effective_replacement_rule": "skipped",
            "touched_update_count": 0,
            "attacked_client_norms_before": {},
            "attacked_client_norms_after": {},
            "avg_attacked_norm_before": None,
            "avg_attacked_norm_after": None,
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        malicious_clients = {
            str(client_id) for client_id in round_state.get("malicious_clients", [])
        }

        if not isinstance(participant_params, dict) or not malicious_clients:
            attack_result = self._empty_result()
            self.last_round_output = attack_result
            self.history.append(attack_result)
            round_state.setdefault("attack_outputs", {})[self.name] = attack_result
            return participant_params

        target_updates = [
            client_update
            for client_id, client_update in participant_params.items()
            if str(client_id) in malicious_clients
        ]
        if not target_updates:
            attack_result = self._empty_result()
            self.last_round_output = attack_result
            self.history.append(attack_result)
            round_state.setdefault("attack_outputs", {})[self.name] = attack_result
            return participant_params

        prototype_update = None
        prototype_ok = False
        effective_rule = self.replacement_rule
        if self.replacement_rule in {"aligned_mean", "mean", "mean_aligned"}:
            prototype_update, _, prototype_ok = self._average_values(target_updates)
            if not prototype_ok:
                effective_rule = "scaled_original_fallback"
        else:
            effective_rule = "scaled_original"

        updated_participant_params = dict(participant_params)
        attacked_clients: List[str] = []
        attacked_client_norms_before: Dict[str, float] = {}
        attacked_client_norms_after: Dict[str, float] = {}
        touched_update_count = 0

        for client_id, client_update in participant_params.items():
            client_id_str = str(client_id)
            if client_id_str not in malicious_clients:
                continue

            source_update = prototype_update if prototype_ok else client_update
            replaced_update, touched_count = self._scale_value(source_update)
            updated_participant_params[client_id] = replaced_update
            attacked_clients.append(client_id_str)
            touched_update_count += touched_count

            norm_before = self._safe_norm(client_update)
            norm_after = self._safe_norm(replaced_update)
            if norm_before is not None:
                attacked_client_norms_before[client_id_str] = norm_before
            if norm_after is not None:
                attacked_client_norms_after[client_id_str] = norm_after

        before_values = list(attacked_client_norms_before.values())
        after_values = list(attacked_client_norms_after.values())
        attack_result = {
            "attacked_clients": attacked_clients,
            "attacked_client_count": len(attacked_clients),
            "replacement_scale": self.replacement_scale,
            "replacement_rule": self.replacement_rule,
            "effective_replacement_rule": effective_rule,
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
                "num_rounds": 0,
                "rounds_with_attacks": 0,
                "total_attacked_clients": 0,
                "max_attacked_client_count": 0,
                "total_touched_update_count": 0,
                "replacement_scale": self.replacement_scale,
                "replacement_rule": self.replacement_rule,
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
            "num_rounds": len(self.history),
            "rounds_with_attacks": sum(1 for count in attacked_counts if count > 0),
            "total_attacked_clients": int(sum(attacked_counts)),
            "max_attacked_client_count": int(max(attacked_counts)),
            "total_touched_update_count": int(
                sum(int(item.get("touched_update_count", 0)) for item in self.history)
            ),
            "replacement_scale": self.replacement_scale,
            "replacement_rule": self.replacement_rule,
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


register_attack("model_replacement", ModelReplacementAttack)
register_attack("modelreplacementattack", ModelReplacementAttack)
