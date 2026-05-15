"""Minimal targeted/preference poisoning proxy.

The current FedVLR hook exposes uploaded client updates, not raw user-item
training samples. This wrapper therefore implements a lightweight update-space
proxy for targeted or preference poisoning. When target item rows can be
identified in item-like tensors, it perturbs those rows; otherwise it can fall
back to a clearly marked global preference-direction proxy.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch

from attacks.base_attack import BaseAttack
from attacks.registry import register_attack


class TargetedPoisoningAttack(BaseAttack):
    """Directionally perturb malicious client updates as a poisoning proxy."""

    attack_family = "poisoning"
    attack_category = "targeted_poisoning_proxy"
    attack_strategy = "update_space_preference_direction"
    attack_display_category = "poisoning attack"
    mutates_participant_params = True
    is_read_only = False

    def __init__(
        self,
        name: str = "targeted_poisoning",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        config = config or {}
        self.strength = float(
            config.get(
                "targeted_poisoning_strength",
                config.get(
                    "preference_poisoning_strength",
                    config.get("poisoning_attack_scale", 0.05),
                ),
            )
        )
        self.direction = str(
            config.get("preference_direction", config.get("target_direction", "increase"))
        ).lower()
        self.target_item_ids = self._parse_target_ids(
            config.get(
                "target_item_ids",
                config.get("targeted_item_ids", config.get("poisoning_target_item_ids", [])),
            )
        )
        self.allow_proxy_fallback = bool(config.get("targeted_allow_proxy_fallback", True))
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _parse_target_ids(value: Any) -> List[int]:
        if value is None or value == "":
            return []
        if isinstance(value, (str, Number)) and not isinstance(value, bool):
            value = [value]
        target_ids: List[int] = []
        if isinstance(value, (list, tuple, set)):
            for item in value:
                try:
                    target_ids.append(int(item))
                except (TypeError, ValueError):
                    continue
        return sorted(set(target_ids))

    @staticmethod
    def _sum_squared_norm(value: Any) -> float:
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            tensor = tensor.float()
            return float(torch.sum(tensor * tensor).item()) if tensor.numel() else 0.0
        if isinstance(value, Number) and not isinstance(value, bool):
            scalar = float(value)
            return scalar * scalar
        if isinstance(value, dict):
            return sum(TargetedPoisoningAttack._sum_squared_norm(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(TargetedPoisoningAttack._sum_squared_norm(item) for item in value)
        return 0.0

    def _safe_norm(self, value: Any) -> Optional[float]:
        try:
            squared_norm = self._sum_squared_norm(value)
        except Exception:
            return None
        return float(math.sqrt(squared_norm)) if squared_norm > 0 else None

    def _sign(self) -> float:
        if self.direction in {"decrease", "demote", "negative", "down"}:
            return -1.0
        return 1.0

    @staticmethod
    def _is_item_like_key(path: str) -> bool:
        normalized = path.lower()
        return "item" in normalized or "embedding" in normalized or "emb" in normalized

    def _perturb_tensor(self, tensor: torch.Tensor, path: str) -> Tuple[torch.Tensor, int, int, str]:
        if tensor.numel() == 0:
            return tensor, 0, 0, "empty_tensor"

        sign = self._sign()
        output = tensor.clone()
        touched_tensor_count = 1
        targeted_row_count = 0
        mode = "global_preference_proxy"

        if self.target_item_ids and output.ndim >= 2 and self._is_item_like_key(path):
            valid_ids = [item_id for item_id in self.target_item_ids if 0 <= item_id < output.shape[0]]
            if valid_ids:
                index = torch.tensor(valid_ids, dtype=torch.long, device=output.device)
                output[index] = output[index] + sign * self.strength
                targeted_row_count = len(valid_ids)
                mode = "target_item_row_proxy"
                return output, touched_tensor_count, targeted_row_count, mode

        if self.target_item_ids and not self.allow_proxy_fallback:
            return tensor, 0, 0, "target_rows_not_found"

        output = output + sign * self.strength
        return output, touched_tensor_count, targeted_row_count, mode

    def _perturb_value(self, value: Any, path: str = "") -> Tuple[Any, int, int, List[str]]:
        if value is None:
            return value, 0, 0, []
        if torch.is_tensor(value):
            updated, touched, targeted_rows, mode = self._perturb_tensor(value, path)
            return updated, touched, targeted_rows, [mode] if touched or mode == "target_rows_not_found" else []
        if isinstance(value, Number) and not isinstance(value, bool):
            if self.target_item_ids and not self.allow_proxy_fallback:
                return value, 0, 0, ["target_rows_not_found"]
            return float(value) + self._sign() * self.strength, 1, 0, ["scalar_preference_proxy"]
        if isinstance(value, dict):
            updated: Dict[Any, Any] = {}
            touched_total = 0
            targeted_total = 0
            modes: List[str] = []
            for key, item in value.items():
                child, touched, targeted_rows, child_modes = self._perturb_value(
                    item, "{}.{}".format(path, key) if path else str(key)
                )
                updated[key] = child
                touched_total += touched
                targeted_total += targeted_rows
                modes.extend(child_modes)
            return updated, touched_total, targeted_total, modes
        if isinstance(value, list):
            updated_list = []
            touched_total = 0
            targeted_total = 0
            modes: List[str] = []
            for index, item in enumerate(value):
                child, touched, targeted_rows, child_modes = self._perturb_value(
                    item, "{}[{}]".format(path, index)
                )
                updated_list.append(child)
                touched_total += touched
                targeted_total += targeted_rows
                modes.extend(child_modes)
            return updated_list, touched_total, targeted_total, modes
        if isinstance(value, tuple):
            updated_items = []
            touched_total = 0
            targeted_total = 0
            modes: List[str] = []
            for index, item in enumerate(value):
                child, touched, targeted_rows, child_modes = self._perturb_value(
                    item, "{}[{}]".format(path, index)
                )
                updated_items.append(child)
                touched_total += touched
                targeted_total += targeted_rows
                modes.extend(child_modes)
            return tuple(updated_items), touched_total, targeted_total, modes
        return value, 0, 0, []

    def _empty_result(self) -> Dict[str, Any]:
        return {
            **self.semantic_metadata(),
            "attack_type": "targeted_poisoning",
            "poisoning_mode": "targeted_or_preference_proxy",
            "proxy_only": True,
            "target_item_ids": list(self.target_item_ids),
            "preference_direction": self.direction,
            "targeted_poisoning_strength": self.strength,
            "attacked_clients": [],
            "attacked_client_count": 0,
            "touched_tensor_count": 0,
            "targeted_row_count": 0,
            "avg_attacked_norm_before": None,
            "avg_attacked_norm_after": None,
            "mutation_modes": [],
            "note": "update-space proxy only; not a full target-item backdoor attack",
            "warnings": [],
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
            result = self._empty_result()
            self.last_round_output = result
            self.history.append(result)
            round_state.setdefault("attack_outputs", {})[self.name] = result
            return participant_params

        updated_participant_params = dict(participant_params)
        attacked_clients: List[str] = []
        norm_before_values: List[float] = []
        norm_after_values: List[float] = []
        touched_tensor_count = 0
        targeted_row_count = 0
        mutation_modes: List[str] = []

        for client_id, client_update in participant_params.items():
            client_id_str = str(client_id)
            if client_id_str not in malicious_clients:
                continue
            updated_update, touched, targeted_rows, modes = self._perturb_value(client_update)
            if touched <= 0:
                mutation_modes.extend(modes)
                continue
            updated_participant_params[client_id] = updated_update
            attacked_clients.append(client_id_str)
            touched_tensor_count += touched
            targeted_row_count += targeted_rows
            mutation_modes.extend(modes)
            norm_before = self._safe_norm(client_update)
            norm_after = self._safe_norm(updated_update)
            if norm_before is not None:
                norm_before_values.append(norm_before)
            if norm_after is not None:
                norm_after_values.append(norm_after)

        result = self._empty_result()
        result.update(
            {
                "attacked_clients": attacked_clients,
                "attacked_client_count": len(attacked_clients),
                "touched_tensor_count": int(touched_tensor_count),
                "targeted_row_count": int(targeted_row_count),
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
                "mutation_modes": sorted(set(mutation_modes)),
                "warnings": (
                    ["target item rows not found; used global preference proxy"]
                    if self.target_item_ids and targeted_row_count == 0 and touched_tensor_count > 0
                    else []
                ),
            }
        )

        self.last_round_output = result
        self.history.append(result)
        round_state.setdefault("attack_outputs", {})[self.name] = result
        round_state.setdefault("attacked_clients", {})[self.name] = attacked_clients
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            result = self._empty_result()
            result["num_rounds"] = 0
            return result

        latest = dict(self.history[-1])
        latest["num_rounds"] = len(self.history)
        latest["rounds_with_attacks"] = sum(
            1 for item in self.history if int(item.get("attacked_client_count", 0)) > 0
        )
        latest["total_attacked_clients"] = int(
            sum(int(item.get("attacked_client_count", 0)) for item in self.history)
        )
        latest["total_touched_tensor_count"] = int(
            sum(int(item.get("touched_tensor_count", 0)) for item in self.history)
        )
        latest["total_targeted_row_count"] = int(
            sum(int(item.get("targeted_row_count", 0)) for item in self.history)
        )
        return latest


register_attack("targeted_poisoning", TargetedPoisoningAttack)
register_attack("targeted_poisoning_attack", TargetedPoisoningAttack)
register_attack("preference_poisoning", TargetedPoisoningAttack)
register_attack("preference_poisoning_attack", TargetedPoisoningAttack)
