"""Minimal trimmed-mean-like robust aggregation defense.

This is not a full Byzantine-robust aggregation framework. It stays inside the
existing FedVLR before_aggregation hook and preserves the participant_params
mapping shape for downstream aggregators.

The defense computes a coordinate-wise trimmed mean prototype across client
updates, then assigns a cloned copy of that prototype back to every participant
key. Keeping the original client keys helps existing aggregation code continue
to work while making the final aggregate equivalent to the trimmed mean for
compatible tensor/numeric update structures.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class TrimmedMeanDefense(BaseDefense):
    """Apply a minimal coordinate-wise trimmed mean before aggregation."""

    def __init__(
        self,
        name: str = "trimmed_mean",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        cfg = config or {}
        self.trim_ratio = float(cfg.get("trim_ratio", 0.2))
        self.min_clients_for_trim = int(cfg.get("min_clients_for_trim", 5))
        self.trim_rule = str(
            cfg.get("trim_rule", "coordinate_trimmed_mean")
        ).lower()
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _safe_trim_ratio(self) -> float:
        if self.trim_ratio < 0:
            return 0.0
        if self.trim_ratio >= 0.5:
            return 0.49
        return self.trim_ratio

    def _build_result(
        self,
        participant_count: int,
        effective_trim_count: int = 0,
        trimmed_mean_applied: bool = False,
        touched_update_count: int = 0,
        retained_client_count_equivalent: Optional[int] = None,
        fallback_reason: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        trimmed_per_coord = (
            effective_trim_count * 2 if trimmed_mean_applied else 0
        )
        return {
            "defense_rule": self.trim_rule,
            "trim_ratio": self.trim_ratio,
            "safe_trim_ratio": self._safe_trim_ratio(),
            "min_clients_for_trim": self.min_clients_for_trim,
            "participant_count": participant_count,
            "effective_trim_count": effective_trim_count,
            "trimmed_mean_applied": trimmed_mean_applied,
            "trimmed_client_count_per_coord": trimmed_per_coord,
            "retained_client_count_equivalent": retained_client_count_equivalent,
            "touched_update_count": touched_update_count,
            "fallback_reason": fallback_reason,
            "notes": notes,
        }

    def _clone_value(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: self._clone_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_value(item) for item in value)
        return value

    def _trim_tensor_values(
        self, values: Sequence[Any], trim_count: int
    ) -> Tuple[Any, int, bool]:
        if not values or not all(torch.is_tensor(value) for value in values):
            return None, 0, False
        if any(value.is_sparse for value in values):
            return None, 0, False

        first = values[0]
        if not torch.is_floating_point(first):
            return None, 0, False
        if any(value.shape != first.shape for value in values):
            return None, 0, False

        try:
            stacked = torch.stack([value.to(device=first.device) for value in values], dim=0)
            sorted_values = torch.sort(stacked, dim=0).values
            trimmed_values = sorted_values[trim_count : len(values) - trim_count]
            return trimmed_values.mean(dim=0), 1, True
        except Exception:
            return None, 0, False

    def _trim_scalar_values(
        self, values: Sequence[Any], trim_count: int
    ) -> Tuple[Any, int, bool]:
        if not all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
            return None, 0, False
        sorted_values = sorted(float(value) for value in values)
        trimmed_values = sorted_values[trim_count : len(values) - trim_count]
        if not trimmed_values:
            return None, 0, False
        return float(sum(trimmed_values) / len(trimmed_values)), 1, True

    def _trim_values(
        self, values: Sequence[Any], trim_count: int
    ) -> Tuple[Any, int, bool]:
        values = [value for value in values if value is not None]
        if not values:
            return None, 0, False

        tensor_value, tensor_touched, tensor_ok = self._trim_tensor_values(
            values, trim_count
        )
        if tensor_ok:
            return tensor_value, tensor_touched, True

        scalar_value, scalar_touched, scalar_ok = self._trim_scalar_values(
            values, trim_count
        )
        if scalar_ok:
            return scalar_value, scalar_touched, True

        if all(isinstance(value, dict) for value in values):
            first = values[0]
            trimmed_dict: Dict[Any, Any] = {}
            touched_count = 0
            any_trimmed = False
            for key in first.keys():
                if not all(key in value for value in values):
                    trimmed_dict[key] = self._clone_value(first[key])
                    continue
                trimmed_item, item_touched, item_ok = self._trim_values(
                    [value[key] for value in values], trim_count
                )
                trimmed_dict[key] = trimmed_item if item_ok else self._clone_value(first[key])
                touched_count += item_touched
                any_trimmed = any_trimmed or item_ok
            return trimmed_dict, touched_count, any_trimmed

        if all(isinstance(value, list) for value in values):
            length = len(values[0])
            if any(len(value) != length for value in values):
                return None, 0, False
            trimmed_list: List[Any] = []
            touched_count = 0
            any_trimmed = False
            for index in range(length):
                trimmed_item, item_touched, item_ok = self._trim_values(
                    [value[index] for value in values], trim_count
                )
                trimmed_list.append(
                    trimmed_item if item_ok else self._clone_value(values[0][index])
                )
                touched_count += item_touched
                any_trimmed = any_trimmed or item_ok
            return trimmed_list, touched_count, any_trimmed

        if all(isinstance(value, tuple) for value in values):
            length = len(values[0])
            if any(len(value) != length for value in values):
                return None, 0, False
            trimmed_items: List[Any] = []
            touched_count = 0
            any_trimmed = False
            for index in range(length):
                trimmed_item, item_touched, item_ok = self._trim_values(
                    [value[index] for value in values], trim_count
                )
                trimmed_items.append(
                    trimmed_item if item_ok else self._clone_value(values[0][index])
                )
                touched_count += item_touched
                any_trimmed = any_trimmed or item_ok
            return tuple(trimmed_items), touched_count, any_trimmed

        return None, 0, False

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict) or not participant_params:
            defense_result = self._build_result(
                participant_count=0,
                fallback_reason="empty_or_unsupported_participant_params",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        participant_count = len(participant_params)
        if participant_count < self.min_clients_for_trim:
            defense_result = self._build_result(
                participant_count=participant_count,
                fallback_reason="insufficient_clients_for_trim",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        trim_count = int(participant_count * self._safe_trim_ratio())
        if trim_count <= 0 or participant_count - 2 * trim_count <= 0:
            defense_result = self._build_result(
                participant_count=participant_count,
                effective_trim_count=trim_count,
                fallback_reason="trim_ratio_does_not_leave_valid_clients",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        participant_updates = list(participant_params.values())
        trimmed_update, touched_update_count, trimmed_ok = self._trim_values(
            participant_updates, trim_count
        )

        if not trimmed_ok:
            defense_result = self._build_result(
                participant_count=participant_count,
                effective_trim_count=trim_count,
                fallback_reason="unsupported_update_structure",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        updated_participant_params = {
            client_id: self._clone_value(trimmed_update)
            for client_id in participant_params.keys()
        }
        retained_equivalent = participant_count - 2 * trim_count
        defense_result = self._build_result(
            participant_count=participant_count,
            effective_trim_count=trim_count,
            trimmed_mean_applied=True,
            touched_update_count=touched_update_count,
            retained_client_count_equivalent=retained_equivalent,
            notes=(
                "participant keys are preserved; each upload is replaced with a "
                "cloned coordinate-wise trimmed mean prototype"
            ),
        )
        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("trimmed_mean_applied", {})[self.name] = True
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "num_rounds": 0,
                "rounds_with_trimmed_mean": 0,
                "trim_ratio": self.trim_ratio,
                "min_clients_for_trim": self.min_clients_for_trim,
                "trim_rule": self.trim_rule,
                "avg_effective_trim_count": None,
                "avg_retained_client_count_equivalent": None,
            }

        applied_rounds = [
            item for item in self.history if item.get("trimmed_mean_applied")
        ]
        effective_counts = [
            int(item.get("effective_trim_count", 0)) for item in applied_rounds
        ]
        retained_counts = [
            int(item.get("retained_client_count_equivalent", 0))
            for item in applied_rounds
            if item.get("retained_client_count_equivalent") is not None
        ]
        return {
            "num_rounds": len(self.history),
            "rounds_with_trimmed_mean": len(applied_rounds),
            "trim_ratio": self.trim_ratio,
            "min_clients_for_trim": self.min_clients_for_trim,
            "trim_rule": self.trim_rule,
            "avg_effective_trim_count": (
                float(sum(effective_counts) / len(effective_counts))
                if effective_counts
                else None
            ),
            "avg_retained_client_count_equivalent": (
                float(sum(retained_counts) / len(retained_counts))
                if retained_counts
                else None
            ),
        }


register_defense("trimmed_mean", TrimmedMeanDefense)
register_defense("trimmedmeandefense", TrimmedMeanDefense)
