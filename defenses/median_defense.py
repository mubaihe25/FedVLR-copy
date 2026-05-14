"""Coordinate-wise median defense for uploaded client updates.

This module stays inside the existing ``before_aggregation`` hook. It replaces
compatible tensor leaves with their coordinate-wise median while preserving the
outer participant mapping expected by model-specific aggregators.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


Mask = Any


class MedianDefense(BaseDefense):
    """Apply coordinate-wise median to compatible tensor update leaves."""

    def __init__(
        self,
        name: str = "median",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

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

    def _median_tensor_values(
        self, values: Sequence[Any]
    ) -> Tuple[Any, Mask, int, int, bool]:
        if not values or not all(torch.is_tensor(value) for value in values):
            return None, False, 0, 1, False
        if any(value.is_sparse for value in values):
            return self._clone_value(values[0]), False, 0, 1, False

        first = values[0]
        if not torch.is_floating_point(first):
            return self._clone_value(first), False, 0, 1, False
        if any(value.shape != first.shape for value in values):
            return self._clone_value(first), False, 0, 1, False

        try:
            stacked = torch.stack(
                [
                    value.detach().to(device=first.device, dtype=first.dtype)
                    for value in values
                ],
                dim=0,
            )
            median_value = torch.median(stacked, dim=0).values
            return median_value, True, 1, 0, True
        except Exception:
            return self._clone_value(first), False, 0, 1, False

    def _median_values(
        self, values: Sequence[Any]
    ) -> Tuple[Any, Mask, int, int, bool]:
        values = [value for value in values if value is not None]
        if not values:
            return None, False, 0, 1, False

        if all(torch.is_tensor(value) for value in values):
            return self._median_tensor_values(values)

        if all(isinstance(value, dict) for value in values):
            first = values[0]
            all_keys = set()
            for value in values:
                all_keys.update(value.keys())

            median_dict: Dict[Any, Any] = {}
            mask_dict: Dict[Any, Mask] = {}
            aggregated_tensor_count = 0
            skipped_key_count = len(all_keys.difference(first.keys()))
            any_aggregated = False

            for key in first.keys():
                if not all(key in value for value in values):
                    median_dict[key] = self._clone_value(first[key])
                    mask_dict[key] = False
                    skipped_key_count += 1
                    continue

                median_item, item_mask, item_count, item_skipped, item_ok = (
                    self._median_values([value[key] for value in values])
                )
                median_dict[key] = median_item
                mask_dict[key] = item_mask
                aggregated_tensor_count += item_count
                skipped_key_count += item_skipped
                any_aggregated = any_aggregated or item_ok

            return (
                median_dict,
                mask_dict,
                aggregated_tensor_count,
                skipped_key_count,
                any_aggregated,
            )

        if all(isinstance(value, list) for value in values):
            length = len(values[0])
            if any(len(value) != length for value in values):
                return self._clone_value(values[0]), False, 0, 1, False

            median_list: List[Any] = []
            mask_list: List[Mask] = []
            aggregated_tensor_count = 0
            skipped_key_count = 0
            any_aggregated = False
            for index in range(length):
                median_item, item_mask, item_count, item_skipped, item_ok = (
                    self._median_values([value[index] for value in values])
                )
                median_list.append(median_item)
                mask_list.append(item_mask)
                aggregated_tensor_count += item_count
                skipped_key_count += item_skipped
                any_aggregated = any_aggregated or item_ok
            return (
                median_list,
                mask_list,
                aggregated_tensor_count,
                skipped_key_count,
                any_aggregated,
            )

        if all(isinstance(value, tuple) for value in values):
            length = len(values[0])
            if any(len(value) != length for value in values):
                return self._clone_value(values[0]), False, 0, 1, False

            median_items: List[Any] = []
            mask_items: List[Mask] = []
            aggregated_tensor_count = 0
            skipped_key_count = 0
            any_aggregated = False
            for index in range(length):
                median_item, item_mask, item_count, item_skipped, item_ok = (
                    self._median_values([value[index] for value in values])
                )
                median_items.append(median_item)
                mask_items.append(item_mask)
                aggregated_tensor_count += item_count
                skipped_key_count += item_skipped
                any_aggregated = any_aggregated or item_ok
            return (
                tuple(median_items),
                tuple(mask_items),
                aggregated_tensor_count,
                skipped_key_count,
                any_aggregated,
            )

        return self._clone_value(values[0]), False, 0, 1, False

    def _apply_median_mask(self, original: Any, prototype: Any, mask: Mask) -> Any:
        if mask is True:
            return self._clone_value(prototype)

        if isinstance(mask, dict) and isinstance(original, dict) and isinstance(prototype, dict):
            updated: Dict[Any, Any] = {}
            for key, value in original.items():
                if key in mask and key in prototype:
                    updated[key] = self._apply_median_mask(value, prototype[key], mask[key])
                else:
                    updated[key] = self._clone_value(value)
            return updated

        if isinstance(mask, list) and isinstance(original, list) and isinstance(prototype, list):
            updated_list: List[Any] = []
            for index, value in enumerate(original):
                if index < len(mask) and index < len(prototype):
                    updated_list.append(
                        self._apply_median_mask(value, prototype[index], mask[index])
                    )
                else:
                    updated_list.append(self._clone_value(value))
            return updated_list

        if isinstance(mask, tuple) and isinstance(original, tuple) and isinstance(prototype, tuple):
            updated_items: List[Any] = []
            for index, value in enumerate(original):
                if index < len(mask) and index < len(prototype):
                    updated_items.append(
                        self._apply_median_mask(value, prototype[index], mask[index])
                    )
                else:
                    updated_items.append(self._clone_value(value))
            return tuple(updated_items)

        return self._clone_value(original)

    def _build_result(
        self,
        participant_count: int,
        aggregated_tensor_count: int = 0,
        skipped_key_count: int = 0,
        median_applied: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "defense_type": "median",
            "participant_count": participant_count,
            "aggregated_tensor_count": aggregated_tensor_count,
            "skipped_key_count": skipped_key_count,
            "median_applied": median_applied,
            "fallback_reason": fallback_reason,
        }

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

        participant_updates = list(participant_params.values())
        participant_count = len(participant_updates)
        prototype, mask, aggregated_count, skipped_count, median_ok = self._median_values(
            participant_updates
        )

        if not median_ok:
            defense_result = self._build_result(
                participant_count=participant_count,
                aggregated_tensor_count=aggregated_count,
                skipped_key_count=skipped_count,
                fallback_reason="no_compatible_tensor_updates",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        updated_participant_params = {
            client_id: self._apply_median_mask(client_update, prototype, mask)
            for client_id, client_update in participant_params.items()
        }
        defense_result = self._build_result(
            participant_count=participant_count,
            aggregated_tensor_count=aggregated_count,
            skipped_key_count=skipped_count,
            median_applied=True,
        )
        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("median_applied", {})[self.name] = True
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "defense_type": "median",
                "num_rounds": 0,
                "rounds_with_median": 0,
                "total_aggregated_tensor_count": 0,
                "total_skipped_key_count": 0,
            }

        return {
            "defense_type": "median",
            "num_rounds": len(self.history),
            "rounds_with_median": sum(
                1 for item in self.history if item.get("median_applied")
            ),
            "total_aggregated_tensor_count": int(
                sum(int(item.get("aggregated_tensor_count", 0) or 0) for item in self.history)
            ),
            "total_skipped_key_count": int(
                sum(int(item.get("skipped_key_count", 0) or 0) for item in self.history)
            ),
        }


register_defense("median", MedianDefense)
register_defense("mediandefense", MedianDefense)
