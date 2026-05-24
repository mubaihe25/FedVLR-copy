"""Central DP-style noise defense for uploaded client updates.

This module clips each client update before aggregation and adds Gaussian noise
to compatible floating-point tensor leaves. It is a central noise defense only:
there is no formal differential privacy accountant, epsilon/delta reporting, or
client-side DP-SGD implementation here.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class DPNoiseDefense(BaseDefense):
    """Clip uploaded updates and add Gaussian noise before aggregation."""

    def __init__(
        self,
        name: str = "dp_noise",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.clip_norm = float(
            self._config_get(config, "clip_norm", self._config_get(config, "defense_clip_norm", 5.0))
        )
        self.noise_multiplier = float(self._config_get(config, "noise_multiplier", 0.0))
        configured_noise_std = self._config_get(config, "noise_std", None)
        self.noise_std = (
            float(configured_noise_std)
            if configured_noise_std is not None
            else float(self.clip_norm * self.noise_multiplier)
        )
        self.seed = self._optional_int(self._config_get(config, "seed", None))
        self._generators: Dict[str, torch.Generator] = {}
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _config_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                try:
                    value = getter(key)
                except Exception:
                    return default
                return default if value is None else value
        return getattr(config, key, default)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _sum_squared_norm(self, value: Any) -> float:
        if value is None:
            return 0.0

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().values()
            if not torch.is_floating_point(tensor):
                return 0.0
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
            if value.is_sparse or not torch.is_floating_point(value):
                return value.clone(), 0
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

    def _generator_for_device(self, device: torch.device) -> Optional[torch.Generator]:
        if self.seed is None:
            return None

        key = str(device)
        if key not in self._generators:
            try:
                generator = torch.Generator(device=device)
            except Exception:
                generator = torch.Generator()
            generator.manual_seed(int(self.seed) + len(self._generators))
            self._generators[key] = generator
        return self._generators[key]

    def _randn_like(self, value: torch.Tensor) -> torch.Tensor:
        generator = self._generator_for_device(value.device)
        kwargs = {
            "size": value.shape,
            "device": value.device,
            "dtype": value.dtype,
        }
        if generator is not None:
            kwargs["generator"] = generator
        try:
            return torch.randn(**kwargs)
        except TypeError:
            kwargs.pop("generator", None)
            return torch.randn(**kwargs)

    def _add_noise(self, value: Any) -> Tuple[Any, int, int]:
        if value is None:
            return value, 0, 0

        if torch.is_tensor(value):
            if value.is_sparse or not torch.is_floating_point(value):
                return value.clone(), 0, 1
            if self.noise_std <= 0:
                return value.clone(), 0, 0
            noise = self._randn_like(value) * self.noise_std
            return value + noise, 1, 0

        if isinstance(value, dict):
            noised_dict: Dict[Any, Any] = {}
            noised_count = 0
            skipped_count = 0
            for key, item in value.items():
                noised_item, item_noised, item_skipped = self._add_noise(item)
                noised_dict[key] = noised_item
                noised_count += item_noised
                skipped_count += item_skipped
            return noised_dict, noised_count, skipped_count

        if isinstance(value, list):
            noised_list: List[Any] = []
            noised_count = 0
            skipped_count = 0
            for item in value:
                noised_item, item_noised, item_skipped = self._add_noise(item)
                noised_list.append(noised_item)
                noised_count += item_noised
                skipped_count += item_skipped
            return noised_list, noised_count, skipped_count

        if isinstance(value, tuple):
            noised_items: List[Any] = []
            noised_count = 0
            skipped_count = 0
            for item in value:
                noised_item, item_noised, item_skipped = self._add_noise(item)
                noised_items.append(noised_item)
                noised_count += item_noised
                skipped_count += item_skipped
            return tuple(noised_items), noised_count, skipped_count

        return value, 0, 0

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict):
            defense_result = self._build_result(participant_count=0)
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        updated_participant_params = dict(participant_params)
        clipped_clients: List[str] = []
        norms_before: Dict[str, float] = {}
        norms_after: Dict[str, float] = {}
        noised_tensor_count = 0
        skipped_tensor_count = 0

        for client_id, client_update in participant_params.items():
            client_id_str = str(client_id)
            norm_before = self._safe_norm(client_update)
            working_update = client_update
            if norm_before is not None:
                norms_before[client_id_str] = norm_before
                if norm_before > self.clip_norm:
                    scale_factor = self.clip_norm / max(norm_before, 1e-12)
                    working_update, _ = self._scale_value(client_update, scale_factor)
                    clipped_clients.append(client_id_str)

            noised_update, item_noised_count, item_skipped_count = self._add_noise(
                working_update
            )
            noised_tensor_count += item_noised_count
            skipped_tensor_count += item_skipped_count
            updated_participant_params[client_id] = noised_update

            norm_after = self._safe_norm(noised_update)
            if norm_after is not None:
                norms_after[client_id_str] = norm_after

        defense_result = self._build_result(
            participant_count=len(participant_params),
            clipped_clients=clipped_clients,
            noised_tensor_count=noised_tensor_count,
            skipped_tensor_count=skipped_tensor_count,
            norms_before=norms_before,
            norms_after=norms_after,
        )
        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("clipped_clients", {})[self.name] = clipped_clients
        return updated_participant_params

    def _build_result(
        self,
        participant_count: int,
        clipped_clients: Optional[List[str]] = None,
        noised_tensor_count: int = 0,
        skipped_tensor_count: int = 0,
        norms_before: Optional[Dict[str, float]] = None,
        norms_after: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        clipped_clients = clipped_clients or []
        return {
            "defense_type": "dp_noise",
            "participant_count": participant_count,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "noise_std": self.noise_std,
            "clipped_clients": clipped_clients,
            "clipped_client_count": len(clipped_clients),
            "noised_tensor_count": noised_tensor_count,
            "skipped_tensor_count": skipped_tensor_count,
            "norms_before": norms_before or {},
            "norms_after": norms_after or {},
            "privacy_mode": "central_dp_style_update_noise",
            "formal_accountant": False,
            "recommended_formal_dp_path": "opacus_toy_or_future_fedavg_adapter",
            "utility_privacy_tradeoff_note": (
                "larger noise may reduce utility; this module reports no epsilon/delta"
            ),
            "opacus_compatible_future_work": True,
            "note": "central noise defense, not full formal DP accountant",
        }

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "defense_type": "dp_noise",
                "num_rounds": 0,
                "clip_norm": self.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "noise_std": self.noise_std,
                "clipped_client_count": 0,
                "noised_tensor_count": 0,
                "total_clipped_clients": 0,
                "total_noised_tensor_count": 0,
                "privacy_mode": "central_dp_style_update_noise",
                "formal_accountant": False,
                "recommended_formal_dp_path": "opacus_toy_or_future_fedavg_adapter",
                "utility_privacy_tradeoff_note": (
                    "larger noise may reduce utility; this module reports no epsilon/delta"
                ),
                "opacus_compatible_future_work": True,
                "note": "central noise defense, not full formal DP accountant",
            }

        clipped_total = int(
            sum(int(item.get("clipped_client_count", 0) or 0) for item in self.history)
        )
        noised_total = int(
            sum(int(item.get("noised_tensor_count", 0) or 0) for item in self.history)
        )
        return {
            "defense_type": "dp_noise",
            "num_rounds": len(self.history),
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "noise_std": self.noise_std,
            "clipped_client_count": clipped_total,
            "noised_tensor_count": noised_total,
            "total_clipped_clients": clipped_total,
            "total_noised_tensor_count": noised_total,
            "privacy_mode": "central_dp_style_update_noise",
            "formal_accountant": False,
            "recommended_formal_dp_path": "opacus_toy_or_future_fedavg_adapter",
            "utility_privacy_tradeoff_note": (
                "larger noise may reduce utility; this module reports no epsilon/delta"
            ),
            "opacus_compatible_future_work": True,
            "note": "central noise defense, not full formal DP accountant",
        }


register_defense("dp_noise", DPNoiseDefense)
register_defense("dpnoise", DPNoiseDefense)
register_defense("dpnoisedefense", DPNoiseDefense)
