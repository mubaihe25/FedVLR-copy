"""Unified non-targeted poisoning attack.

This module provides one front-end/API facing entry point for the existing
non-targeted poisoning strategies:
- client_update_scale
- sign_flip
- model_replacement

It intentionally does not implement targeted poisoning, target item attacks,
backdoors, or extra local training. Instead, it partitions the current round's
malicious clients across the enabled sub-strategies and applies exactly one
strategy to each malicious client before aggregation.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import torch

from attacks.base_attack import BaseAttack
from attacks.client_update_scale_attack import ClientUpdateScaleAttack
from attacks.model_replacement_attack import ModelReplacementAttack
from attacks.registry import register_attack
from attacks.sign_flip_attack import SignFlipAttack


class PoisoningAttack(BaseAttack):
    """Route malicious clients to non-targeted poisoning sub-strategies."""

    attack_family = "poisoning"
    attack_category = "poisoning"
    attack_strategy = "unified_nondirected_poisoning"
    attack_display_category = "投毒攻击"
    mutates_participant_params = True
    is_read_only = False

    DEFAULT_SUBSTRATEGIES = [
        "client_update_scale",
        "sign_flip",
        "model_replacement",
    ]

    STRATEGY_ALIASES = {
        "scale": "client_update_scale",
        "client_update_scale": "client_update_scale",
        "update_scale": "client_update_scale",
        "sign": "sign_flip",
        "sign_flip": "sign_flip",
        "flip": "sign_flip",
        "replacement": "model_replacement",
        "model_replacement": "model_replacement",
    }

    def __init__(
        self,
        name: str = "poisoning_attack",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.poisoning_mix_rule = str(
            self.config.get("poisoning_mix_rule", "round_robin")
        ).lower()
        self.enabled_substrategies = self._normalize_substrategies(
            self.config.get("poisoning_enabled_substrategies")
        )

        self.sub_attacks = {
            "client_update_scale": ClientUpdateScaleAttack(
                config={
                    "attack_scale": float(
                        self.config.get(
                            "poisoning_attack_scale",
                            self.config.get("attack_scale", 2.0),
                        )
                    )
                }
            ),
            "sign_flip": SignFlipAttack(
                config={
                    "sign_flip_scale": float(
                        self.config.get(
                            "poisoning_sign_flip_scale",
                            self.config.get("sign_flip_scale", 1.0),
                        )
                    )
                }
            ),
            "model_replacement": ModelReplacementAttack(
                config={
                    "replacement_scale": float(
                        self.config.get(
                            "poisoning_replacement_scale",
                            self.config.get("replacement_scale", 5.0),
                        )
                    ),
                    "replacement_rule": str(
                        self.config.get(
                            "poisoning_replacement_rule",
                            self.config.get("replacement_rule", "aligned_mean"),
                        )
                    ),
                }
            ),
        }

        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _normalize_substrategies(self, raw_value: Any) -> List[str]:
        if raw_value is None or raw_value == "":
            return list(self.DEFAULT_SUBSTRATEGIES)

        if isinstance(raw_value, str):
            raw_items = [item.strip() for item in raw_value.split(",")]
        elif isinstance(raw_value, Iterable):
            raw_items = [str(item).strip() for item in raw_value]
        else:
            raw_items = []

        normalized: List[str] = []
        for item in raw_items:
            key = self.STRATEGY_ALIASES.get(item.lower())
            if key and key not in normalized:
                normalized.append(key)

        return normalized or list(self.DEFAULT_SUBSTRATEGIES)

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

    def _participant_key_by_string(
        self, participant_params: MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        return {str(client_id): client_id for client_id in participant_params.keys()}

    def _partition_clients(
        self,
        malicious_clients: List[str],
        participant_params: MutableMapping[str, Any],
    ) -> Dict[str, List[str]]:
        present_client_ids = set(self._participant_key_by_string(participant_params).keys())
        target_clients = [
            str(client_id)
            for client_id in malicious_clients
            if str(client_id) in present_client_ids
        ]

        partitions = {strategy: [] for strategy in self.enabled_substrategies}
        if not target_clients or not self.enabled_substrategies:
            return partitions

        # The first engineering version intentionally keeps routing simple and
        # deterministic. Unsupported mix rules safely fall back to round robin.
        for index, client_id in enumerate(target_clients):
            strategy = self.enabled_substrategies[index % len(self.enabled_substrategies)]
            partitions[strategy].append(client_id)
        return partitions

    def _empty_result(self, fallback_reason: str = "no_malicious_clients") -> Dict[str, Any]:
        return {
            **self.semantic_metadata(),
            "poisoning_mode": "nondirected",
            "poisoning_mix_rule": self.poisoning_mix_rule,
            "effective_poisoning_mix_rule": "round_robin",
            "poisoning_enabled_substrategies": list(self.enabled_substrategies),
            "poisoned_clients": [],
            "poisoned_client_count": 0,
            "attacked_clients": [],
            "attacked_client_count": 0,
            "strategy_client_counts": {
                strategy: 0 for strategy in self.enabled_substrategies
            },
            "strategy_attacked_clients": {
                strategy: [] for strategy in self.enabled_substrategies
            },
            "strategy_metrics": {},
            "touched_update_count": 0,
            "avg_poisoned_norm_before": None,
            "avg_poisoned_norm_after": None,
            "fallback_reason": fallback_reason,
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict):
            attack_result = self._empty_result("unsupported_participant_params")
            self.last_round_output = attack_result
            self.history.append(attack_result)
            round_state.setdefault("attack_outputs", {})[self.name] = attack_result
            return participant_params

        malicious_clients = [
            str(client_id) for client_id in round_state.get("malicious_clients", [])
        ]
        if not malicious_clients:
            attack_result = self._empty_result()
            self.last_round_output = attack_result
            self.history.append(attack_result)
            round_state.setdefault("attack_outputs", {})[self.name] = attack_result
            return participant_params

        partitions = self._partition_clients(malicious_clients, participant_params)
        updated_participant_params: MutableMapping[str, Any] = dict(participant_params)
        key_by_string = self._participant_key_by_string(participant_params)

        strategy_metrics: Dict[str, Dict[str, Any]] = {}
        strategy_attacked_clients: Dict[str, List[str]] = {}
        strategy_client_counts: Dict[str, int] = {}

        for strategy, target_clients in partitions.items():
            strategy_client_counts[strategy] = len(target_clients)
            strategy_attacked_clients[strategy] = list(target_clients)
            if not target_clients:
                continue

            sub_attack = self.sub_attacks[strategy]
            sub_round_state = dict(round_state)
            sub_round_state["malicious_clients"] = list(target_clients)
            sub_round_state["attack_outputs"] = {}
            sub_round_state["attacked_clients"] = {}

            updated_participant_params = sub_attack.before_aggregation(
                updated_participant_params,
                sub_round_state,
            )
            strategy_metrics[strategy] = sub_attack.collect_metrics()

        poisoned_clients = [
            client_id
            for strategy in self.enabled_substrategies
            for client_id in strategy_attacked_clients.get(strategy, [])
        ]
        poisoned_client_set = set(poisoned_clients)
        norms_before: Dict[str, float] = {}
        norms_after: Dict[str, float] = {}

        for client_id in poisoned_clients:
            original_key = key_by_string.get(client_id)
            if original_key is None:
                continue
            norm_before = self._safe_norm(participant_params.get(original_key))
            norm_after = self._safe_norm(updated_participant_params.get(original_key))
            if norm_before is not None:
                norms_before[client_id] = norm_before
            if norm_after is not None:
                norms_after[client_id] = norm_after

        before_values = list(norms_before.values())
        after_values = list(norms_after.values())
        touched_update_count = int(
            sum(
                int(metrics.get("touched_update_count", 0))
                for metrics in strategy_metrics.values()
            )
        )

        attack_result = {
            **self.semantic_metadata(),
            "poisoning_mode": "nondirected",
            "poisoning_mix_rule": self.poisoning_mix_rule,
            "effective_poisoning_mix_rule": "round_robin",
            "poisoning_enabled_substrategies": list(self.enabled_substrategies),
            "poisoned_clients": poisoned_clients,
            "poisoned_client_count": len(poisoned_client_set),
            # Keep old active-attack naming for downstream compatibility.
            "attacked_clients": poisoned_clients,
            "attacked_client_count": len(poisoned_client_set),
            "strategy_client_counts": strategy_client_counts,
            "strategy_attacked_clients": strategy_attacked_clients,
            "strategy_metrics": strategy_metrics,
            "touched_update_count": touched_update_count,
            "avg_poisoned_norm_before": (
                float(sum(before_values) / len(before_values)) if before_values else None
            ),
            "avg_poisoned_norm_after": (
                float(sum(after_values) / len(after_values)) if after_values else None
            ),
        }

        self.last_round_output = attack_result
        self.history.append(attack_result)
        round_state.setdefault("attack_outputs", {})[self.name] = attack_result
        round_state.setdefault("attacked_clients", {})[self.name] = poisoned_clients
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
                "total_poisoned_clients": 0,
                "max_poisoned_client_count": 0,
                "strategy_client_counts_total": {},
                "strategy_round_coverage": {},
                "poisoning_mode": "nondirected",
                "poisoning_mix_rule": self.poisoning_mix_rule,
                "poisoning_enabled_substrategies": list(self.enabled_substrategies),
            }

        poisoned_counts = [
            int(item.get("poisoned_client_count", 0)) for item in self.history
        ]
        strategy_client_counts_total = {
            strategy: int(
                sum(
                    int(item.get("strategy_client_counts", {}).get(strategy, 0))
                    for item in self.history
                )
            )
            for strategy in self.enabled_substrategies
        }
        strategy_round_coverage = {
            strategy: int(
                sum(
                    1
                    for item in self.history
                    if int(item.get("strategy_client_counts", {}).get(strategy, 0)) > 0
                )
            )
            for strategy in self.enabled_substrategies
        }
        return {
            **self.semantic_metadata(),
            "num_rounds": len(self.history),
            "rounds_with_attacks": sum(1 for count in poisoned_counts if count > 0),
            "total_poisoned_clients": int(sum(poisoned_counts)),
            "max_poisoned_client_count": int(max(poisoned_counts)),
            "strategy_client_counts_total": strategy_client_counts_total,
            "strategy_round_coverage": strategy_round_coverage,
            "poisoning_mode": "nondirected",
            "poisoning_mix_rule": self.poisoning_mix_rule,
            "poisoning_enabled_substrategies": list(self.enabled_substrategies),
            "avg_poisoned_norm_before": self._average_history_value(
                "avg_poisoned_norm_before"
            ),
            "avg_poisoned_norm_after": self._average_history_value(
                "avg_poisoned_norm_after"
            ),
        }

    def _average_history_value(self, key: str) -> Optional[float]:
        values = [
            float(item[key])
            for item in self.history
            if item.get(key) is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))


register_attack("poisoning_attack", PoisoningAttack)
register_attack("poisoning", PoisoningAttack)
register_attack("nondirected_poisoning", PoisoningAttack)
register_attack("poisoningattack", PoisoningAttack)
