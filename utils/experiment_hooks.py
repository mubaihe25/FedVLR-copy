"""Minimal runtime hooks for future security extensions.

This module is intentionally lightweight:
- no attack/defense logic is implemented
- no training behavior is changed by default
- only in-memory round-level placeholders are collected
"""

from __future__ import annotations

from datetime import datetime
import math
import random
from typing import Any, Dict, List, Optional

from attacks import get_attack, list_attacks
from attacks.base_attack import BaseAttack
from defenses import get_defense, list_defenses
from defenses.base_defense import BaseDefense
from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval import get_privacy_metric, list_privacy_metrics
from privacy_eval.result_schema import ExperimentResult, RoundMetric, build_empty_result


class ExperimentHookManager:
    """Runtime container for future attack/defense/privacy extensions."""

    def __init__(self, config) -> None:
        self.config = config
        self.enabled = bool(config.get("enable_experiment_hooks", False))
        self.collect_round_metrics = bool(config.get("collect_round_metrics", True))
        self.enable_malicious_clients = bool(
            config.get("enable_malicious_clients", False)
        )
        self.malicious_client_mode = str(
            config.get("malicious_client_mode", "none")
        ).lower()
        self.malicious_client_ratio = float(config.get("malicious_client_ratio", 0.0) or 0.0)
        configured_malicious_ids = (
            config.get("malicious_client_ids")
            or config.get("malicious_clients")
            or []
        )
        self.configured_malicious_client_ids = [
            str(client_id) for client_id in configured_malicious_ids
        ]
        self.base_seed = int(config.get("seed", 0) or 0)
        self.enabled_attacks = self._normalize_module_names(
            config.get("enabled_attacks", [])
        )
        self.enabled_defenses = self._normalize_module_names(
            config.get("enabled_defenses", [])
        )
        self.enabled_privacy_metrics = self._normalize_module_names(
            config.get("enabled_privacy_metrics", [])
        )

        self.attacks: List[BaseAttack] = []
        self.defenses: List[BaseDefense] = []
        self.privacy_metrics: List[BasePrivacyMetric] = []
        self.round_states: Dict[int, Dict[str, Any]] = {}

        experiment_id = config.get("experiment_id") or self._build_experiment_id()
        self.result: ExperimentResult = build_empty_result(
            experiment_id=experiment_id,
            model=config["model"],
            dataset=config["dataset"],
        )
        self.result.attack_type = config.get("attack_type")
        self.result.defense_type = config.get("defense_type")
        self.result.metadata.update(
            {
                "hooks_enabled": self.enabled,
                "round_metrics_enabled": self.collect_round_metrics,
                "enable_malicious_clients": self.enable_malicious_clients,
                "malicious_client_mode": self.malicious_client_mode,
                "malicious_client_ratio": self.malicious_client_ratio,
                "configured_malicious_client_ids": list(
                    self.configured_malicious_client_ids
                ),
                "enabled_attacks": list(self.enabled_attacks),
                "enabled_defenses": list(self.enabled_defenses),
                "enabled_privacy_metrics": list(self.enabled_privacy_metrics),
                "type": config.get("type"),
                "comment": config.get("comment"),
            }
        )

        self.attacks, unknown_attacks = self._load_attack_instances(self.enabled_attacks)
        self.defenses, unknown_defenses = self._load_defense_instances(
            self.enabled_defenses
        )
        self.privacy_metrics, unknown_privacy_metrics = self._load_privacy_metric_instances(
            self.enabled_privacy_metrics
        )

        self.result.metadata.update(
            {
                "loaded_attacks": [attack.name for attack in self.attacks],
                "loaded_defenses": [defense.name for defense in self.defenses],
                "loaded_privacy_metrics": [metric.name for metric in self.privacy_metrics],
                "unknown_attacks": unknown_attacks,
                "unknown_defenses": unknown_defenses,
                "unknown_privacy_metrics": unknown_privacy_metrics,
                "available_attacks": list_attacks(),
                "available_defenses": list_defenses(),
                "available_privacy_metrics": list_privacy_metrics(),
            }
        )

        self.result.malicious_clients = []

    def _normalize_module_names(self, names: Any) -> List[str]:
        if names is None:
            return []
        if isinstance(names, str):
            return [names.strip()] if names.strip() else []
        normalized = []
        for name in names:
            if name is None:
                continue
            name_str = str(name).strip()
            if name_str:
                normalized.append(name_str)
        return normalized

    def _load_attack_instances(self, names: List[str]) -> tuple[List[BaseAttack], List[str]]:
        attacks: List[BaseAttack] = []
        unknown_names: List[str] = []
        for name in names:
            try:
                attack_cls = get_attack(name)
                attacks.append(attack_cls(name=name, config=self.config))
            except KeyError:
                unknown_names.append(name)
        return attacks, unknown_names

    def _load_defense_instances(self, names: List[str]) -> tuple[List[BaseDefense], List[str]]:
        defenses: List[BaseDefense] = []
        unknown_names: List[str] = []
        for name in names:
            try:
                defense_cls = get_defense(name)
                defenses.append(defense_cls(name=name, config=self.config))
            except KeyError:
                unknown_names.append(name)
        return defenses, unknown_names

    def _load_privacy_metric_instances(
        self, names: List[str]
    ) -> tuple[List[BasePrivacyMetric], List[str]]:
        metrics: List[BasePrivacyMetric] = []
        unknown_names: List[str] = []
        for name in names:
            try:
                metric_cls = get_privacy_metric(name)
                metrics.append(metric_cls(name=name, config=self.config))
            except KeyError:
                unknown_names.append(name)
        return metrics, unknown_names

    def _collect_attack_metrics(self) -> Dict[str, Any]:
        return {attack.name: attack.collect_metrics() for attack in self.attacks}

    def _collect_defense_metrics(self) -> Dict[str, Any]:
        return {defense.name: defense.collect_metrics() for defense in self.defenses}

    def _collect_privacy_metrics(
        self,
        round_state: Dict[str, Any],
        participant_params: Any,
        aggregation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        aggregation_result = aggregation_result or {}
        return {
            metric.name: metric.evaluate_round(
                round_state, participant_params, aggregation_result
            )
            for metric in self.privacy_metrics
        }

    def _build_experiment_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return "{}-{}-{}".format(
            self.config["model"],
            self.config["dataset"],
            timestamp,
        )

    def _get_round_state(self, round_index: int) -> Dict[str, Any]:
        return self.round_states.setdefault(
            round_index,
            {
                "round_index": round_index,
                "sampled_clients": [],
                "client_losses": {},
                "participant_count": 0,
            },
        )

    def _upsert_round_metric(self, round_index: int) -> RoundMetric:
        for metric in self.result.round_metrics:
            if metric.round_index == round_index:
                return metric
        metric = RoundMetric(round_index=round_index, round_id=round_index)
        self.result.round_metrics.append(metric)
        return metric

    def _extract_metric(self, metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not metrics:
            return None
        for candidate in (key, key.lower(), key.upper()):
            if candidate in metrics:
                value = metrics[candidate]
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _update_global_malicious_clients(self, round_malicious_clients: List[str]) -> None:
        existing = list(self.result.malicious_clients)
        for client_id in round_malicious_clients:
            if client_id not in existing:
                existing.append(client_id)
        self.result.malicious_clients = existing

    def _resolve_round_malicious_clients(
        self, round_index: int, participant_clients: List[str]
    ) -> List[str]:
        if not self.enable_malicious_clients:
            return []

        if self.configured_malicious_client_ids:
            configured = set(self.configured_malicious_client_ids)
            return [client_id for client_id in participant_clients if client_id in configured]

        if self.malicious_client_mode == "ratio" and self.malicious_client_ratio > 0:
            num_participants = len(participant_clients)
            if num_participants == 0:
                return []

            num_malicious = min(
                num_participants,
                max(1, int(math.ceil(num_participants * self.malicious_client_ratio))),
            )
            rng = random.Random(self.base_seed + round_index)
            return rng.sample(participant_clients, num_malicious)

        return []

    def start_round(self, round_index: int, sampled_clients: List[Any]) -> Dict[str, Any]:
        round_state = self._get_round_state(round_index)
        participant_clients = [str(client_id) for client_id in sampled_clients]
        round_malicious_clients = self._resolve_round_malicious_clients(
            round_index, participant_clients
        )
        round_state["sampled_clients"] = participant_clients
        round_state["participant_count"] = len(sampled_clients)
        round_state["malicious_clients"] = list(round_malicious_clients)
        self._update_global_malicious_clients(round_malicious_clients)

        if self.collect_round_metrics:
            metric = self._upsert_round_metric(round_index)
            metric.round_id = round_index
            metric.participant_clients = participant_clients
            metric.num_participants = len(sampled_clients)
            metric.participant_count = len(sampled_clients)
            metric.hooks_enabled = self.enabled
            metric.malicious_clients = list(round_malicious_clients)
            metric.malicious_client_count = len(round_malicious_clients)
            metric.extra.update(
                {
                    "loaded_attacks": [attack.name for attack in self.attacks],
                    "loaded_defenses": [defense.name for defense in self.defenses],
                    "loaded_privacy_metrics": [
                        metric_obj.name for metric_obj in self.privacy_metrics
                    ],
                }
            )

        if self.enabled:
            for attack in self.attacks:
                round_state = attack.before_round(round_state)
            for defense in self.defenses:
                round_state = defense.before_round(round_state)

        return round_state

    def record_client_train(
        self, round_index: int, client_id: Any, client_losses: List[float]
    ) -> None:
        if not self.collect_round_metrics:
            return

        round_state = self._get_round_state(round_index)
        round_state["client_losses"][str(client_id)] = [float(loss) for loss in client_losses]

    def after_local_train(self, round_index: int, client_id: Any, client_update: Any) -> Any:
        round_state = self._get_round_state(round_index)

        if self.enabled:
            for attack in self.attacks:
                client_update = attack.after_local_train(client_id, client_update, round_state)
            for defense in self.defenses:
                client_update = defense.inspect_client_update(
                    client_id, client_update, round_state
                )

        return client_update

    def before_aggregation(self, round_index: int, participant_params: Any) -> Any:
        round_state = self._get_round_state(round_index)

        if self.enabled:
            for attack in self.attacks:
                participant_params = attack.before_aggregation(participant_params, round_state)
            for defense in self.defenses:
                participant_params = defense.before_aggregation(
                    participant_params, round_state
                )

        round_state["privacy_metric_outputs"] = self._collect_privacy_metrics(
            round_state=round_state,
            participant_params=participant_params,
            aggregation_result={},
        )
        return participant_params

    def finish_train_round(
        self, round_index: int, train_loss: Optional[float], participant_count: int
    ) -> None:
        if not self.collect_round_metrics:
            return

        metric = self._upsert_round_metric(round_index)
        round_state = self._get_round_state(round_index)
        round_malicious_clients = list(round_state.get("malicious_clients", []))

        metric.round_id = round_index
        metric.participant_clients = list(round_state.get("sampled_clients", []))
        metric.num_participants = participant_count
        metric.participant_count = participant_count
        metric.avg_train_loss = None if train_loss is None else float(train_loss)
        metric.malicious_client_count = len(round_malicious_clients)
        metric.malicious_clients = round_malicious_clients
        metric.hooks_enabled = self.enabled
        metric.train_loss = None if train_loss is None else float(train_loss)
        metric.extra.update(
            {
                "sampled_clients": round_state.get("sampled_clients", []),
                "malicious_clients": round_malicious_clients,
                "client_losses": round_state.get("client_losses", {}),
                "attack_metrics": self._collect_attack_metrics(),
                "defense_metrics": self._collect_defense_metrics(),
                "privacy_metric_outputs": round_state.get("privacy_metric_outputs", {}),
            }
        )

    def record_epoch_exit(
        self,
        round_index: int,
        train_loss: Optional[float],
        valid_score: Optional[float] = None,
        test_score: Optional[float] = None,
        valid_result: Optional[Dict[str, Any]] = None,
        test_result: Optional[Dict[str, Any]] = None,
        stop_flag: bool = False,
    ) -> None:
        if not self.collect_round_metrics:
            return

        metric = self._upsert_round_metric(round_index)
        round_state = self._get_round_state(round_index)
        round_malicious_clients = list(round_state.get("malicious_clients", []))
        metric.round_id = round_index
        metric.hooks_enabled = self.enabled
        metric.malicious_clients = round_malicious_clients
        metric.malicious_client_count = len(round_malicious_clients)
        if train_loss is not None and metric.train_loss is None:
            metric.train_loss = float(train_loss)
        if train_loss is not None and metric.avg_train_loss is None:
            metric.avg_train_loss = float(train_loss)
        metric.valid_score = None if valid_score is None else float(valid_score)
        metric.test_score = None if test_score is None else float(test_score)

        metric.extra.update(
            {
                "stop_flag": bool(stop_flag),
                "valid_result": valid_result or {},
                "test_result": test_result or {},
                "malicious_clients": round_malicious_clients,
                "loaded_attacks": [attack.name for attack in self.attacks],
                "loaded_defenses": [defense.name for defense in self.defenses],
                "loaded_privacy_metrics": [
                    metric_obj.name for metric_obj in self.privacy_metrics
                ],
                "privacy_metric_outputs": round_state.get("privacy_metric_outputs", {}),
            }
        )

    def finalize_experiment(
        self,
        best_valid_result: Optional[Dict[str, Any]],
        best_test_result: Optional[Dict[str, Any]],
    ) -> ExperimentResult:
        self.result.final_eval.recall20 = self._extract_metric(best_test_result, "recall@20")
        self.result.final_eval.ndcg20 = self._extract_metric(best_test_result, "ndcg@20")

        if self.result.round_metrics:
            self.result.final_eval.loss = self.result.round_metrics[-1].train_loss

        self.result.metadata["best_valid_result"] = best_valid_result or {}
        self.result.metadata["best_test_result"] = best_test_result or {}
        self.result.metadata["privacy_metric_summaries"] = {
            metric.name: metric.summarize(self.result.metadata)
            for metric in self.privacy_metrics
        }
        return self.result

    def to_dict(self) -> Dict[str, Any]:
        return self.result.to_dict()
