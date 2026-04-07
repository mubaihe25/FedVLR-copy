"""Minimal runtime hooks for future security extensions.

This module is intentionally lightweight:
- no attack/defense logic is implemented
- no training behavior is changed by default
- only in-memory round-level placeholders are collected
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from attacks.base_attack import BaseAttack
from defenses.base_defense import BaseDefense
from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.result_schema import ExperimentResult, RoundMetric, build_empty_result


class ExperimentHookManager:
    """Runtime container for future attack/defense/privacy extensions."""

    def __init__(self, config) -> None:
        self.config = config
        self.enabled = bool(config.get("enable_experiment_hooks", False))
        self.collect_round_metrics = bool(config.get("collect_round_metrics", True))

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
                "type": config.get("type"),
                "comment": config.get("comment"),
            }
        )

        malicious_clients = config.get("malicious_clients", []) or []
        self.result.malicious_clients = [str(client_id) for client_id in malicious_clients]

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
        metric = RoundMetric(round_index=round_index)
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

    def start_round(self, round_index: int, sampled_clients: List[Any]) -> Dict[str, Any]:
        round_state = self._get_round_state(round_index)
        round_state["sampled_clients"] = [str(client_id) for client_id in sampled_clients]
        round_state["participant_count"] = len(sampled_clients)

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

        return participant_params

    def finish_train_round(
        self, round_index: int, train_loss: Optional[float], participant_count: int
    ) -> None:
        metric = self._upsert_round_metric(round_index)
        round_state = self._get_round_state(round_index)

        metric.participant_count = participant_count
        metric.malicious_client_count = len(self.result.malicious_clients)
        metric.train_loss = None if train_loss is None else float(train_loss)
        metric.extra.update(
            {
                "sampled_clients": round_state.get("sampled_clients", []),
                "client_losses": round_state.get("client_losses", {}),
            }
        )

    def record_epoch_exit(
        self,
        round_index: int,
        train_loss: Optional[float],
        valid_result: Optional[Dict[str, Any]] = None,
        test_result: Optional[Dict[str, Any]] = None,
        stop_flag: bool = False,
    ) -> None:
        metric = self._upsert_round_metric(round_index)
        if train_loss is not None and metric.train_loss is None:
            metric.train_loss = float(train_loss)

        metric.extra.update(
            {
                "stop_flag": bool(stop_flag),
                "valid_result": valid_result or {},
                "test_result": test_result or {},
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
        return self.result

    def to_dict(self) -> Dict[str, Any]:
        return self.result.to_dict()
