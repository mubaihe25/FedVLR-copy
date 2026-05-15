"""Simplified Bulyan robust aggregation defense.

Bulyan is usually defined with strict client-count assumptions. This module
implements a small, explainable version for FedVLR: select a candidate set with
Multi-Krum-style scores, then apply coordinate-wise trimmed mean to that set.
When the client count is too small, it reports a fallback instead of failing.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional, Tuple

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense
from defenses.robust_aggregation_utils import (
    config_get,
    fallback_trim_count,
    krum_scores,
    median_values,
    optional_int,
    participant_vectors,
    replace_all_with_prototype,
    safe_byzantine_count,
    score_summary,
    trimmed_mean_values,
)


class BulyanDefense(BaseDefense):
    """Apply a simplified Multi-Krum plus trimmed-mean Bulyan variant."""

    def __init__(
        self,
        name: str = "bulyan",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        configured_byzantine = config_get(config, "byzantine_count", None)
        if configured_byzantine is None:
            configured_byzantine = config_get(config, "estimated_malicious_count", None)
        self.estimated_malicious_count = optional_int(configured_byzantine)
        self.candidate_count = optional_int(
            config_get(config, "bulyan_candidate_count", config_get(config, "candidate_count", None))
        )
        self.trim_ratio = float(config_get(config, "trim_ratio", 0.2))
        self.distance_norm = str(config_get(config, "distance_norm", "l2")).lower()
        self.fallback_rule = str(config_get(config, "fallback_rule", "trimmed_mean")).lower()
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _candidate_count(self, participant_count: int, byzantine_count: int) -> int:
        if self.candidate_count is not None:
            return max(1, min(int(self.candidate_count), participant_count))
        return max(1, min(participant_count - 2 * byzantine_count, participant_count))

    def _build_result(
        self,
        participant_count: int,
        selected_indices: Optional[List[int]] = None,
        rejected_indices: Optional[List[int]] = None,
        estimated_malicious_count: int = 0,
        candidate_count: int = 0,
        trimmed_coordinate_count: int = 0,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
        distance_summary: Optional[Dict[str, Any]] = None,
        skipped_key_count: int = 0,
    ) -> Dict[str, Any]:
        selected_indices = selected_indices or []
        rejected_indices = rejected_indices or []
        status = "skipped_or_fallback" if fallback_used else "applied"
        return {
            "defense_type": "bulyan",
            "status": status,
            "participant_count": participant_count,
            "selected_client_count": len(selected_indices),
            "candidate_count": candidate_count,
            "rejected_client_count": len(rejected_indices),
            "selected_indices": selected_indices,
            "rejected_indices": rejected_indices,
            "estimated_malicious_count": estimated_malicious_count,
            "trimmed_coordinate_count": int(trimmed_coordinate_count),
            "fallback_used": bool(fallback_used),
            "fallback_rule": self.fallback_rule,
            "fallback_reason": fallback_reason,
            "insufficient_clients_for_bulyan": bool(
                fallback_reason == "insufficient_clients_for_bulyan"
            ),
            "distance_norm": self.distance_norm,
            "distance_summary": distance_summary or {},
            "skipped_key_count": skipped_key_count,
            "note": "simplified Bulyan: Multi-Krum candidate selection plus coordinate-wise trimmed mean",
        }

    def _fallback_prototype(
        self,
        values: List[Any],
        participant_count: int,
    ) -> Tuple[Any, int, int, bool, str]:
        if self.fallback_rule == "median":
            prototype, touched, skipped, ok = median_values(values)
            return prototype, touched, skipped, ok, "median"
        trim_count = fallback_trim_count(participant_count, self.trim_ratio)
        prototype, touched, skipped, ok = trimmed_mean_values(values, trim_count)
        if ok:
            return prototype, touched, skipped, ok, "trimmed_mean"
        prototype, touched, skipped, ok = median_values(values)
        return prototype, touched, skipped, ok, "median"

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict) or not participant_params:
            result = self._build_result(
                participant_count=0,
                fallback_used=True,
                fallback_reason="empty_or_unsupported_participant_params",
            )
            self.last_round_output = result
            self.history.append(result)
            round_state.setdefault("defense_outputs", {})[self.name] = result
            return participant_params

        participant_items, vectors, skipped_key_count, flatten_error = participant_vectors(
            participant_params
        )
        participant_count = len(participant_items)
        byzantine_count = safe_byzantine_count(
            self.estimated_malicious_count,
            participant_count,
            round_state,
        )
        if flatten_error:
            result = self._build_result(
                participant_count=participant_count,
                estimated_malicious_count=byzantine_count,
                fallback_used=True,
                fallback_reason=flatten_error,
                skipped_key_count=skipped_key_count,
            )
            self.last_round_output = result
            self.history.append(result)
            round_state.setdefault("defense_outputs", {})[self.name] = result
            return participant_params

        theoretical_min_clients = 4 * byzantine_count + 3 if byzantine_count > 0 else 3
        all_updates = [client_update for _, client_update in participant_items]
        if participant_count < theoretical_min_clients:
            prototype, touched, skipped, ok, fallback_rule = self._fallback_prototype(
                all_updates,
                participant_count,
            )
            result = self._build_result(
                participant_count=participant_count,
                selected_indices=list(range(participant_count)) if ok else [],
                rejected_indices=[],
                estimated_malicious_count=byzantine_count,
                candidate_count=participant_count if ok else 0,
                trimmed_coordinate_count=touched,
                fallback_used=True,
                fallback_reason="insufficient_clients_for_bulyan",
                skipped_key_count=skipped_key_count + skipped,
            )
            result["fallback_rule"] = fallback_rule
            self.last_round_output = result
            self.history.append(result)
            round_state.setdefault("defense_outputs", {})[self.name] = result
            if ok:
                return replace_all_with_prototype(participant_params, prototype)
            return participant_params

        scores = krum_scores(vectors, byzantine_count, self.distance_norm)
        candidate_count = self._candidate_count(participant_count, byzantine_count)
        selected_indices = [index for index, _ in scores[:candidate_count]]
        selected_set = set(selected_indices)
        rejected_indices = [
            index for index in range(participant_count) if index not in selected_set
        ]
        candidate_updates = [
            client_update
            for index, (_, client_update) in enumerate(participant_items)
            if index in selected_set
        ]
        trim_count = min(byzantine_count, max(0, (len(candidate_updates) - 1) // 2))
        prototype, touched, skipped, ok = trimmed_mean_values(candidate_updates, trim_count)
        if not ok:
            prototype, touched, skipped, ok = median_values(candidate_updates)
            fallback_used = True
            fallback_reason = "candidate_trimmed_mean_failed"
        else:
            fallback_used = False
            fallback_reason = None

        result = self._build_result(
            participant_count=participant_count,
            selected_indices=selected_indices,
            rejected_indices=rejected_indices,
            estimated_malicious_count=byzantine_count,
            candidate_count=candidate_count,
            trimmed_coordinate_count=touched * max(1, 2 * trim_count),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            distance_summary=score_summary(scores),
            skipped_key_count=skipped_key_count + skipped,
        )
        result["selected_clients"] = [
            str(client_id)
            for index, (client_id, _) in enumerate(participant_items)
            if index in selected_set
        ]
        result["rejected_clients"] = [
            str(client_id)
            for index, (client_id, _) in enumerate(participant_items)
            if index not in selected_set
        ]

        self.last_round_output = result
        self.history.append(result)
        round_state.setdefault("defense_outputs", {})[self.name] = result
        round_state.setdefault("filtered_clients", {})[self.name] = result["rejected_clients"]
        if ok:
            return replace_all_with_prototype(participant_params, prototype)
        return participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "defense_type": "bulyan",
                "num_rounds": 0,
                "rounds_applied": 0,
                "fallback_used": False,
                "estimated_malicious_count": self.estimated_malicious_count,
            }
        return {
            "defense_type": "bulyan",
            "num_rounds": len(self.history),
            "rounds_applied": sum(
                1 for item in self.history if item.get("status") == "applied"
            ),
            "rounds_with_fallback": sum(
                1 for item in self.history if item.get("fallback_used")
            ),
            "total_selected_clients": int(
                sum(int(item.get("selected_client_count", 0) or 0) for item in self.history)
            ),
            "last_selected_indices": list(self.history[-1].get("selected_indices", [])),
            "last_candidate_count": self.history[-1].get("candidate_count"),
            "last_trimmed_coordinate_count": self.history[-1].get("trimmed_coordinate_count"),
            "fallback_used": bool(self.history[-1].get("fallback_used")),
            "fallback_reason": self.history[-1].get("fallback_reason"),
            "estimated_malicious_count": self.estimated_malicious_count,
            "note": "simplified Bulyan; falls back instead of failing when client count is insufficient",
        }


register_defense("bulyan", BulyanDefense)
register_defense("bulyan_defense", BulyanDefense)
