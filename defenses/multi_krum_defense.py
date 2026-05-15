"""Multi-Krum robust aggregation defense.

This is a lightweight before_aggregation implementation based on the common
Krum idea: flatten compatible client updates, score each client by distances to
near neighbors, keep several majority-like updates, then aggregate the selected
updates. It is designed for FedVLR smoke and showcase experiments, not as a
formal Byzantine benchmark suite.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense
from defenses.robust_aggregation_utils import (
    config_get,
    krum_scores,
    mean_values,
    optional_int,
    participant_vectors,
    replace_all_with_prototype,
    safe_byzantine_count,
    score_summary,
)


class MultiKrumDefense(BaseDefense):
    """Select several Krum-scored clients and aggregate them."""

    def __init__(
        self,
        name: str = "multi_krum",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        configured_byzantine = config_get(config, "byzantine_count", None)
        if configured_byzantine is None:
            configured_byzantine = config_get(config, "estimated_malicious_count", None)
        self.estimated_malicious_count = optional_int(configured_byzantine)
        self.multi_krum_k = optional_int(
            config_get(config, "multi_krum_k", config_get(config, "selected_client_count", None))
        )
        self.distance_norm = str(config_get(config, "distance_norm", "l2")).lower()
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _selected_count(self, participant_count: int, byzantine_count: int) -> int:
        if participant_count <= 0:
            return 0
        if self.multi_krum_k is not None:
            return max(1, min(int(self.multi_krum_k), participant_count))
        theoretical_default = max(1, participant_count - byzantine_count - 2)
        return max(1, min(theoretical_default, participant_count))

    def _build_result(
        self,
        participant_count: int,
        selected_indices: Optional[List[int]] = None,
        rejected_indices: Optional[List[int]] = None,
        estimated_malicious_count: int = 0,
        fallback_reason: Optional[str] = None,
        distance_summary: Optional[Dict[str, Any]] = None,
        aggregated_tensor_count: int = 0,
        skipped_key_count: int = 0,
    ) -> Dict[str, Any]:
        selected_indices = selected_indices or []
        rejected_indices = rejected_indices or []
        return {
            "defense_type": "multi_krum",
            "participant_count": participant_count,
            "selected_client_count": len(selected_indices),
            "rejected_client_count": len(rejected_indices),
            "selected_indices": selected_indices,
            "rejected_indices": rejected_indices,
            "estimated_malicious_count": estimated_malicious_count,
            "multi_krum_k": self.multi_krum_k,
            "distance_norm": self.distance_norm,
            "distance_summary": distance_summary or {},
            "aggregated_tensor_count": aggregated_tensor_count,
            "skipped_key_count": skipped_key_count,
            "fallback_reason": fallback_reason,
            "note": "lightweight Multi-Krum-style selector; not a full Byzantine benchmark implementation",
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict) or not participant_params:
            result = self._build_result(
                participant_count=0,
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
                fallback_reason=flatten_error,
                skipped_key_count=skipped_key_count,
            )
            self.last_round_output = result
            self.history.append(result)
            round_state.setdefault("defense_outputs", {})[self.name] = result
            return participant_params

        scores = krum_scores(vectors, byzantine_count, self.distance_norm)
        selected_count = self._selected_count(participant_count, byzantine_count)
        selected_indices = [index for index, _ in scores[:selected_count]]
        selected_set = set(selected_indices)
        rejected_indices = [
            index for index in range(participant_count) if index not in selected_set
        ]
        selected_updates = [
            client_update
            for index, (_, client_update) in enumerate(participant_items)
            if index in selected_set
        ]
        prototype, aggregated_count, aggregate_skipped, aggregate_ok = mean_values(
            selected_updates
        )
        if aggregate_ok:
            updated_params = replace_all_with_prototype(participant_params, prototype)
            fallback_reason = None
        else:
            updated_params = {
                client_id: client_update
                for index, (client_id, client_update) in enumerate(participant_items)
                if index in selected_set
            }
            fallback_reason = "selected_updates_could_not_be_averaged"

        result = self._build_result(
            participant_count=participant_count,
            selected_indices=selected_indices,
            rejected_indices=rejected_indices,
            estimated_malicious_count=byzantine_count,
            distance_summary=score_summary(scores),
            aggregated_tensor_count=aggregated_count,
            skipped_key_count=skipped_key_count + aggregate_skipped,
            fallback_reason=fallback_reason,
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
        result["score_order"] = [index for index, _ in scores]

        self.last_round_output = result
        self.history.append(result)
        round_state.setdefault("defense_outputs", {})[self.name] = result
        round_state.setdefault("filtered_clients", {})[self.name] = result["rejected_clients"]
        return updated_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "defense_type": "multi_krum",
                "num_rounds": 0,
                "rounds_with_selection": 0,
                "total_selected_clients": 0,
                "total_rejected_clients": 0,
                "estimated_malicious_count": self.estimated_malicious_count,
                "multi_krum_k": self.multi_krum_k,
                "distance_norm": self.distance_norm,
            }
        return {
            "defense_type": "multi_krum",
            "num_rounds": len(self.history),
            "rounds_with_selection": sum(
                1 for item in self.history if not item.get("fallback_reason")
            ),
            "total_selected_clients": int(
                sum(int(item.get("selected_client_count", 0) or 0) for item in self.history)
            ),
            "total_rejected_clients": int(
                sum(int(item.get("rejected_client_count", 0) or 0) for item in self.history)
            ),
            "estimated_malicious_count": self.estimated_malicious_count,
            "multi_krum_k": self.multi_krum_k,
            "distance_norm": self.distance_norm,
            "last_selected_indices": list(self.history[-1].get("selected_indices", [])),
            "last_rejected_indices": list(self.history[-1].get("rejected_indices", [])),
            "last_distance_summary": dict(self.history[-1].get("distance_summary", {})),
        }


register_defense("multi_krum", MultiKrumDefense)
register_defense("multikrum", MultiKrumDefense)
register_defense("multi_krum_defense", MultiKrumDefense)
