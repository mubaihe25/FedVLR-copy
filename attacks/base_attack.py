"""Minimal attack abstraction.

This module only defines a future extension contract.
It does not change or hook into the current training pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional


class BaseAttack:
    """Base class for future attack modules.

    Responsibility:
        Provide a minimal interface for attack components that may mutate
        round state, client updates, or aggregation inputs in the future.

    Expected future inputs:
        - round_state: metadata for the current federated round
        - client_update: one client's serialized update or local output
        - participant_params: all uploaded client updates before aggregation
        - aggregation_result: the server-side aggregation output

    Expected future outputs:
        - updated round_state
        - updated client_update
        - updated participant_params
        - attack metrics as a serializable dictionary
    """

    attack_family = "unspecified"
    attack_category = "unspecified"
    attack_strategy = "unspecified"
    attack_display_category = "未分类攻击"
    mutates_participant_params = False
    is_read_only = True

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}

    def semantic_metadata(self) -> Dict[str, Any]:
        """Return stable attack taxonomy metadata for result/API/front-end use."""
        return {
            "attack_family": self.attack_family,
            "attack_category": self.attack_category,
            "attack_strategy": self.attack_strategy,
            "attack_display_category": self.attack_display_category,
            "mutates_participant_params": self.mutates_participant_params,
            "is_read_only": self.is_read_only,
        }

    def before_round(
        self, round_state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Prepare attack state before a federated round starts."""
        return round_state

    def after_local_train(
        self,
        client_id: Any,
        client_update: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Optionally mutate a client's output after local training."""
        return client_update

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Optionally mutate participant uploads before aggregation."""
        return participant_params

    def after_round(
        self,
        aggregation_result: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Optionally update round-level state after aggregation."""
        return aggregation_result

    def collect_metrics(self) -> Dict[str, Any]:
        """Return serializable attack-side metrics."""
        return {}

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return serializable experiment-level attack summaries."""
        return {}
