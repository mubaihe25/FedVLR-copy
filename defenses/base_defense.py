"""Minimal defense abstraction.

This module only defines a future extension contract.
It does not change or hook into the current training pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional


class BaseDefense:
    """Base class for future defense modules.

    Responsibility:
        Provide a minimal interface for filtering, reweighting, or transforming
        client updates and aggregation outputs.

    Expected future inputs:
        - round_state: metadata for the current federated round
        - client_update: one client's serialized update or local output
        - participant_params: all uploaded client updates before aggregation
        - aggregation_result: the server-side aggregation output

    Expected future outputs:
        - cleaned client_update
        - filtered or reweighted participant_params
        - defended aggregation_result
        - defense metrics as a serializable dictionary
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}

    def before_round(
        self, round_state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Prepare defense state before a federated round starts."""
        return round_state

    def inspect_client_update(
        self,
        client_id: Any,
        client_update: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Inspect or modify a client's update."""
        return client_update

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Filter or transform uploads before server aggregation."""
        return participant_params

    def after_aggregation(
        self,
        aggregation_result: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Adjust aggregation outputs after server aggregation."""
        return aggregation_result

    def collect_metrics(self) -> Dict[str, Any]:
        """Return serializable defense-side metrics."""
        return {}
