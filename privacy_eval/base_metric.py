"""Minimal privacy metric abstraction."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional


class BasePrivacyMetric:
    """Base class for future privacy evaluation modules.

    Responsibility:
        Define a minimal interface for privacy-related measurement without
        coupling the current training code to a specific metric implementation.

    Expected future inputs:
        - round_state: metadata for the current federated round
        - participant_params: uploaded client information before aggregation
        - aggregation_result: the server-side aggregation output
        - metadata: optional experiment-level context

    Expected future outputs:
        - serializable round-level privacy metrics
        - serializable final privacy metrics
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        """Return round-level privacy metrics."""
        return {}

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return final privacy metrics for the full experiment."""
        return {}
