"""No-op privacy metric placeholder."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


class NoOpPrivacyMetric(BasePrivacyMetric):
    """Privacy metric placeholder that only returns empty summaries."""

    def __init__(self, name: str = "noop", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, config=config)

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        return {}

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        return {}


register_privacy_metric("noop", NoOpPrivacyMetric)
register_privacy_metric("noopprivacymetric", NoOpPrivacyMetric)
