"""Minimal registry for privacy metric skeletons."""

from __future__ import annotations

from typing import Dict, List, Type

from privacy_eval.base_metric import BasePrivacyMetric

_PRIVACY_METRIC_REGISTRY: Dict[str, Type[BasePrivacyMetric]] = {}


def register_privacy_metric(
    name: str, metric_cls: Type[BasePrivacyMetric]
) -> None:
    """Register a privacy metric class by name."""
    _PRIVACY_METRIC_REGISTRY[name.lower()] = metric_cls


def get_privacy_metric(name: str) -> Type[BasePrivacyMetric]:
    """Return a registered privacy metric class."""
    key = name.lower()
    if key not in _PRIVACY_METRIC_REGISTRY:
        raise KeyError("Unknown privacy metric: {}".format(name))
    return _PRIVACY_METRIC_REGISTRY[key]


def list_privacy_metrics() -> List[str]:
    """List all registered privacy metric names."""
    return sorted(_PRIVACY_METRIC_REGISTRY.keys())
