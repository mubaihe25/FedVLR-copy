"""Defense extension skeletons for the FedVLR project."""

from defenses.base_defense import BaseDefense
from defenses.client_update_anomaly_detector import ClientUpdateAnomalyDetector
from defenses.norm_clip_defense import NormClipDefense
from defenses.noop_defense import NoOpDefense
from defenses.update_filter_defense import UpdateFilterDefense
from defenses.registry import get_defense, list_defenses, register_defense

__all__ = [
    "BaseDefense",
    "ClientUpdateAnomalyDetector",
    "NormClipDefense",
    "UpdateFilterDefense",
    "NoOpDefense",
    "register_defense",
    "get_defense",
    "list_defenses",
]
