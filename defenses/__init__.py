"""Defense extension skeletons for the FedVLR project."""

from defenses.base_defense import BaseDefense
from defenses.bulyan_defense import BulyanDefense
from defenses.client_update_anomaly_detector import ClientUpdateAnomalyDetector
from defenses.dp_noise_defense import DPNoiseDefense
from defenses.krum_defense import KrumDefense
from defenses.median_defense import MedianDefense
from defenses.multi_krum_defense import MultiKrumDefense
from defenses.norm_clip_defense import NormClipDefense
from defenses.noop_defense import NoOpDefense
from defenses.robust_defense import RobustDefense
from defenses.secure_aggregation_sim import SecureAggregationSimDefense
from defenses.trimmed_mean_defense import TrimmedMeanDefense
from defenses.update_filter_defense import UpdateFilterDefense
from defenses.registry import get_defense, list_defenses, register_defense

__all__ = [
    "BaseDefense",
    "BulyanDefense",
    "ClientUpdateAnomalyDetector",
    "DPNoiseDefense",
    "KrumDefense",
    "MedianDefense",
    "MultiKrumDefense",
    "NormClipDefense",
    "RobustDefense",
    "SecureAggregationSimDefense",
    "TrimmedMeanDefense",
    "UpdateFilterDefense",
    "NoOpDefense",
    "register_defense",
    "get_defense",
    "list_defenses",
]
