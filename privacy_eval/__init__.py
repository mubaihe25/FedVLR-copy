"""Privacy evaluation skeletons for the FedVLR project."""

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.membership_inference_probe import MembershipInferenceProbe
from privacy_eval.noop_metric import NoOpPrivacyMetric
from privacy_eval.preference_inference_probe import PreferenceInferenceProbe
from privacy_eval.registry import (
    get_privacy_metric,
    list_privacy_metrics,
    register_privacy_metric,
)
from privacy_eval.result_schema import (
    ExperimentResult,
    FinalEval,
    RoundMetric,
    build_empty_result,
)

try:
    from privacy_eval.client_update_norm_metric import ClientUpdateNormMetric
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    ClientUpdateNormMetric = None  # type: ignore[assignment]

try:
    from privacy_eval.gradient_leakage_probe import GradientLeakageProbe
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    GradientLeakageProbe = None  # type: ignore[assignment]

try:
    from privacy_eval.update_leakage_risk_probe import UpdateLeakageRiskProbe
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    UpdateLeakageRiskProbe = None  # type: ignore[assignment]

__all__ = [
    "BasePrivacyMetric",
    "ClientUpdateNormMetric",
    "GradientLeakageProbe",
    "MembershipInferenceProbe",
    "PreferenceInferenceProbe",
    "UpdateLeakageRiskProbe",
    "NoOpPrivacyMetric",
    "RoundMetric",
    "FinalEval",
    "ExperimentResult",
    "build_empty_result",
    "register_privacy_metric",
    "get_privacy_metric",
    "list_privacy_metrics",
]
