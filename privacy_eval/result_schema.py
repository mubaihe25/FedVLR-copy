"""Serializable result schema draft for future API integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoundMetric:
    """Round-level metrics emitted by future training, attack, and defense code."""

    round_index: int
    round_id: int = 0
    participant_clients: List[str] = field(default_factory=list)
    num_participants: int = 0
    avg_train_loss: Optional[float] = None
    valid_score: Optional[float] = None
    test_score: Optional[float] = None
    hooks_enabled: bool = False
    malicious_clients: List[str] = field(default_factory=list)
    participant_count: int = 0
    malicious_client_count: int = 0
    train_loss: Optional[float] = None
    attack_success_rate: Optional[float] = None
    privacy_risk_score: Optional[float] = None
    robustness_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalEval:
    """Final evaluation block for recommendation and security outputs."""

    recall20: Optional[float] = None
    ndcg20: Optional[float] = None
    loss: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Top-level result payload suggested for future API and frontend use."""

    experiment_id: str
    model: str
    dataset: str
    active_attacks: List[str] = field(default_factory=list)
    active_defenses: List[str] = field(default_factory=list)
    active_privacy_metrics: List[str] = field(default_factory=list)
    experiment_mode: Optional[str] = None
    scenario_tags: List[str] = field(default_factory=list)
    attack_type: Optional[str] = None
    defense_type: Optional[str] = None
    malicious_clients: List[str] = field(default_factory=list)
    round_metrics: List[RoundMetric] = field(default_factory=list)
    attack_success_rate: Optional[float] = None
    privacy_risk_score: Optional[float] = None
    robustness_score: Optional[float] = None
    final_eval: FinalEval = field(default_factory=FinalEval)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert nested dataclasses into a plain serializable dictionary."""
        return asdict(self)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert the full experiment result into a lightweight summary."""
        return build_experiment_summary(self)


def _sum_metric_counts(metric_outputs: Dict[str, Any], field_name: str) -> int:
    total = 0
    for metric_output in metric_outputs.values():
        if not isinstance(metric_output, dict):
            continue
        value = metric_output.get(field_name, 0)
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _build_pipeline_summary(pipeline_info: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pipeline_info, dict):
        return {}
    return {
        "active_attacks": list(pipeline_info.get("active_attacks", [])),
        "active_defenses": list(pipeline_info.get("active_defenses", [])),
        "active_privacy_metrics": list(
            pipeline_info.get("active_privacy_metrics", [])
        ),
        "attack_taxonomy": pipeline_info.get("attack_taxonomy", {}),
        "experiment_mode": pipeline_info.get("experiment_mode"),
        "scenario_tags": list(pipeline_info.get("scenario_tags", [])),
        "malicious_client_count": int(pipeline_info.get("malicious_client_count", 0)),
    }


def build_round_summary(round_metric: RoundMetric) -> Dict[str, Any]:
    """Build a lightweight summary for a single round."""
    extra = round_metric.extra or {}
    attack_metrics = extra.get("attack_metrics", {})
    defense_metrics = extra.get("defense_metrics", {})
    pipeline_info = extra.get("pipeline_info", {})

    return {
        "round_id": round_metric.round_id or round_metric.round_index,
        "num_participants": round_metric.num_participants or round_metric.participant_count,
        "avg_train_loss": round_metric.avg_train_loss
        if round_metric.avg_train_loss is not None
        else round_metric.train_loss,
        "valid_score": round_metric.valid_score,
        "test_score": round_metric.test_score,
        "malicious_client_count": round_metric.malicious_client_count,
        "attacked_client_count": _sum_metric_counts(
            attack_metrics, "attacked_client_count"
        ),
        "clipped_client_count": _sum_metric_counts(
            defense_metrics, "clipped_client_count"
        ),
        "pipeline_info": _build_pipeline_summary(pipeline_info),
    }


def build_experiment_summary(result: ExperimentResult) -> Dict[str, Any]:
    """Build a lightweight experiment summary for API/frontend consumption."""
    metadata = result.metadata or {}
    return {
        "experiment_id": result.experiment_id,
        "model": result.model,
        "dataset": result.dataset,
        "experiment_mode": result.experiment_mode,
        "scenario_tags": list(result.scenario_tags),
        "output_run_id": metadata.get("output_run_id"),
        "active_attacks": list(result.active_attacks),
        "active_defenses": list(result.active_defenses),
        "active_privacy_metrics": list(result.active_privacy_metrics),
        "attack_taxonomy": metadata.get("attack_taxonomy", {}),
        "malicious_client_summary": metadata.get("malicious_client_summary", {}),
        "final_eval": asdict(result.final_eval),
        "round_summaries": [
            build_round_summary(round_metric) for round_metric in result.round_metrics
        ],
    }


def build_empty_result(experiment_id: str, model: str, dataset: str) -> ExperimentResult:
    """Build an empty result payload with required identity fields."""
    return ExperimentResult(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
    )
