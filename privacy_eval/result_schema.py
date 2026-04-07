"""Serializable result schema draft for future API integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoundMetric:
    """Round-level metrics emitted by future training, attack, and defense code."""

    round_index: int
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


def build_empty_result(experiment_id: str, model: str, dataset: str) -> ExperimentResult:
    """Build an empty result payload with required identity fields."""
    return ExperimentResult(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
    )
