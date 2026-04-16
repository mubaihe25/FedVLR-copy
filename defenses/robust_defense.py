"""Unified robust-defense wrapper for existing FedVLR defenses.

This module is a minimal engineering wrapper, not a new robust aggregation
algorithm. It reuses existing defenses and exposes one stable entry point:

- ``norm_clip`` as clipping-style robust preprocessing
- ``update_filter`` as filtering-style robust preprocessing
- ``trimmed_mean`` as the current trimmed-mean-like robust aggregation path

The wrapper runs in the existing ``before_aggregation`` hook and returns a
participant_params object that remains compatible with the current trainer.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional

from defenses.base_defense import BaseDefense
from defenses.norm_clip_defense import NormClipDefense
from defenses.registry import register_defense
from defenses.trimmed_mean_defense import TrimmedMeanDefense
from defenses.update_filter_defense import UpdateFilterDefense


class RobustDefense(BaseDefense):
    """Coordinate existing robust defenses through one configurable entry."""

    MODE_STEPS = {
        "clip": ["norm_clip"],
        "filter": ["update_filter"],
        "trimmed_mean": ["trimmed_mean"],
        "clip_then_trimmed_mean": ["norm_clip", "trimmed_mean"],
        "filter_then_trimmed_mean": ["update_filter", "trimmed_mean"],
        "clip_then_filter_then_trimmed_mean": [
            "norm_clip",
            "update_filter",
            "trimmed_mean",
        ],
    }

    def __init__(
        self,
        name: str = "robust_defense",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        cfg = dict(config or {})
        cfg.update({key: value for key, value in kwargs.items() if value is not None})
        super().__init__(name=name, config=cfg)
        self.robust_defense_mode = str(
            cfg.get("robust_defense_mode", "trimmed_mean")
        ).lower()
        self.robust_clip_norm = float(
            cfg.get("robust_clip_norm", cfg.get("defense_clip_norm", 5.0))
        )
        self.robust_filter_rule = str(
            cfg.get(
                "robust_filter_rule",
                cfg.get("filter_rule", "update_norm > mean + filter_std_factor * std"),
            )
        )
        self.robust_filter_std_factor = float(
            cfg.get("robust_filter_std_factor", cfg.get("filter_std_factor", 2.0))
        )
        self.robust_max_filtered_ratio = float(
            cfg.get("robust_max_filtered_ratio", cfg.get("max_filtered_ratio", 0.5))
        )
        self.robust_trim_ratio = float(
            cfg.get("robust_trim_ratio", cfg.get("trim_ratio", 0.2))
        )
        self.robust_min_clients_for_trim = int(
            cfg.get("robust_min_clients_for_trim", cfg.get("min_clients_for_trim", 5))
        )
        self.robust_trim_rule = str(
            cfg.get("robust_trim_rule", cfg.get("trim_rule", "coordinate_trimmed_mean"))
        )

        self._clip_defense = NormClipDefense(
            config={"defense_clip_norm": self.robust_clip_norm}
        )
        self._filter_defense = UpdateFilterDefense(
            config={
                "filter_rule": self.robust_filter_rule,
                "filter_std_factor": self.robust_filter_std_factor,
                "max_filtered_ratio": self.robust_max_filtered_ratio,
            }
        )
        self._trimmed_mean_defense = TrimmedMeanDefense(
            config={
                "trim_ratio": self.robust_trim_ratio,
                "min_clients_for_trim": self.robust_min_clients_for_trim,
                "trim_rule": self.robust_trim_rule,
            }
        )
        self._step_defenses = {
            "norm_clip": self._clip_defense,
            "update_filter": self._filter_defense,
            "trimmed_mean": self._trimmed_mean_defense,
        }
        self.last_round_output: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def _steps_for_mode(self) -> List[str]:
        return list(self.MODE_STEPS.get(self.robust_defense_mode, ["trimmed_mean"]))

    def _participant_count(self, participant_params: Any) -> int:
        return len(participant_params) if isinstance(participant_params, dict) else 0

    def _base_result(
        self,
        participant_count_before: int,
        participant_count_after: int,
        applied_steps: List[str],
        step_metrics: Dict[str, Any],
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        clipped_count = 0
        filtered_count = 0
        trimmed_mean_applied = False
        retained_equivalent = None

        clip_metrics = step_metrics.get("norm_clip", {})
        if isinstance(clip_metrics, dict):
            clipped_count = int(clip_metrics.get("clipped_client_count", 0) or 0)

        filter_metrics = step_metrics.get("update_filter", {})
        if isinstance(filter_metrics, dict):
            filtered_count = int(filter_metrics.get("filtered_client_count", 0) or 0)

        trim_metrics = step_metrics.get("trimmed_mean", {})
        if isinstance(trim_metrics, dict):
            trimmed_mean_applied = bool(trim_metrics.get("trimmed_mean_applied", False))
            retained_equivalent = trim_metrics.get("retained_client_count_equivalent")

        return {
            "defense_family": "robust_defense",
            "defense_category": "robust_defense",
            "defense_strategy": "unified_robust_defense",
            "defense_display_category": "鲁棒防御",
            "robust_defense_mode": self.robust_defense_mode,
            "applied_steps": applied_steps,
            "participant_count_before": participant_count_before,
            "participant_count_after": participant_count_after,
            "clipped_client_count": clipped_count,
            "filtered_client_count": filtered_count,
            "trimmed_mean_applied": trimmed_mean_applied,
            "retained_client_count_equivalent": retained_equivalent,
            "step_metrics": step_metrics,
            "fallback_reason": fallback_reason,
        }

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        participant_count_before = self._participant_count(participant_params)
        if not isinstance(participant_params, dict) or not participant_params:
            defense_result = self._base_result(
                participant_count_before=participant_count_before,
                participant_count_after=participant_count_before,
                applied_steps=[],
                step_metrics={},
                fallback_reason="empty_or_unsupported_participant_params",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        steps = self._steps_for_mode()
        current_params: MutableMapping[str, Any] = participant_params
        step_metrics: Dict[str, Any] = {}
        applied_steps: List[str] = []

        for step_name in steps:
            step_defense = self._step_defenses.get(step_name)
            if step_defense is None:
                continue
            current_params = step_defense.before_aggregation(current_params, round_state)
            step_metrics[step_name] = step_defense.collect_metrics()
            applied_steps.append(step_name)

        defense_result = self._base_result(
            participant_count_before=participant_count_before,
            participant_count_after=self._participant_count(current_params),
            applied_steps=applied_steps,
            step_metrics=step_metrics,
            fallback_reason=None if applied_steps else "unsupported_robust_defense_mode",
        )
        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("robust_defense_applied", {})[self.name] = bool(
            applied_steps
        )
        return current_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "num_rounds": 0,
                "rounds_with_robust_defense": 0,
                "robust_defense_mode": self.robust_defense_mode,
                "mode_usage_summary": {},
                "total_clipped_clients": 0,
                "total_filtered_clients": 0,
                "rounds_with_trimmed_mean": 0,
                "avg_retained_client_count_equivalent": None,
            }

        retained_values = [
            int(item["retained_client_count_equivalent"])
            for item in self.history
            if item.get("retained_client_count_equivalent") is not None
        ]
        mode_usage_summary = {
            self.robust_defense_mode: sum(
                1 for item in self.history if item.get("applied_steps")
            )
        }
        return {
            "num_rounds": len(self.history),
            "rounds_with_robust_defense": sum(
                1 for item in self.history if item.get("applied_steps")
            ),
            "robust_defense_mode": self.robust_defense_mode,
            "mode_usage_summary": mode_usage_summary,
            "total_clipped_clients": int(
                sum(int(item.get("clipped_client_count", 0) or 0) for item in self.history)
            ),
            "total_filtered_clients": int(
                sum(int(item.get("filtered_client_count", 0) or 0) for item in self.history)
            ),
            "rounds_with_trimmed_mean": sum(
                1 for item in self.history if item.get("trimmed_mean_applied")
            ),
            "avg_retained_client_count_equivalent": (
                float(sum(retained_values) / len(retained_values))
                if retained_values
                else None
            ),
        }


register_defense("robust_defense", RobustDefense)
register_defense("robust", RobustDefense)
register_defense("robust_aggregation_defense", RobustDefense)
