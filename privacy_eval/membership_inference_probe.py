"""Score-based membership inference privacy probe.

This probe is intentionally lightweight and independent from the FedVLR training
loop. It estimates whether member samples receive higher model scores than
non-member samples. It is not a full industrial black-box membership inference
attack.
"""

from __future__ import annotations

import json
import math
from numbers import Number
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


class MembershipInferenceProbe(BasePrivacyMetric):
    """Evaluate score separation between member and non-member samples."""

    MEMBER_SCORE_KEYS = (
        "member_scores",
        "membership_member_scores",
        "mia_member_scores",
    )
    NON_MEMBER_SCORE_KEYS = (
        "non_member_scores",
        "membership_non_member_scores",
        "mia_non_member_scores",
    )

    def __init__(
        self,
        name: str = "membership_inference_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.score_direction = str(
            self._config_get(config, "score_direction", "higher_is_member")
        ).lower()
        self.threshold = self._optional_float(
            self._config_get(config, "membership_threshold", None)
        )
        if self.threshold is None:
            self.threshold = self._optional_float(self._config_get(config, "threshold", None))
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _config_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                try:
                    value = getter(key)
                except Exception:
                    return default
                return default if value is None else value
        return getattr(config, key, default)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _as_float_list(self, value: Any) -> List[float]:
        if value is None:
            return []
        if isinstance(value, bool):
            return []
        if isinstance(value, Number):
            number = float(value)
            return [number] if math.isfinite(number) else []
        if hasattr(value, "detach") and callable(getattr(value, "detach")):
            try:
                value = value.detach().cpu().reshape(-1).tolist()
            except Exception:
                return []
        elif hasattr(value, "tolist") and callable(getattr(value, "tolist")):
            try:
                value = value.tolist()
            except Exception:
                return []
        if isinstance(value, dict):
            scores: List[float] = []
            for item in value.values():
                scores.extend(self._as_float_list(item))
            return scores
        if isinstance(value, (list, tuple, set)):
            scores = []
            for item in value:
                scores.extend(self._as_float_list(item))
            return scores
        return []

    def _read_score_pair(self, source: Any) -> Tuple[List[float], List[float]]:
        member_scores: List[float] = []
        non_member_scores: List[float] = []

        for key in self.MEMBER_SCORE_KEYS:
            member_scores = self._as_float_list(self._config_get(source, key, None))
            if member_scores:
                break

        for key in self.NON_MEMBER_SCORE_KEYS:
            non_member_scores = self._as_float_list(self._config_get(source, key, None))
            if non_member_scores:
                break

        return member_scores, non_member_scores

    def _candidate_sources(
        self,
        round_state: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Iterable[Any]:
        yield self.config
        for source in (round_state, aggregation_result):
            if not isinstance(source, dict):
                continue
            yield source
            for nested_key in (
                "membership_inference",
                "membership_inference_probe",
                "privacy_probe_inputs",
                "privacy_metric_inputs",
            ):
                nested = source.get(nested_key)
                if isinstance(nested, dict):
                    yield nested
                    for probe_key in (
                        "membership_inference",
                        "membership_inference_probe",
                    ):
                        probe_nested = nested.get(probe_key)
                        if isinstance(probe_nested, dict):
                            yield probe_nested

    def _extract_scores(
        self,
        round_state: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Tuple[List[float], List[float]]:
        for source in self._candidate_sources(round_state, aggregation_result):
            member_scores, non_member_scores = self._read_score_pair(source)
            if member_scores and non_member_scores:
                return member_scores, non_member_scores
        return [], []

    def _attack_scores(self, scores: List[float]) -> List[float]:
        if self.score_direction in {"lower_is_member", "loss_lower_is_member"}:
            return [-score for score in scores]
        return list(scores)

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return float(sum(values) / len(values)) if values else None

    @staticmethod
    def _std(values: List[float], mean_value: Optional[float]) -> Optional[float]:
        if not values or mean_value is None:
            return None
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return float(math.sqrt(variance))

    @staticmethod
    def _manual_auc(member_scores: List[float], non_member_scores: List[float]) -> Optional[float]:
        if not member_scores or not non_member_scores:
            return None
        wins = 0.0
        for member_score in member_scores:
            for non_member_score in non_member_scores:
                if member_score > non_member_score:
                    wins += 1.0
                elif member_score == non_member_score:
                    wins += 0.5
        return float(wins / (len(member_scores) * len(non_member_scores)))

    def _accuracy_for_threshold(
        self,
        member_scores: List[float],
        non_member_scores: List[float],
        threshold: float,
    ) -> float:
        correct_members = sum(1 for score in member_scores if score >= threshold)
        correct_non_members = sum(1 for score in non_member_scores if score < threshold)
        total = len(member_scores) + len(non_member_scores)
        return float((correct_members + correct_non_members) / total) if total else 0.0

    def _best_threshold_accuracy(
        self,
        member_scores: List[float],
        non_member_scores: List[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        if not member_scores or not non_member_scores:
            return None, None

        if self.threshold is not None:
            return self.threshold, self._accuracy_for_threshold(
                member_scores, non_member_scores, self.threshold
            )

        unique_scores = sorted(set(member_scores + non_member_scores))
        if not unique_scores:
            return None, None

        candidates: List[float] = [unique_scores[0] - 1e-12, unique_scores[-1] + 1e-12]
        candidates.extend(unique_scores)
        candidates.extend(
            (left + right) / 2.0
            for left, right in zip(unique_scores[:-1], unique_scores[1:])
        )

        best_threshold = candidates[0]
        best_accuracy = -1.0
        for threshold in candidates:
            accuracy = self._accuracy_for_threshold(member_scores, non_member_scores, threshold)
            if accuracy > best_accuracy:
                best_threshold = threshold
                best_accuracy = accuracy
        return float(best_threshold), float(best_accuracy)

    def _risk_level(
        self,
        attack_accuracy: Optional[float],
        attack_auc: Optional[float],
        member_scores: List[float],
        non_member_scores: List[float],
    ) -> str:
        member_mean = self._mean(member_scores)
        non_member_mean = self._mean(non_member_scores)
        member_std = self._std(member_scores, member_mean)
        non_member_std = self._std(non_member_scores, non_member_mean)
        if member_mean is None or non_member_mean is None:
            return "low"

        pooled_std = math.sqrt(((member_std or 0.0) ** 2 + (non_member_std or 0.0) ** 2) / 2.0)
        standardized_gap = abs(member_mean - non_member_mean) / max(pooled_std, 1e-12)
        gap_score = min(1.0, standardized_gap / 2.0)
        risk_score = max(attack_accuracy or 0.0, attack_auc or 0.0, gap_score)

        if risk_score >= 0.75:
            return "high"
        if risk_score >= 0.60:
            return "medium"
        return "low"

    def evaluate_scores(
        self,
        member_scores: List[float],
        non_member_scores: List[float],
    ) -> Dict[str, Any]:
        attack_member_scores = self._attack_scores(member_scores)
        attack_non_member_scores = self._attack_scores(non_member_scores)
        threshold, attack_accuracy = self._best_threshold_accuracy(
            attack_member_scores, attack_non_member_scores
        )
        attack_auc = self._manual_auc(attack_member_scores, attack_non_member_scores)
        member_mean = self._mean(member_scores)
        non_member_mean = self._mean(non_member_scores)
        score_gap = (
            float(member_mean - non_member_mean)
            if member_mean is not None and non_member_mean is not None
            else None
        )
        sample_count = len(member_scores) + len(non_member_scores)
        risk_level = self._risk_level(
            attack_accuracy, attack_auc, member_scores, non_member_scores
        )

        result = {
            "probe_type": "membership_inference",
            "sample_count": sample_count,
            "member_count": len(member_scores),
            "non_member_count": len(non_member_scores),
            "attack_accuracy": attack_accuracy,
            "attack_auc": attack_auc,
            "member_score_mean": member_mean,
            "non_member_score_mean": non_member_mean,
            "member_score_gap": score_gap,
            "decision_threshold": threshold,
            "score_direction": self.score_direction,
            "risk_level": risk_level,
            "note": "score-based membership inference probe, not a full black-box MIA",
        }
        result["membership_inference"] = {
            key: result[key]
            for key in (
                "sample_count",
                "member_count",
                "non_member_count",
                "attack_accuracy",
                "attack_auc",
                "member_score_gap",
                "risk_level",
            )
        }
        result["privacy_attack_summaries"] = {
            "membership_inference": dict(result["membership_inference"])
        }
        result["privacy_risk_summary"] = {
            "membership_inference": {
                "risk_level": risk_level,
                "attack_accuracy": attack_accuracy,
                "attack_auc": attack_auc,
            }
        }
        return result

    def _empty_result(self, fallback_reason: str) -> Dict[str, Any]:
        result = self.evaluate_scores([], [])
        result["fallback_reason"] = fallback_reason
        return result

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        del participant_params
        member_scores, non_member_scores = self._extract_scores(round_state, aggregation_result)
        if not member_scores or not non_member_scores:
            round_result = self._empty_result("missing_member_or_non_member_scores")
        else:
            round_result = self.evaluate_scores(member_scores, non_member_scores)
        self.history.append(round_result)
        return round_result

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            summary = self._empty_result("no_probe_history")
            summary["num_rounds"] = 0
            return summary

        latest = dict(self.history[-1])
        latest["num_rounds"] = len(self.history)
        return latest


def run_synthetic_smoke() -> Dict[str, Any]:
    """Run a tiny synthetic score separation check."""
    probe = MembershipInferenceProbe(
        config={
            "member_scores": [0.58, 0.62, 0.69, 0.72, 0.80],
            "non_member_scores": [0.35, 0.41, 0.48, 0.51, 0.56],
        }
    )
    result = probe.evaluate_round({}, {}, {})
    assert result["attack_accuracy"] is not None
    assert result["member_score_gap"] is not None
    assert result["member_score_gap"] > 0
    assert result["risk_level"] in {"low", "medium", "high"}
    return result


if __name__ == "__main__":
    print(json.dumps(run_synthetic_smoke(), indent=2, sort_keys=True))


register_privacy_metric("membership_inference_probe", MembershipInferenceProbe)
register_privacy_metric("membership_inference", MembershipInferenceProbe)
register_privacy_metric("membershipinferenceprobe", MembershipInferenceProbe)
