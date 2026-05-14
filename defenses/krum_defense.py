"""Krum-style client selection defense for uploaded updates.

The implementation is intentionally lightweight. It flattens compatible tensor
leaves from each client update, scores every client by the distance to its
nearest neighbors, and keeps the lowest-scoring update or updates before the
existing model-specific aggregation code runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class KrumDefense(BaseDefense):
    """Select one or more majority-like client updates with a Krum score."""

    CONFIG_KEYS = (
        "byzantine_count",
        "estimated_malicious_count",
        "multi_krum_k",
        "distance_norm",
    )

    def __init__(
        self,
        name: str = "krum",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.estimated_malicious_count = self._optional_int(
            self._config_get(config, "byzantine_count", None)
        )
        if self.estimated_malicious_count is None:
            self.estimated_malicious_count = self._optional_int(
                self._config_get(config, "estimated_malicious_count", None)
            )
        self.multi_krum_k = self._optional_int(
            self._config_get(config, "multi_krum_k", None)
        )
        self.distance_norm = str(
            self._config_get(config, "distance_norm", "l2")
        ).lower()
        self.last_round_output: Dict[str, Any] = {}
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
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _flatten_tensors(self, value: Any) -> Tuple[List[torch.Tensor], int]:
        if value is None:
            return [], 1

        if torch.is_tensor(value):
            if value.is_sparse:
                value = value.coalesce().values()
            if not torch.is_floating_point(value):
                return [], 1
            if value.numel() == 0:
                return [], 1
            return [value.detach().float().reshape(-1).cpu()], 0

        if isinstance(value, dict):
            vectors: List[torch.Tensor] = []
            skipped = 0
            for key in sorted(value.keys(), key=lambda item: str(item)):
                item_vectors, item_skipped = self._flatten_tensors(value[key])
                vectors.extend(item_vectors)
                skipped += item_skipped
            return vectors, skipped

        if isinstance(value, (list, tuple)):
            vectors = []
            skipped = 0
            for item in value:
                item_vectors, item_skipped = self._flatten_tensors(item)
                vectors.extend(item_vectors)
                skipped += item_skipped
            return vectors, skipped

        return [], 1

    def _client_vector(self, client_update: Any) -> Tuple[Optional[torch.Tensor], int]:
        vectors, skipped = self._flatten_tensors(client_update)
        if not vectors:
            return None, skipped
        try:
            return torch.cat(vectors), skipped
        except Exception:
            return None, skipped + 1

    def _distance(self, left: torch.Tensor, right: torch.Tensor) -> float:
        diff = left - right
        if self.distance_norm == "l1":
            return float(torch.sum(torch.abs(diff)).item())
        if self.distance_norm not in {"l2", "euclidean"}:
            return float(torch.linalg.vector_norm(diff, ord=2).item())
        return float(torch.linalg.vector_norm(diff, ord=2).item())

    def _safe_byzantine_count(
        self,
        participant_count: int,
        round_state: MutableMapping[str, Any],
    ) -> int:
        raw_count = self.estimated_malicious_count
        if raw_count is None:
            raw_count = self._optional_int(round_state.get("malicious_client_count"))
        if raw_count is None:
            malicious_clients = round_state.get("malicious_clients", [])
            raw_count = len(malicious_clients) if isinstance(malicious_clients, list) else 0

        raw_count = max(0, int(raw_count))
        if participant_count <= 2:
            return 0
        return min(raw_count, max(0, (participant_count - 2) // 2))

    def _selected_count(self, participant_count: int, byzantine_count: int) -> int:
        if self.multi_krum_k is None:
            return 1
        max_selectable = max(1, participant_count - byzantine_count)
        return max(1, min(int(self.multi_krum_k), max_selectable, participant_count))

    def _build_result(
        self,
        selected_indices: Optional[List[int]] = None,
        participant_count: int = 0,
        estimated_malicious_count: int = 0,
        fallback_reason: Optional[str] = None,
        warning_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        selected_indices = selected_indices or []
        return {
            "defense_type": "krum",
            "selected_client_count": len(selected_indices),
            "rejected_client_count": max(0, participant_count - len(selected_indices)),
            "selected_indices": selected_indices,
            "estimated_malicious_count": estimated_malicious_count,
            "distance_norm": self.distance_norm,
            "fallback_reason": fallback_reason,
            "warning_summary": warning_summary or {},
        }

    def _score_clients(
        self,
        vectors: Sequence[torch.Tensor],
        byzantine_count: int,
    ) -> List[Tuple[int, float]]:
        participant_count = len(vectors)
        neighbor_count = max(1, min(participant_count - 1, participant_count - byzantine_count - 2))
        scores: List[Tuple[int, float]] = []

        for index, vector in enumerate(vectors):
            distances = [
                self._distance(vector, other)
                for other_index, other in enumerate(vectors)
                if other_index != index
            ]
            distances.sort()
            score = float(sum(distances[:neighbor_count]))
            scores.append((index, score))

        scores.sort(key=lambda item: item[1])
        return scores

    def before_aggregation(
        self,
        participant_params: MutableMapping[str, Any],
        round_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(participant_params, dict) or not participant_params:
            defense_result = self._build_result(
                participant_count=0,
                fallback_reason="empty_or_unsupported_participant_params",
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        participant_items = list(participant_params.items())
        participant_count = len(participant_items)
        byzantine_count = self._safe_byzantine_count(participant_count, round_state)
        client_vectors: List[torch.Tensor] = []
        skipped_key_count = 0

        for _, client_update in participant_items:
            vector, skipped = self._client_vector(client_update)
            skipped_key_count += skipped
            if vector is None:
                defense_result = self._build_result(
                    participant_count=participant_count,
                    estimated_malicious_count=byzantine_count,
                    fallback_reason="unable_to_flatten_all_client_updates",
                    warning_summary={"skipped_key_count": skipped_key_count},
                )
                self.last_round_output = defense_result
                self.history.append(defense_result)
                round_state.setdefault("defense_outputs", {})[self.name] = defense_result
                return participant_params
            client_vectors.append(vector)

        vector_lengths = {int(vector.numel()) for vector in client_vectors}
        if len(vector_lengths) != 1:
            defense_result = self._build_result(
                participant_count=participant_count,
                estimated_malicious_count=byzantine_count,
                fallback_reason="incompatible_flattened_update_shapes",
                warning_summary={
                    "flattened_lengths": sorted(vector_lengths),
                    "skipped_key_count": skipped_key_count,
                },
            )
            self.last_round_output = defense_result
            self.history.append(defense_result)
            round_state.setdefault("defense_outputs", {})[self.name] = defense_result
            return participant_params

        if participant_count <= 2:
            selected_indices = list(range(participant_count))
        else:
            scores = self._score_clients(client_vectors, byzantine_count)
            selected_count = self._selected_count(participant_count, byzantine_count)
            selected_indices = [index for index, _ in scores[:selected_count]]

        selected_set = set(selected_indices)
        updated_participant_params = {
            client_id: client_update
            for index, (client_id, client_update) in enumerate(participant_items)
            if index in selected_set
        }

        rejected_clients = [
            str(client_id)
            for index, (client_id, _) in enumerate(participant_items)
            if index not in selected_set
        ]
        warning_summary = {
            "skipped_key_count": skipped_key_count,
            "distance_shape": int(client_vectors[0].numel()) if client_vectors else 0,
        }
        if self.distance_norm not in {"l1", "l2", "euclidean"}:
            warning_summary["distance_norm_fallback"] = "l2"

        defense_result = self._build_result(
            selected_indices=selected_indices,
            participant_count=participant_count,
            estimated_malicious_count=byzantine_count,
            warning_summary=warning_summary,
        )
        defense_result["selected_clients"] = [
            str(client_id)
            for index, (client_id, _) in enumerate(participant_items)
            if index in selected_set
        ]
        defense_result["rejected_clients"] = rejected_clients

        self.last_round_output = defense_result
        self.history.append(defense_result)
        round_state.setdefault("defense_outputs", {})[self.name] = defense_result
        round_state.setdefault("filtered_clients", {})[self.name] = rejected_clients
        return updated_participant_params

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_round_output)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.history:
            return {
                "defense_type": "krum",
                "num_rounds": 0,
                "rounds_with_selection": 0,
                "total_selected_clients": 0,
                "total_rejected_clients": 0,
                "estimated_malicious_count": self.estimated_malicious_count,
            }

        return {
            "defense_type": "krum",
            "num_rounds": len(self.history),
            "rounds_with_selection": sum(
                1 for item in self.history if not item.get("fallback_reason")
            ),
            "total_selected_clients": int(
                sum(int(item.get("selected_client_count", 0) or 0) for item in self.history)
            ),
            "total_rejected_clients": int(
                sum(int(item.get("rejected_client_count", 0) or 0) for item in self.history)
            ),
            "estimated_malicious_count": self.estimated_malicious_count,
            "multi_krum_k": self.multi_krum_k,
            "distance_norm": self.distance_norm,
        }


register_defense("krum", KrumDefense)
register_defense("krumdefense", KrumDefense)
