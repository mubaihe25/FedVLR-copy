"""Small helpers for robust aggregation defenses.

The helpers in this file intentionally stay tensor-structure agnostic. They
operate on the ``participant_params`` objects already passed through the
existing before_aggregation hook and never call model-specific aggregation code.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch


def config_get(config: Any, key: str, default: Any = None) -> Any:
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


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return value


def flatten_tensors(value: Any) -> Tuple[List[torch.Tensor], int]:
    if value is None:
        return [], 1

    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.is_sparse:
            tensor = tensor.coalesce().values()
        if not torch.is_floating_point(tensor) or tensor.numel() == 0:
            return [], 1
        return [tensor.float().reshape(-1).cpu()], 0

    if isinstance(value, dict):
        vectors: List[torch.Tensor] = []
        skipped = 0
        for key in sorted(value.keys(), key=lambda item: str(item)):
            item_vectors, item_skipped = flatten_tensors(value[key])
            vectors.extend(item_vectors)
            skipped += item_skipped
        return vectors, skipped

    if isinstance(value, (list, tuple)):
        vectors = []
        skipped = 0
        for item in value:
            item_vectors, item_skipped = flatten_tensors(item)
            vectors.extend(item_vectors)
            skipped += item_skipped
        return vectors, skipped

    return [], 1


def client_vector(client_update: Any) -> Tuple[Optional[torch.Tensor], int]:
    vectors, skipped = flatten_tensors(client_update)
    if not vectors:
        return None, skipped
    try:
        return torch.cat(vectors), skipped
    except Exception:
        return None, skipped + 1


def participant_vectors(
    participant_params: MutableMapping[str, Any],
) -> Tuple[List[Tuple[Any, Any]], List[torch.Tensor], int, Optional[str]]:
    participant_items = list(participant_params.items())
    vectors: List[torch.Tensor] = []
    skipped_key_count = 0
    for _, client_update in participant_items:
        vector, skipped = client_vector(client_update)
        skipped_key_count += skipped
        if vector is None:
            return participant_items, [], skipped_key_count, "unable_to_flatten_all_client_updates"
        vectors.append(vector)
    vector_lengths = {int(vector.numel()) for vector in vectors}
    if len(vector_lengths) > 1:
        return participant_items, vectors, skipped_key_count, "incompatible_flattened_update_shapes"
    return participant_items, vectors, skipped_key_count, None


def distance(left: torch.Tensor, right: torch.Tensor, distance_norm: str = "l2") -> float:
    diff = left - right
    if distance_norm == "l1":
        return float(torch.sum(torch.abs(diff)).item())
    return float(torch.linalg.vector_norm(diff, ord=2).item())


def safe_byzantine_count(
    configured_count: Optional[int],
    participant_count: int,
    round_state: MutableMapping[str, Any],
) -> int:
    raw_count = configured_count
    if raw_count is None:
        raw_count = optional_int(round_state.get("malicious_client_count"))
    if raw_count is None:
        malicious_clients = round_state.get("malicious_clients", [])
        raw_count = len(malicious_clients) if isinstance(malicious_clients, list) else 0
    raw_count = max(0, int(raw_count))
    if participant_count <= 2:
        return 0
    return min(raw_count, max(0, (participant_count - 2) // 2))


def krum_scores(
    vectors: Sequence[torch.Tensor],
    byzantine_count: int,
    distance_norm: str = "l2",
) -> List[Tuple[int, float]]:
    participant_count = len(vectors)
    if participant_count <= 1:
        return [(index, 0.0) for index in range(participant_count)]
    neighbor_count = max(
        1,
        min(participant_count - 1, participant_count - byzantine_count - 2),
    )
    scores: List[Tuple[int, float]] = []
    for index, vector in enumerate(vectors):
        distances = [
            distance(vector, other, distance_norm=distance_norm)
            for other_index, other in enumerate(vectors)
            if other_index != index
        ]
        distances.sort()
        scores.append((index, float(sum(distances[:neighbor_count]))))
    scores.sort(key=lambda item: item[1])
    return scores


def score_summary(scores: Sequence[Tuple[int, float]]) -> Dict[str, Any]:
    values = [float(score) for _, score in scores]
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def mean_values(values: Sequence[Any]) -> Tuple[Any, int, int, bool]:
    values = [value for value in values if value is not None]
    if not values:
        return None, 0, 1, False

    if all(torch.is_tensor(value) for value in values):
        first = values[0]
        if (
            any(value.is_sparse for value in values)
            or not torch.is_floating_point(first)
            or any(value.shape != first.shape for value in values)
        ):
            return clone_value(first), 0, 1, False
        stacked = torch.stack(
            [value.detach().to(device=first.device, dtype=first.dtype) for value in values],
            dim=0,
        )
        return stacked.mean(dim=0), 1, 0, True

    if all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
        return float(sum(float(value) for value in values) / len(values)), 1, 0, True

    if all(isinstance(value, dict) for value in values):
        first = values[0]
        output: Dict[Any, Any] = {}
        touched = 0
        skipped = 0
        ok = False
        for key in first.keys():
            if not all(key in value for value in values):
                output[key] = clone_value(first[key])
                skipped += 1
                continue
            item, item_touched, item_skipped, item_ok = mean_values(
                [value[key] for value in values]
            )
            output[key] = item if item_ok else clone_value(first[key])
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return output, touched, skipped, ok

    if all(isinstance(value, list) for value in values):
        length = len(values[0])
        if any(len(value) != length for value in values):
            return clone_value(values[0]), 0, 1, False
        output_list: List[Any] = []
        touched = 0
        skipped = 0
        ok = False
        for index in range(length):
            item, item_touched, item_skipped, item_ok = mean_values(
                [value[index] for value in values]
            )
            output_list.append(item if item_ok else clone_value(values[0][index]))
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return output_list, touched, skipped, ok

    if all(isinstance(value, tuple) for value in values):
        length = len(values[0])
        if any(len(value) != length for value in values):
            return clone_value(values[0]), 0, 1, False
        output_items: List[Any] = []
        touched = 0
        skipped = 0
        ok = False
        for index in range(length):
            item, item_touched, item_skipped, item_ok = mean_values(
                [value[index] for value in values]
            )
            output_items.append(item if item_ok else clone_value(values[0][index]))
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return tuple(output_items), touched, skipped, ok

    return clone_value(values[0]), 0, 1, False


def trimmed_mean_values(values: Sequence[Any], trim_count: int) -> Tuple[Any, int, int, bool]:
    values = [value for value in values if value is not None]
    if not values:
        return None, 0, 1, False
    trim_count = max(0, int(trim_count))

    if all(torch.is_tensor(value) for value in values):
        first = values[0]
        if (
            any(value.is_sparse for value in values)
            or not torch.is_floating_point(first)
            or any(value.shape != first.shape for value in values)
            or len(values) - 2 * trim_count <= 0
        ):
            return clone_value(first), 0, 1, False
        stacked = torch.stack(
            [value.detach().to(device=first.device, dtype=first.dtype) for value in values],
            dim=0,
        )
        if trim_count > 0:
            stacked = torch.sort(stacked, dim=0).values[trim_count : len(values) - trim_count]
        return stacked.mean(dim=0), 1, 0, True

    if all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
        if len(values) - 2 * trim_count <= 0:
            return clone_value(values[0]), 0, 1, False
        sorted_values = sorted(float(value) for value in values)
        if trim_count > 0:
            sorted_values = sorted_values[trim_count : len(sorted_values) - trim_count]
        return float(sum(sorted_values) / len(sorted_values)), 1, 0, True

    if all(isinstance(value, dict) for value in values):
        first = values[0]
        output: Dict[Any, Any] = {}
        touched = 0
        skipped = 0
        ok = False
        for key in first.keys():
            if not all(key in value for value in values):
                output[key] = clone_value(first[key])
                skipped += 1
                continue
            item, item_touched, item_skipped, item_ok = trimmed_mean_values(
                [value[key] for value in values],
                trim_count,
            )
            output[key] = item if item_ok else clone_value(first[key])
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return output, touched, skipped, ok

    if all(isinstance(value, list) for value in values):
        length = len(values[0])
        if any(len(value) != length for value in values):
            return clone_value(values[0]), 0, 1, False
        output_list: List[Any] = []
        touched = 0
        skipped = 0
        ok = False
        for index in range(length):
            item, item_touched, item_skipped, item_ok = trimmed_mean_values(
                [value[index] for value in values],
                trim_count,
            )
            output_list.append(item if item_ok else clone_value(values[0][index]))
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return output_list, touched, skipped, ok

    if all(isinstance(value, tuple) for value in values):
        length = len(values[0])
        if any(len(value) != length for value in values):
            return clone_value(values[0]), 0, 1, False
        output_items: List[Any] = []
        touched = 0
        skipped = 0
        ok = False
        for index in range(length):
            item, item_touched, item_skipped, item_ok = trimmed_mean_values(
                [value[index] for value in values],
                trim_count,
            )
            output_items.append(item if item_ok else clone_value(values[0][index]))
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return tuple(output_items), touched, skipped, ok

    return clone_value(values[0]), 0, 1, False


def median_values(values: Sequence[Any]) -> Tuple[Any, int, int, bool]:
    values = [value for value in values if value is not None]
    if not values:
        return None, 0, 1, False

    if all(torch.is_tensor(value) for value in values):
        first = values[0]
        if (
            any(value.is_sparse for value in values)
            or not torch.is_floating_point(first)
            or any(value.shape != first.shape for value in values)
        ):
            return clone_value(first), 0, 1, False
        stacked = torch.stack(
            [value.detach().to(device=first.device, dtype=first.dtype) for value in values],
            dim=0,
        )
        return torch.median(stacked, dim=0).values, 1, 0, True

    if all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
        sorted_values = sorted(float(value) for value in values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2:
            median_value = sorted_values[mid]
        else:
            median_value = (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        return float(median_value), 1, 0, True

    if all(isinstance(value, dict) for value in values):
        first = values[0]
        output: Dict[Any, Any] = {}
        touched = 0
        skipped = 0
        ok = False
        for key in first.keys():
            if not all(key in value for value in values):
                output[key] = clone_value(first[key])
                skipped += 1
                continue
            item, item_touched, item_skipped, item_ok = median_values(
                [value[key] for value in values]
            )
            output[key] = item if item_ok else clone_value(first[key])
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return output, touched, skipped, ok

    if all(isinstance(value, list) for value in values):
        length = len(values[0])
        if any(len(value) != length for value in values):
            return clone_value(values[0]), 0, 1, False
        output = []
        touched = 0
        skipped = 0
        ok = False
        for index in range(length):
            item, item_touched, item_skipped, item_ok = median_values(
                [value[index] for value in values]
            )
            output.append(item if item_ok else clone_value(values[0][index]))
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return output, touched, skipped, ok

    if all(isinstance(value, tuple) for value in values):
        length = len(values[0])
        if any(len(value) != length for value in values):
            return clone_value(values[0]), 0, 1, False
        output = []
        touched = 0
        skipped = 0
        ok = False
        for index in range(length):
            item, item_touched, item_skipped, item_ok = median_values(
                [value[index] for value in values]
            )
            output.append(item if item_ok else clone_value(values[0][index]))
            touched += item_touched
            skipped += item_skipped
            ok = ok or item_ok
        return tuple(output), touched, skipped, ok

    return clone_value(values[0]), 0, 1, False


def replace_all_with_prototype(
    participant_params: MutableMapping[str, Any],
    prototype: Any,
) -> MutableMapping[str, Any]:
    return {client_id: clone_value(prototype) for client_id in participant_params.keys()}


def fallback_trim_count(participant_count: int, trim_ratio: float) -> int:
    if participant_count <= 2:
        return 0
    safe_ratio = min(max(float(trim_ratio), 0.0), 0.49)
    trim_count = int(math.floor(participant_count * safe_ratio))
    return max(0, min(trim_count, (participant_count - 1) // 2))
