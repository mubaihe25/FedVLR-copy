"""Interaction candidate reconstruction probe from update/embedding signals.

This probe estimates candidate item ids from item-like embedding updates. It is
not image gradient inversion, not DLG, and not a full recovery of user training
history. When item embedding tensors cannot be identified, it returns
not_available instead of inventing candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_eval.base_metric import BasePrivacyMetric
from privacy_eval.registry import register_privacy_metric


ITEM_TOKENS = ("item", "item_embedding", "item_commonality", "item_personality", "embedding_item")


class InteractionReconstructionProbe(BasePrivacyMetric):
    """Infer high-change item candidates from real participant_params."""

    def __init__(
        self,
        name: str = "interaction_reconstruction_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.topk = int(self._config_get(config, "topk", 10))
        self.min_rows = int(self._config_get(config, "min_item_rows", 2))
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
                value = getter(key)
                return default if value is None else value
        return getattr(config, key, default)

    @staticmethod
    def _is_item_embedding(path: str, tensor: Any, min_rows: int) -> bool:
        normalized = path.lower()
        if not any(token in normalized for token in ITEM_TOKENS):
            return False
        if torch is not None and torch.is_tensor(tensor):
            if tensor.ndim < 2:
                return False
            return int(tensor.shape[0]) >= min_rows
        if not isinstance(tensor, list) or not tensor or not isinstance(tensor[0], list):
            return False
        return len(tensor) >= min_rows

    def _walk_item_tensors(self, value: Any, path: str = "root") -> List[Tuple[str, Any]]:
        if torch is not None and torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_sparse:
                tensor = tensor.coalesce().to_dense()
            if torch.is_floating_point(tensor) and self._is_item_embedding(path, tensor, self.min_rows):
                return [(path, tensor.float().cpu())]
            return []
        if isinstance(value, list) and self._is_item_embedding(path, value, self.min_rows):
            return [(path, value)]
        if isinstance(value, dict):
            tensors: List[Tuple[str, Any]] = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                tensors.extend(self._walk_item_tensors(value[key], "{}.{}".format(path, key)))
            return tensors
        if isinstance(value, (list, tuple)):
            tensors = []
            for index, item in enumerate(value):
                tensors.extend(self._walk_item_tensors(item, "{}[{}]".format(path, index)))
            return tensors
        return []

    @staticmethod
    def _row_norms(tensor: Any) -> List[float]:
        if torch is not None and torch.is_tensor(tensor):
            return [
                float(value)
                for value in torch.norm(tensor.reshape(tensor.shape[0], -1), dim=1).tolist()
            ]
        norms: List[float] = []
        if isinstance(tensor, list):
            for row in tensor:
                if not isinstance(row, list):
                    continue
                total = 0.0
                for value in row:
                    try:
                        total += float(value) * float(value)
                    except (TypeError, ValueError):
                        continue
                norms.append(total ** 0.5)
        return norms

    def _candidate_scores(self, participant_params: Any) -> Tuple[Dict[str, float], int, int]:
        if not isinstance(participant_params, dict):
            return {}, 0, 0
        scores: Dict[str, float] = {}
        embedding_tensor_count = 0
        client_count = 0
        for client_id, update in participant_params.items():
            tensors = self._walk_item_tensors(update, path="client_{}".format(client_id))
            if not tensors:
                continue
            client_count += 1
            for path, tensor in tensors:
                embedding_tensor_count += 1
                for item_index, value in enumerate(self._row_norms(tensor)):
                    item_id = str(item_index)
                    scores[item_id] = max(scores.get(item_id, 0.0), float(value))
        return scores, embedding_tensor_count, client_count

    @staticmethod
    def _risk_level(candidate_count: int, hit_at_k: Optional[float]) -> str:
        if hit_at_k is not None and hit_at_k >= 0.5:
            return "high"
        if candidate_count >= 10:
            return "medium"
        if candidate_count > 0:
            return "low"
        return "not_available"

    def evaluate_updates(
        self,
        participant_params: Any,
        train_item_ids: Optional[Set[str]] = None,
        source: str = "real_participant_params",
    ) -> Dict[str, Any]:
        scores, embedding_tensor_count, client_count = self._candidate_scores(participant_params)
        if not scores:
            return {
                "probe_type": "interaction_reconstruction",
                "status": "not_available",
                "source": "not_available",
                "client_count": 0,
                "embedding_tensor_count": 0,
                "candidate_item_ids": [],
                "candidate_scores": [],
                "hit_at_k": None,
                "topk": self.topk,
                "risk_level": "not_available",
                "warnings": ["no identifiable item embedding update tensors found"],
                "note": "interaction candidate reconstruction only; not full user history recovery or image DLG",
            }

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.topk]
        candidate_ids = [item_id for item_id, _ in ranked]
        hit_at_k = None
        if train_item_ids:
            hit_count = sum(1 for item_id in candidate_ids if item_id in train_item_ids)
            hit_at_k = float(hit_count / len(candidate_ids)) if candidate_ids else None

        return {
            "probe_type": "interaction_reconstruction",
            "status": "available",
            "source": source,
            "client_count": client_count,
            "embedding_tensor_count": embedding_tensor_count,
            "candidate_item_ids": candidate_ids,
            "candidate_scores": [
                {"item_id": item_id, "score": score, "rank": index + 1}
                for index, (item_id, score) in enumerate(ranked)
            ],
            "hit_at_k": hit_at_k,
            "topk": self.topk,
            "risk_level": self._risk_level(len(candidate_ids), hit_at_k),
            "warnings": [],
            "note": "interaction candidate reconstruction only; not full user history recovery or image DLG",
        }

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        del round_state, aggregation_result
        summary = self.evaluate_updates(participant_params, source="real_participant_params")
        self.history.append(summary)
        return summary

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        del experiment_metadata
        available = [entry for entry in self.history if entry.get("status") == "available"]
        if not available:
            summary = self.evaluate_updates({}, source="not_available")
            summary["num_rounds"] = 0
            return summary
        latest = dict(available[-1])
        latest["num_rounds"] = len(available)
        return latest


def read_train_item_ids(path: Optional[Path]) -> Set[str]:
    if path is None or not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return set()
    items: Set[str] = set()
    for row in rows:
        if row.get("split_label") not in (None, "", "0"):
            continue
        item_id = row.get("itemID") or row.get("item_id") or row.get("item")
        if item_id not in (None, ""):
            items.add(str(item_id))
    return items


def run_synthetic_smoke() -> Dict[str, Any]:
    participant_params = {
        "client_1": {
            "item_commonality.weight": [[0.1, 0.1], [3.0, 0.1], [0.2, 0.2], [2.5, 0.0]]
        },
        "client_2": {
            "item_personality.weight": [[0.0, 0.0], [2.2, 0.2], [0.1, 0.1], [0.3, 0.3]]
        },
    }
    probe = InteractionReconstructionProbe(config={"topk": 2})
    summary = probe.evaluate_updates(
        participant_params,
        train_item_ids={"1", "3"},
        source="synthetic",
    )
    assert summary["status"] == "available"
    assert summary["candidate_item_ids"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an interaction reconstruction synthetic smoke."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-file")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output-json")
    return parser


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        summary = run_synthetic_smoke()
    else:
        train_items = read_train_item_ids(Path(args.train_file)) if args.train_file else set()
        probe = InteractionReconstructionProbe(config={"topk": args.topk})
        summary = probe.evaluate_updates(
            {},
            train_item_ids=train_items,
            source="not_available",
        )
    if args.output_json:
        write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


register_privacy_metric("interaction_reconstruction_probe", InteractionReconstructionProbe)
register_privacy_metric("interaction_reconstruction", InteractionReconstructionProbe)


if __name__ == "__main__":
    raise SystemExit(main())
