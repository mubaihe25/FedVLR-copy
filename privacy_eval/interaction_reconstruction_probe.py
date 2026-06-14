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
MODALITY_TOKENS = {
    "item_embedding": ("item_embedding", "embedding_item", "item_commonality", "item_personality"),
    "image": ("image", "vision", "visual", "v_feat"),
    "text": ("text", "txt", "t_feat"),
    "modality": ("modality", "fusion", "multi_modal"),
}


class InteractionReconstructionProbe(BasePrivacyMetric):
    """Infer high-change item candidates from real participant_params."""

    @staticmethod
    def _normalize_topk(value: Any, default: int = 10) -> int:
        if isinstance(value, (list, tuple)):
            numeric_values = []
            for item in value:
                try:
                    numeric_values.append(int(item))
                except (TypeError, ValueError):
                    continue
            return max(numeric_values) if numeric_values else default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def __init__(
        self,
        name: str = "interaction_reconstruction_probe",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.topk = self._normalize_topk(
            self._config_get(
                config,
                "interaction_reconstruction_topk",
                self._config_get(config, "topk", 10),
            )
        )
        self.min_rows = int(self._config_get(config, "min_item_rows", 2))
        self.train_interaction_file = self._config_get(config, "interaction_file", None)
        self.dataset = self._config_get(config, "dataset", None)
        self.data_path = self._config_get(config, "data_path", None)
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

    @staticmethod
    def _modality_for_path(path: str) -> str:
        normalized = path.lower()
        for modality, tokens in MODALITY_TOKENS.items():
            if any(token in normalized for token in tokens):
                return modality
        return "item_like_update"

    @staticmethod
    def _top_candidates(scores: Dict[str, float], topk: int) -> List[Dict[str, Any]]:
        return [
            {"item_id": item_id, "score": score, "rank": index + 1}
            for index, (item_id, score) in enumerate(
                sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topk]
            )
        ]

    @staticmethod
    def _hit_at(candidate_ids: Sequence[str], train_item_ids: Optional[Set[str]], k: int) -> Optional[float]:
        if not train_item_ids:
            return None
        scoped = list(candidate_ids)[:k]
        if not scoped:
            return None
        return float(sum(1 for item_id in scoped if item_id in train_item_ids) / len(scoped))

    def _candidate_scores(
        self, participant_params: Any
    ) -> Tuple[Dict[str, float], int, int, Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
        if not isinstance(participant_params, dict):
            return {}, 0, 0, {}, {}
        scores: Dict[str, float] = {}
        per_client_scores: Dict[str, Dict[str, float]] = {}
        modality_scores: Dict[str, Dict[str, float]] = {}
        modality_tensor_counts: Dict[str, int] = {}
        embedding_tensor_count = 0
        client_count = 0
        for client_id, update in participant_params.items():
            client_id_str = str(client_id)
            tensors = self._walk_item_tensors(update, path="client_{}".format(client_id))
            if not tensors:
                continue
            client_count += 1
            client_scores = per_client_scores.setdefault(client_id_str, {})
            for path, tensor in tensors:
                embedding_tensor_count += 1
                modality = self._modality_for_path(path)
                modality_tensor_counts[modality] = modality_tensor_counts.get(modality, 0) + 1
                scoped_modality_scores = modality_scores.setdefault(modality, {})
                for item_index, value in enumerate(self._row_norms(tensor)):
                    item_id = str(item_index)
                    scores[item_id] = max(scores.get(item_id, 0.0), float(value))
                    client_scores[item_id] = max(client_scores.get(item_id, 0.0), float(value))
                    scoped_modality_scores[item_id] = max(
                        scoped_modality_scores.get(item_id, 0.0),
                        float(value),
                    )
        per_client_candidates = {
            client_id: self._top_candidates(client_scores, self.topk)
            for client_id, client_scores in sorted(per_client_scores.items())
        }
        modality_breakdown = {
            modality: {
                "embedding_tensor_count": modality_tensor_counts.get(modality, 0),
                "candidate_item_count": len(modality_scores.get(modality, {})),
                "top_candidates": self._top_candidates(modality_scores.get(modality, {}), self.topk),
            }
            for modality in sorted(modality_scores)
        }
        return scores, embedding_tensor_count, client_count, per_client_candidates, modality_breakdown

    def _default_train_file(self) -> Optional[Path]:
        if self.train_interaction_file:
            return Path(str(self.train_interaction_file))
        if self.dataset:
            base = Path(str(self.data_path)) if self.data_path else ROOT / "datasets"
            return base / str(self.dataset) / "inter.csv"
        return None

    def _round_train_items(self, round_state: MutableMapping[str, Any]) -> Tuple[Set[str], Dict[str, Any]]:
        participant_clients = {
            str(client_id)
            for client_id in round_state.get("sampled_clients", [])
        }
        train_file = self._default_train_file()
        metadata = {
            "train_interaction_file": str(train_file) if train_file else None,
            "train_item_reference_client_count": len(participant_clients),
            "train_item_reference_available": False,
        }
        if train_file is None or not train_file.exists():
            return set(), metadata
        try:
            with train_file.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            return set(), metadata
        items: Set[str] = set()
        for row in rows:
            if row.get("split_label") not in (None, "", "0"):
                continue
            user_id = row.get("userID") or row.get("user_id") or row.get("user")
            if participant_clients and str(user_id) not in participant_clients:
                continue
            item_id = row.get("itemID") or row.get("item_id") or row.get("item")
            if item_id not in (None, ""):
                items.add(str(item_id))
        metadata["train_item_reference_available"] = bool(items)
        metadata["train_item_reference_count"] = len(items)
        return items, metadata

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
        (
            scores,
            embedding_tensor_count,
            client_count,
            per_client_candidates,
            modality_breakdown,
        ) = self._candidate_scores(participant_params)
        if not scores:
            return {
                "probe_type": "interaction_reconstruction",
                "metric_type": "interaction_reconstruction",
                "summary_type": "interaction_reconstruction_summary",
                "status": "not_available",
                "source": "not_available",
                "client_count": 0,
                "embedding_tensor_count": 0,
                "candidate_item_count": 0,
                "candidate_item_ids": [],
                "candidate_scores": [],
                "per_client_candidates": {},
                "hit_at_k": None,
                "hit_at_10": None,
                "hit_at_20": None,
                "hit_at_50": None,
                "modality_breakdown": {},
                "highest_risk_modality": None,
                "reconstruction_risk_level": "not_available",
                "topk": self.topk,
                "risk_level": "not_available",
                "warnings": ["no identifiable item embedding update tensors found"],
                "note": "interaction candidate reconstruction only; not full user history recovery or image DLG",
            }

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.topk]
        candidate_ids = [item_id for item_id, _ in ranked]
        hit_at_k = self._hit_at(candidate_ids, train_item_ids, self.topk)
        hit_at_10 = self._hit_at(candidate_ids, train_item_ids, 10)
        hit_at_20 = self._hit_at(candidate_ids, train_item_ids, 20)
        hit_at_50 = self._hit_at(candidate_ids, train_item_ids, 50)
        highest_risk_modality = None
        if modality_breakdown:
            highest_risk_modality = max(
                modality_breakdown.items(),
                key=lambda item: item[1].get("candidate_item_count", 0),
            )[0]
        risk = self._risk_level(len(candidate_ids), hit_at_50 if hit_at_50 is not None else hit_at_k)

        return {
            "probe_type": "interaction_reconstruction",
            "metric_type": "interaction_reconstruction",
            "summary_type": "interaction_reconstruction_summary",
            "status": "available",
            "source": source,
            "client_count": client_count,
            "embedding_tensor_count": embedding_tensor_count,
            "candidate_item_count": len(candidate_ids),
            "candidate_item_ids": candidate_ids,
            "candidate_scores": [
                {"item_id": item_id, "score": score, "rank": index + 1}
                for index, (item_id, score) in enumerate(ranked)
            ],
            "per_client_candidates": per_client_candidates,
            "hit_at_k": hit_at_k,
            "hit_at_10": hit_at_10,
            "hit_at_20": hit_at_20,
            "hit_at_50": hit_at_50,
            "modality_breakdown": modality_breakdown,
            "highest_risk_modality": highest_risk_modality,
            "reconstruction_risk_level": risk,
            "topk": self.topk,
            "risk_level": risk,
            "warnings": [],
            "note": "interaction candidate reconstruction only; not full user history recovery or image DLG",
        }

    def evaluate_round(
        self,
        round_state: MutableMapping[str, Any],
        participant_params: MutableMapping[str, Any],
        aggregation_result: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        del aggregation_result
        train_items, reference_metadata = self._round_train_items(round_state)
        summary = self.evaluate_updates(
            participant_params,
            train_item_ids=train_items,
            source="real_participant_params",
        )
        summary.update(reference_metadata)
        if summary.get("status") == "available" and summary.get("hit_at_k") is None:
            summary.setdefault("warnings", []).append(
                "train interaction reference unavailable; hit_at_k not computed"
            )
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
