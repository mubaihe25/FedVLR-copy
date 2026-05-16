"""Target interaction injection planner.

This module does not mutate FedVLR training data. It plans which malicious
clients would receive synthetic positive interactions for target items and
writes a sidecar plan that can be inspected or wired into a future data-loader
hook. The current implementation is planner-only by design.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.base_attack import BaseAttack
from attacks.registry import register_attack


DEFAULT_USER_FIELD = "userID"
DEFAULT_ITEM_FIELD = "itemID"
DEFAULT_SPLIT_FIELD = "split_label"


class TargetInteractionInjectionPlanner(BaseAttack):
    """Planner-only target promotion via local interaction injection."""

    attack_family = "poisoning"
    attack_category = "target_interaction_injection_planner"
    attack_strategy = "target_item_positive_interaction_plan"
    attack_display_category = "poisoning attack"
    mutates_participant_params = False
    is_read_only = True

    def __init__(
        self,
        name: str = "target_interaction_injection",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.last_summary: Dict[str, Any] = self._summary_from_config()

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value in (None, ""):
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _summary_from_config(self) -> Dict[str, Any]:
        target_item_ids = [str(item) for item in self._as_list(self.config.get("target_item_ids"))]
        return {
            **self.semantic_metadata(),
            "attack_type": "target_interaction_injection",
            "status": "planner_only",
            "proxy_only": True,
            "target_item_ids": target_item_ids,
            "target_item_count": len(target_item_ids),
            "target_item_strategy": self.config.get("target_item_strategy", "explicit_or_high_frequency"),
            "injection_ratio": self.config.get("injection_ratio"),
            "malicious_client_count": None,
            "planned_interaction_count": 0,
            "requires_training_data_hook": True,
            "does_not_modify_training_data": True,
            "note": (
                "Planner-only target item promotion. It does not modify FedVLR local "
                "training samples until a future data-loader hook consumes the plan."
            ),
            "warnings": ["standalone planner only; no training data was modified"],
        }

    def before_round(
        self, round_state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        malicious_clients = [str(client_id) for client_id in round_state.get("malicious_clients", [])]
        self.last_summary.update(
            {
                "malicious_client_ids": malicious_clients,
                "malicious_client_count": len(malicious_clients),
            }
        )
        round_state.setdefault("attack_outputs", {})[self.name] = dict(self.last_summary)
        return round_state

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self.last_summary)

    def summarize(
        self, experiment_metadata: Optional[MutableMapping[str, Any]] = None
    ) -> Dict[str, Any]:
        del experiment_metadata
        return dict(self.last_summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a planner-only malicious_interaction_plan.json sidecar."
    )
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--data-root")
    parser.add_argument("--interaction-file")
    parser.add_argument("--target-item-ids", nargs="*", default=[])
    parser.add_argument("--target-items-json")
    parser.add_argument(
        "--target-item-strategy",
        default="high_frequency",
        choices=["explicit", "high_frequency"],
    )
    parser.add_argument("--injection-ratio", type=float, default=0.1)
    parser.add_argument("--malicious-client-ids", nargs="*", default=[])
    parser.add_argument("--malicious-client-ratio", type=float, default=0.1)
    parser.add_argument("--max-injections-per-client", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def read_simple_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def detect_delimiter(path: Path) -> str:
    try:
        header = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    except Exception:
        return ","
    if "\t" in header and "," not in header:
        return "\t"
    return ","


def read_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=detect_delimiter(path)))
    except Exception:
        return []


def infer_paths(dataset: str, data_root: Optional[str], interaction_file: Optional[str]) -> Tuple[Path, str, str]:
    if interaction_file:
        config = read_simple_yaml(ROOT / "configs" / "datasets" / "{}.yaml".format(dataset.lower()))
        return Path(interaction_file), config.get("USER_ID_FIELD", DEFAULT_USER_FIELD), config.get("ITEM_ID_FIELD", DEFAULT_ITEM_FIELD)
    root = Path(data_root) if data_root else ROOT / "datasets"
    config = read_simple_yaml(ROOT / "configs" / "datasets" / "{}.yaml".format(dataset.lower()))
    dataset_dir = root / dataset
    inter_name = config.get("inter_file_name", "inter.csv")
    return dataset_dir / inter_name, config.get("USER_ID_FIELD", DEFAULT_USER_FIELD), config.get("ITEM_ID_FIELD", DEFAULT_ITEM_FIELD)


def load_target_items(path: Optional[str], explicit_items: Sequence[Any]) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    target_items = [str(item) for item in explicit_items if item not in (None, "")]
    if path:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            value = payload.get("target_items") if isinstance(payload, dict) else payload
            if isinstance(value, list):
                target_items.extend(str(item) for item in value)
            else:
                warnings.append("target_items_json does not contain a target_items list")
        except Exception as exc:
            warnings.append("target_items_json could not be parsed: {}".format(exc))
    return sorted(set(target_items), key=str), warnings


def train_pairs_by_user(
    rows: Iterable[Dict[str, str]],
    user_field: str,
    item_field: str,
) -> Dict[str, Set[str]]:
    pairs: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        if row.get(DEFAULT_SPLIT_FIELD) not in (None, "", "0"):
            continue
        user_id = row.get(user_field)
        item_id = row.get(item_field)
        if user_id not in (None, "") and item_id not in (None, ""):
            pairs[str(user_id)].add(str(item_id))
    return pairs


def choose_malicious_clients(
    users: Sequence[str],
    explicit: Sequence[Any],
    ratio: float,
    rng: random.Random,
) -> List[str]:
    if explicit:
        return sorted({str(user) for user in explicit})
    ordered = sorted(users, key=lambda value: (len(str(value)), str(value)))
    if not ordered:
        return []
    count = max(1, int(round(len(ordered) * max(0.0, min(1.0, ratio)))))
    return sorted(rng.sample(ordered, min(count, len(ordered))), key=lambda value: (len(str(value)), str(value)))


def choose_high_frequency_items(pairs_by_user: Dict[str, Set[str]], count: int = 3) -> List[str]:
    counter: Counter[str] = Counter()
    for items in pairs_by_user.values():
        counter.update(items)
    return [item for item, _ in counter.most_common(max(1, count))]


def build_interaction_injection_plan(
    dataset: str = "KU",
    data_root: Optional[str] = None,
    interaction_file: Optional[str] = None,
    target_item_ids: Optional[Sequence[Any]] = None,
    target_items_json: Optional[str] = None,
    target_item_strategy: str = "high_frequency",
    injection_ratio: float = 0.1,
    malicious_client_ids: Optional[Sequence[Any]] = None,
    malicious_client_ratio: float = 0.1,
    max_injections_per_client: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    interaction_path, user_field, item_field = infer_paths(dataset, data_root, interaction_file)
    rows = read_rows(interaction_path)
    warnings: List[str] = []
    if not rows:
        warnings.append("interaction file missing or unreadable: {}".format(interaction_path))
    pairs_by_user = train_pairs_by_user(rows, user_field, item_field)
    target_items, target_warnings = load_target_items(target_items_json, target_item_ids or [])
    warnings.extend(target_warnings)
    if not target_items and target_item_strategy == "high_frequency":
        target_items = choose_high_frequency_items(pairs_by_user)
        warnings.append("target_item_ids not provided; selected high-frequency items as planner targets")
    if not target_items:
        warnings.append("no target items available")

    malicious_clients = choose_malicious_clients(
        list(pairs_by_user.keys()),
        malicious_client_ids or [],
        malicious_client_ratio,
        rng,
    )
    planned_interactions: List[Dict[str, Any]] = []
    per_client_target_count = max(1, int(round(len(target_items) * max(0.0, injection_ratio)))) if target_items else 0
    per_client_target_count = min(per_client_target_count, max(0, max_injections_per_client))
    for client_id in malicious_clients:
        existing_items = pairs_by_user.get(str(client_id), set())
        candidate_targets = [item for item in target_items if item not in existing_items]
        if not candidate_targets:
            continue
        selected = candidate_targets[:per_client_target_count]
        for item_id in selected:
            planned_interactions.append(
                {
                    "user_id": str(client_id),
                    "item_id": str(item_id),
                    "rating": 1.0,
                    "source": "planned_target_injection",
                }
            )

    return {
        "attack_type": "target_interaction_injection",
        "status": "planner_only",
        "proxy_only": True,
        "dataset": dataset,
        "interaction_file": str(interaction_path),
        "user_id_field": user_field,
        "item_id_field": item_field,
        "target_item_ids": target_items,
        "target_item_count": len(target_items),
        "target_item_strategy": target_item_strategy,
        "injection_ratio": injection_ratio,
        "malicious_client_ids": malicious_clients,
        "malicious_client_count": len(malicious_clients),
        "planned_interaction_count": len(planned_interactions),
        "planned_interactions": planned_interactions,
        "requires_training_data_hook": True,
        "does_not_modify_training_data": True,
        "note": (
            "Planner-only target item promotion. This sidecar does not modify "
            "FedVLR training data or guarantee item-level backdoor success."
        ),
        "warnings": warnings + ["no training data was modified"],
    }


def run_smoke(output_json: Path) -> Dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    interaction_path = output_json.parent / "target_interaction_injection_smoke_inter.csv"
    with interaction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["userID", "itemID", "rating", "split_label"])
        writer.writeheader()
        writer.writerow({"userID": "u1", "itemID": "i1", "rating": 1.0, "split_label": 0})
        writer.writerow({"userID": "u2", "itemID": "i2", "rating": 1.0, "split_label": 0})
        writer.writerow({"userID": "u3", "itemID": "i2", "rating": 1.0, "split_label": 0})
    return build_interaction_injection_plan(
        dataset="KU",
        interaction_file=str(interaction_path),
        target_item_ids=["target"],
        target_item_strategy="explicit",
        malicious_client_ids=["u1", "u2"],
        injection_ratio=1.0,
        seed=7,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    output_json = Path(args.output_json)
    if args.smoke:
        summary = run_smoke(output_json)
    else:
        summary = build_interaction_injection_plan(
            dataset=args.dataset,
            data_root=args.data_root,
            interaction_file=args.interaction_file,
            target_item_ids=args.target_item_ids,
            target_items_json=args.target_items_json,
            target_item_strategy=args.target_item_strategy,
            injection_ratio=args.injection_ratio,
            malicious_client_ids=args.malicious_client_ids,
            malicious_client_ratio=args.malicious_client_ratio,
            max_injections_per_client=args.max_injections_per_client,
            seed=args.seed,
        )
    write_json(output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


register_attack("target_interaction_injection", TargetInteractionInjectionPlanner)
register_attack("target_interaction_injection_planner", TargetInteractionInjectionPlanner)
register_attack("target_promotion", TargetInteractionInjectionPlanner)


if __name__ == "__main__":
    raise SystemExit(main())
