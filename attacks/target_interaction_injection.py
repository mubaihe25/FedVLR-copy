"""Target interaction injection planner and default-off local-data hook.

By default this module is planner-only. When the attack is enabled in a run and
``target_interaction_injection.enabled=true`` is passed through the unified
config, it injects target positive interactions into malicious clients'
in-memory local dataloaders before local training. It never rewrites dataset
files on disk.
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
    """Target promotion planner with an optional in-memory local-data hook."""

    attack_family = "poisoning"
    attack_category = "target_interaction_injection_planner"
    attack_strategy = "target_item_positive_interaction_plan"
    attack_display_category = "poisoning attack"
    mutates_participant_params = False
    is_read_only = False

    def __init__(
        self,
        name: str = "target_interaction_injection",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.enabled = bool(
            self.config.get(
                "target_interaction_injection_enabled",
                self.config.get("enabled", False),
            )
        )
        self.target_item_ids = [
            str(item)
            for item in self._as_list(
                self.config.get(
                    "target_item_ids",
                    self.config.get("targeted_item_ids", []),
                )
            )
        ]
        self.injection_ratio = float(self.config.get("injection_ratio", 0.0) or 0.0)
        self.max_injections_per_client = int(
            self.config.get("max_injections_per_client", 5) or 5
        )
        self.target_user_strategy = str(self.config.get("target_user_strategy", "random"))
        self.repeat_target_interactions = bool(self.config.get("repeat_target_interactions", False))
        self.rating_value = float(self.config.get("rating_value", 1.0) or 1.0)
        self.only_non_train_positive_targets = bool(
            self.config.get("only_non_train_positive_targets", True)
        )
        self.target_promotion_loss_enabled = bool(
            self.config.get("target_promotion_loss_enabled", False)
            or self.config.get("target_promotion_loss", False)
        )
        self.target_promotion_loss_lambda = float(
            self.config.get(
                "loss_weight",
                self.config.get(
                    "target_promotion_loss_weight",
                    self.config.get("target_promotion_loss_lambda", 0.0),
                ),
            )
            or 0.0
        )
        self.injected_clients: List[str] = []
        self.injected_interactions: List[Dict[str, Any]] = []
        self.per_user_injected_items: Dict[str, List[str]] = defaultdict(list)
        self.warnings: List[str] = []
        self.last_summary: Dict[str, Any] = self._summary_from_config()

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value in (None, ""):
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _summary_from_config(self) -> Dict[str, Any]:
        return {
            **self.semantic_metadata(),
            "attack_type": "target_interaction_injection",
            "status": "hook_enabled" if self.enabled else "planner_only",
            "proxy_only": not self.enabled,
            "hook_enabled": self.enabled,
            "target_item_ids": list(self.target_item_ids),
            "target_item_count": len(self.target_item_ids),
            "target_item_strategy": self.config.get("target_item_strategy", "explicit_or_high_frequency"),
            "target_user_strategy": self.target_user_strategy,
            "injection_ratio": self.injection_ratio,
            "repeat_target_interactions": self.repeat_target_interactions,
            "rating_value": self.rating_value,
            "only_non_train_positive_targets": self.only_non_train_positive_targets,
            "target_promotion_loss": {
                "enabled": self.target_promotion_loss_enabled,
                "lambda": self.target_promotion_loss_lambda,
                "loss_weight": self.target_promotion_loss_lambda,
                "status": "feasibility_only",
                "reason": "FedVLR local train path does not expose a stable target-score loss hook for all models.",
            },
            "malicious_client_count": None,
            "planned_interaction_count": 0,
            "injected_client_count": 0,
            "injected_interaction_count": 0,
            "per_user_injected_items": {},
            "requires_training_data_hook": not self.enabled,
            "does_not_modify_training_data": not self.enabled,
            "mutates_local_training_data": self.enabled,
            "note": (
                "Default-off target item promotion. With enabled=true it mutates "
                "only in-memory malicious client dataloaders for the current run; "
                "dataset files are not rewritten."
            ),
            "warnings": (
                []
                if self.enabled
                else ["planner only; no training data was modified"]
            ),
        }

    @staticmethod
    def _loader_columns(client_loader: Any) -> Tuple[Optional[str], Optional[str]]:
        dataset = getattr(client_loader, "dataset", None)
        return getattr(dataset, "uid_field", None), getattr(dataset, "iid_field", None)

    def _valid_target_items(self, client_loader: Any) -> List[int]:
        dataset = getattr(client_loader, "dataset", None)
        item_num = getattr(dataset, "item_num", None)
        valid_items: List[int] = []
        for raw_item in self.target_item_ids:
            try:
                item_id = int(raw_item)
            except (TypeError, ValueError):
                self.warnings.append("target item is not an integer internal item id: {}".format(raw_item))
                continue
            if item_num is not None and not (0 <= item_id < int(item_num)):
                self.warnings.append("target item out of range for local dataset: {}".format(raw_item))
                continue
            valid_items.append(item_id)
        return valid_items

    def before_client_train(
        self,
        client_id: Any,
        client_loader: Any,
        round_state: MutableMapping[str, Any],
    ) -> Any:
        if not self.enabled:
            return client_loader
        malicious_clients = {str(value) for value in round_state.get("malicious_clients", [])}
        client_id_str = str(client_id)
        if client_id_str not in malicious_clients:
            return client_loader
        if self.injection_ratio <= 0:
            self.warnings.append("injection_ratio <= 0; no target interactions injected")
            return client_loader

        try:
            import pandas as pd
        except Exception as exc:  # noqa: BLE001
            self.warnings.append("pandas unavailable for target interaction hook: {}".format(exc))
            return client_loader

        dataset = getattr(client_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "df"):
            self.warnings.append("client loader has no mutable dataset.df")
            return client_loader
        uid_field, iid_field = self._loader_columns(client_loader)
        if not uid_field or not iid_field:
            self.warnings.append("client loader missing uid/iid fields")
            return client_loader
        valid_targets = self._valid_target_items(client_loader)
        if not valid_targets:
            return client_loader

        df = dataset.df
        existing_items = set(df[iid_field].astype(int).tolist()) if iid_field in df else set()
        if self.only_non_train_positive_targets:
            candidate_targets = [item for item in valid_targets if item not in existing_items]
        else:
            candidate_targets = list(valid_targets)
        if not candidate_targets:
            self.warnings.append("all target items already present for client {}".format(client_id_str))
            return client_loader

        base_count = max(1, len(df))
        injection_count = max(1, int(round(base_count * self.injection_ratio)))
        injection_count = min(injection_count, self.max_injections_per_client)
        if not self.repeat_target_interactions:
            injection_count = min(injection_count, len(candidate_targets))
        rows = []
        for index in range(injection_count):
            item_id = candidate_targets[index % len(candidate_targets)]
            new_row = {column: None for column in df.columns}
            new_row[uid_field] = int(client_id) if str(client_id).isdigit() else client_id
            new_row[iid_field] = int(item_id)
            if "rating" in new_row:
                new_row["rating"] = self.rating_value
            rows.append(new_row)

        if not rows:
            return client_loader
        dataset.df = pd.concat([df, pd.DataFrame(rows, columns=df.columns)], ignore_index=True)
        dataset.inter_num = len(dataset.df)
        history = getattr(client_loader, "history_items_per_u", None)
        if isinstance(history, dict):
            history.setdefault(int(client_id) if str(client_id).isdigit() else client_id, set()).update(
                row[iid_field] for row in rows
            )
        if hasattr(client_loader, "all_items"):
            current_items = list(getattr(client_loader, "all_items", []))
            for row in rows:
                if row[iid_field] not in current_items:
                    current_items.append(row[iid_field])
            client_loader.all_items = current_items
        if hasattr(client_loader, "all_items_set"):
            client_loader.all_items_set = set(getattr(client_loader, "all_items_set", set())) | {
                row[iid_field] for row in rows
            }

        self.injected_clients.append(client_id_str)
        for row in rows:
            self.per_user_injected_items[str(row[uid_field])].append(str(row[iid_field]))
            self.injected_interactions.append(
                {
                    "user_id": str(row[uid_field]),
                    "item_id": str(row[iid_field]),
                    "rating": self.rating_value,
                    "source": "in_memory_target_injection_hook",
                }
            )
        self.last_summary = self._summary_from_config()
        self.last_summary.update(
            {
                "malicious_client_ids": sorted(malicious_clients),
                "malicious_client_count": len(malicious_clients),
                "injected_clients": sorted(set(self.injected_clients)),
                "injected_client_count": len(set(self.injected_clients)),
                "injected_interaction_count": len(self.injected_interactions),
                "planned_interaction_count": len(self.injected_interactions),
                "injected_interactions": list(self.injected_interactions),
                "per_user_injected_items": {
                    user_id: list(items)
                    for user_id, items in sorted(self.per_user_injected_items.items())
                },
                "status": "hook_active",
                "proxy_only": False,
                "requires_training_data_hook": False,
                "does_not_modify_training_data": False,
                "warnings": sorted(set(self.warnings)),
            }
        )
        round_state.setdefault("attack_outputs", {})[self.name] = dict(self.last_summary)
        return client_loader

    def before_round(
        self, round_state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        malicious_clients = [str(client_id) for client_id in round_state.get("malicious_clients", [])]
        self.last_summary.update(
            {
                "malicious_client_ids": malicious_clients,
                "malicious_client_count": len(malicious_clients),
                "injected_clients": sorted(set(self.injected_clients)),
                "injected_client_count": len(set(self.injected_clients)),
                "injected_interaction_count": len(self.injected_interactions),
                "injected_interactions": list(self.injected_interactions),
                "per_user_injected_items": {
                    user_id: list(items)
                    for user_id, items in sorted(self.per_user_injected_items.items())
                },
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
    parser.add_argument(
        "--target-user-strategy",
        default="random",
        choices=["random", "active_users", "evaluated_users", "rank_near_top_users"],
    )
    parser.add_argument("--repeat-target-interactions", action="store_true")
    parser.add_argument("--rating-value", type=float, default=1.0)
    parser.add_argument("--only-non-train-positive-targets", action="store_true", default=True)
    parser.add_argument("--target-promotion-loss", action="store_true")
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hook-smoke", action="store_true")
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
    target_user_strategy: str = "random",
    pairs_by_user: Optional[Dict[str, Set[str]]] = None,
) -> List[str]:
    if explicit:
        return sorted({str(user) for user in explicit})
    ordered = sorted(users, key=lambda value: (len(str(value)), str(value)))
    if target_user_strategy == "active_users" and pairs_by_user:
        ordered = sorted(
            ordered,
            key=lambda user_id: (-len(pairs_by_user.get(str(user_id), set())), len(str(user_id)), str(user_id)),
        )
    if not ordered:
        return []
    count = max(1, int(round(len(ordered) * max(0.0, min(1.0, ratio)))))
    if target_user_strategy == "active_users":
        return sorted(ordered[: min(count, len(ordered))], key=lambda value: (len(str(value)), str(value)))
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
    target_user_strategy: str = "random",
    repeat_target_interactions: bool = False,
    rating_value: float = 1.0,
    only_non_train_positive_targets: bool = True,
    target_promotion_loss: bool = False,
    loss_weight: float = 0.0,
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
        target_user_strategy=target_user_strategy,
        pairs_by_user=pairs_by_user,
    )
    planned_interactions: List[Dict[str, Any]] = []
    per_client_target_count = max(1, int(round(len(target_items) * max(0.0, injection_ratio)))) if target_items else 0
    per_client_target_count = min(per_client_target_count, max(0, max_injections_per_client))
    for client_id in malicious_clients:
        existing_items = pairs_by_user.get(str(client_id), set())
        candidate_targets = [
            item for item in target_items if (not only_non_train_positive_targets or item not in existing_items)
        ]
        if not candidate_targets:
            continue
        selected_count = per_client_target_count
        if not repeat_target_interactions:
            selected_count = min(selected_count, len(candidate_targets))
        for index in range(selected_count):
            item_id = candidate_targets[index % len(candidate_targets)]
            planned_interactions.append(
                {
                    "user_id": str(client_id),
                    "item_id": str(item_id),
                    "rating": rating_value,
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
        "target_user_strategy": target_user_strategy,
        "injection_ratio": injection_ratio,
        "repeat_target_interactions": repeat_target_interactions,
        "rating_value": rating_value,
        "only_non_train_positive_targets": only_non_train_positive_targets,
        "target_promotion_loss": {
            "enabled": bool(target_promotion_loss),
            "lambda": float(loss_weight or 0.0),
            "loss_weight": float(loss_weight or 0.0),
            "status": "feasibility_only",
            "reason": "Planner cannot add model-specific target-score loss without the training hook.",
        },
        "malicious_client_ids": malicious_clients,
        "malicious_client_count": len(malicious_clients),
        "planned_interaction_count": len(planned_interactions),
        "planned_interactions": planned_interactions,
        "per_user_injected_items": {
            client_id: [item["item_id"] for item in planned_interactions if item["user_id"] == client_id]
            for client_id in malicious_clients
        },
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


def run_hook_smoke(output_json: Path) -> Dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        return {
            "attack_type": "target_interaction_injection",
            "status": "not_available",
            "hook_smoke_available": False,
            "warnings": ["pandas unavailable for hook smoke: {}".format(exc)],
        }

    class FakeDataset:
        uid_field = DEFAULT_USER_FIELD
        iid_field = DEFAULT_ITEM_FIELD
        item_num = 8

        def __init__(self) -> None:
            self.df = pd.DataFrame(
                [
                    {DEFAULT_USER_FIELD: 1, DEFAULT_ITEM_FIELD: 0, "rating": 1.0},
                    {DEFAULT_USER_FIELD: 1, DEFAULT_ITEM_FIELD: 4, "rating": 1.0},
                ]
            )
            self.inter_num = len(self.df)

    class FakeLoader:
        def __init__(self) -> None:
            self.dataset = FakeDataset()
            self.history_items_per_u = {1: {0, 4}}
            self.all_items = [0, 4]
            self.all_items_set = {0, 4}

    loader = FakeLoader()
    attack = TargetInteractionInjectionPlanner(
        config={
            "enabled": True,
            "target_item_ids": ["2", "3"],
            "injection_ratio": 1.0,
            "max_injections_per_client": 2,
        }
    )
    before_count = len(loader.dataset.df)
    updated_loader = attack.before_client_train(
        client_id=1,
        client_loader=loader,
        round_state={"malicious_clients": [1]},
    )
    summary = attack.collect_metrics()
    summary.update(
        {
            "hook_smoke_available": True,
            "row_count_before": before_count,
            "row_count_after": len(updated_loader.dataset.df),
            "dataset_file_modified": False,
        }
    )
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    output_json = Path(args.output_json)
    if args.hook_smoke:
        summary = run_hook_smoke(output_json)
    elif args.smoke:
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
            target_user_strategy=args.target_user_strategy,
            repeat_target_interactions=args.repeat_target_interactions,
            rating_value=args.rating_value,
            only_non_train_positive_targets=args.only_non_train_positive_targets,
            target_promotion_loss=args.target_promotion_loss,
            loss_weight=args.loss_weight,
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
