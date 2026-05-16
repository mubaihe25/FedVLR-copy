"""Build membership labels from FedVLR train/test splits.

The output is a sidecar for score/rank-based membership probes. It records real
train pairs as members and non-training pairs as non-members. When a TopK file is
provided, labels are restricted to the users/items touched by that file where
possible. No score or attack result is fabricated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_USER_FIELD = "userID"
DEFAULT_ITEM_FIELD = "itemID"
DEFAULT_SPLIT_FIELD = "split_label"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate membership_labels.json from FedVLR dataset splits."
    )
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--data-root", help="Dataset root. Defaults to ./datasets.")
    parser.add_argument("--train-file", help="Optional explicit train interaction file.")
    parser.add_argument("--test-file", help="Optional explicit test/non-member file.")
    parser.add_argument("--recommend-topk-file", help="Optional TopK CSV/TSV filter file.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-members", type=int)
    parser.add_argument("--max-non-members", type=int)
    parser.add_argument("--seed", type=int, default=42)
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
    delimiter = detect_delimiter(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    except Exception:
        return []


def normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def row_value(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    normalized = {normalize_key(key): key for key in row.keys()}
    for key in keys:
        original = normalized.get(normalize_key(key))
        if original is None:
            continue
        value = row.get(original)
        if value not in (None, ""):
            return str(value)
    return None


def pair_key(user_id: Any, item_id: Any) -> str:
    return "{}::{}".format(str(user_id), str(item_id))


def pair_dict(pair: Tuple[str, str]) -> Dict[str, str]:
    return {"user_id": str(pair[0]), "item_id": str(pair[1])}


def parse_pairs(
    rows: Iterable[Dict[str, str]],
    user_field: str,
    item_field: str,
) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for row in rows:
        user_id = row_value(row, (user_field, "user_id", "userID", "id", "user"))
        item_id = row_value(row, (item_field, "item_id", "itemID", "item", "asin"))
        if user_id is not None and item_id is not None:
            pairs.add((str(user_id), str(item_id)))
    return pairs


def split_interactions(
    rows: Sequence[Dict[str, str]],
    split_field: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    warnings: List[str] = []
    if not rows:
        return [], [], ["interaction file is empty or unreadable"]
    if split_field not in rows[0]:
        return list(rows), [], ["split field not found; treating all rows as train"]
    train_rows = [row for row in rows if str(row.get(split_field)) == "0"]
    test_rows = [row for row in rows if str(row.get(split_field)) == "2"]
    if not test_rows:
        test_rows = [row for row in rows if str(row.get(split_field)) in {"1", "2"}]
        if test_rows:
            warnings.append("test split_label=2 not found; using valid/test rows as non-members")
    if not train_rows:
        warnings.append("train split_label=0 not found")
    if not test_rows:
        warnings.append("non-member split rows not found")
    return train_rows, test_rows, warnings


def is_top_column(key: str) -> bool:
    normalized = normalize_key(key)
    return normalized.startswith("top") and normalized[3:].isdigit()


def parse_topk_file(path: Optional[Path]) -> Tuple[Set[Tuple[str, str]], Set[str], Set[str], List[str]]:
    if path is None:
        return set(), set(), set(), []
    rows = read_rows(path)
    if not rows:
        return set(), set(), set(), ["TopK file could not be parsed: {}".format(path)]

    topk_pairs: Set[Tuple[str, str]] = set()
    users: Set[str] = set()
    items: Set[str] = set()
    warnings: List[str] = []
    for row in rows:
        user_id = row_value(row, ("user_id", "client_id", "id", "user"))
        if user_id is not None:
            users.add(str(user_id))

        top_columns = sorted(
            [key for key in row.keys() if is_top_column(str(key))],
            key=lambda key: int(normalize_key(key)[3:] or "0"),
        )
        if top_columns:
            for column in top_columns:
                item_id = row.get(column)
                if item_id in (None, ""):
                    continue
                items.add(str(item_id))
                if user_id is not None:
                    topk_pairs.add((str(user_id), str(item_id)))
            continue

        item_id = row_value(row, ("item_id", "recommended_item_id", "item", "recommended_item"))
        if item_id is not None:
            items.add(str(item_id))
            if user_id is not None:
                topk_pairs.add((str(user_id), str(item_id)))

    if items and not users:
        warnings.append("TopK file has item ids but no user id; item-level labels only")
    return topk_pairs, users, items, warnings


def limit_pairs(pairs: Set[Tuple[str, str]], limit: Optional[int], rng: random.Random) -> List[Tuple[str, str]]:
    ordered = sorted(pairs)
    if limit is not None and limit >= 0 and len(ordered) > limit:
        ordered = sorted(rng.sample(ordered, limit))
    return ordered


def limit_items(items: Set[str], limit: Optional[int], rng: random.Random) -> List[str]:
    ordered = sorted(items, key=lambda value: (len(str(value)), str(value)))
    if limit is not None and limit >= 0 and len(ordered) > limit:
        ordered = sorted(rng.sample(ordered, limit), key=lambda value: (len(str(value)), str(value)))
    return ordered


def infer_dataset_paths(dataset: str, data_root: Optional[str]) -> Tuple[Path, Dict[str, str]]:
    root = Path(data_root) if data_root else ROOT / "datasets"
    if root.name == dataset and (root / "inter.csv").exists():
        dataset_dir = root
    else:
        dataset_dir = root / dataset
    config = read_simple_yaml(ROOT / "configs" / "datasets" / "{}.yaml".format(dataset.lower()))
    return dataset_dir, config


def generate_membership_labels(
    dataset: str = "KU",
    data_root: Optional[str] = None,
    train_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    recommend_topk_file: Optional[Path] = None,
    max_members: Optional[int] = None,
    max_non_members: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    dataset_dir, dataset_config = infer_dataset_paths(dataset, data_root)
    user_field = dataset_config.get("USER_ID_FIELD", DEFAULT_USER_FIELD)
    item_field = dataset_config.get("ITEM_ID_FIELD", DEFAULT_ITEM_FIELD)
    inter_file_name = dataset_config.get("inter_file_name", "inter.csv")
    split_field = DEFAULT_SPLIT_FIELD
    warnings: List[str] = []

    train_rows: List[Dict[str, str]]
    test_rows: List[Dict[str, str]]
    label_source = "train_test_split"
    if train_file is not None or test_file is not None:
        train_rows = read_rows(train_file) if train_file is not None else []
        test_rows = read_rows(test_file) if test_file is not None else []
        label_source = "explicit_train_test_files"
        if not train_rows:
            warnings.append("train file missing or unreadable")
        if not test_rows:
            warnings.append("test/non-member file missing or unreadable")
    else:
        inter_path = dataset_dir / inter_file_name
        rows = read_rows(inter_path)
        train_rows, test_rows, split_warnings = split_interactions(rows, split_field)
        warnings.extend(split_warnings)

    train_pairs = parse_pairs(train_rows, user_field, item_field)
    test_pairs = parse_pairs(test_rows, user_field, item_field)
    train_items = {item_id for _, item_id in train_pairs}
    test_items = {item_id for _, item_id in test_pairs}

    topk_pairs, topk_users, topk_items, topk_warnings = parse_topk_file(recommend_topk_file)
    warnings.extend(topk_warnings)

    label_granularity = "user_item_pair"
    if recommend_topk_file is not None and topk_items and not topk_users:
        label_granularity = "item_level_proxy"

    if topk_users:
        train_pairs = {pair for pair in train_pairs if pair[0] in topk_users}
        test_pairs = {pair for pair in test_pairs if pair[0] in topk_users}
    if topk_pairs:
        topk_non_member_pairs = {pair for pair in topk_pairs if pair not in train_pairs}
        test_pairs = test_pairs.union(topk_non_member_pairs)
        label_source = "{}+topk_filter".format(label_source)
    if topk_items:
        train_items = train_items.intersection(topk_items)
        test_items = test_items.intersection(topk_items).union(topk_items - train_items)

    non_member_pairs = {pair for pair in test_pairs if pair not in train_pairs}
    member_pairs = train_pairs
    member_items = train_items
    non_member_items = {item for item in test_items if item not in train_items}

    if not member_pairs:
        warnings.append("no member user-item pairs generated")
    if not non_member_pairs:
        warnings.append("no non-member user-item pairs generated")
    if label_granularity == "item_level_proxy":
        warnings.append("membership labels are item-level proxies because user ids are unavailable")

    limited_member_pairs = limit_pairs(member_pairs, max_members, rng)
    limited_non_member_pairs = limit_pairs(non_member_pairs, max_non_members, rng)
    payload = {
        "dataset": dataset,
        "label_source": label_source,
        "label_granularity": label_granularity,
        "user_id_field": user_field,
        "item_id_field": item_field,
        "member_pairs": [pair_dict(pair) for pair in limited_member_pairs],
        "non_member_pairs": [pair_dict(pair) for pair in limited_non_member_pairs],
        "member_items": limit_items(member_items, max_members, rng),
        "non_member_items": limit_items(non_member_items, max_non_members, rng),
        "counts": {
            "member_pair_count": len(limited_member_pairs),
            "non_member_pair_count": len(limited_non_member_pairs),
            "member_item_count": len(member_items),
            "non_member_item_count": len(non_member_items),
            "topk_pair_count": len(topk_pairs),
            "topk_user_count": len(topk_users),
            "topk_item_count": len(topk_items),
        },
        "warnings": warnings,
    }
    if recommend_topk_file is not None:
        payload["recommend_topk_file"] = str(recommend_topk_file)
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    payload = generate_membership_labels(
        dataset=args.dataset,
        data_root=args.data_root,
        train_file=Path(args.train_file) if args.train_file else None,
        test_file=Path(args.test_file) if args.test_file else None,
        recommend_topk_file=Path(args.recommend_topk_file) if args.recommend_topk_file else None,
        max_members=args.max_members,
        max_non_members=args.max_non_members,
        seed=args.seed,
    )
    output_file = Path(args.output_file)
    write_json(output_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
