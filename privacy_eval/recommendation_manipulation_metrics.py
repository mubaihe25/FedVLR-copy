"""Recommendation manipulation metrics from baseline/attack/defense TopK files.

This module measures list-level changes and optional target-item exposure. It
does not infer attacker intent when target_items.json is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute TopK recommendation manipulation metrics."
    )
    parser.add_argument("--baseline-topk")
    parser.add_argument("--attack-topk")
    parser.add_argument("--defense-topk")
    parser.add_argument("--target-items")
    parser.add_argument("--target-interaction-plan")
    parser.add_argument(
        "--injected-users",
        help="Comma-separated injected user ids, or a JSON/CSV/TXT file containing injected users.",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    return parser


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


def is_top_column(key: str) -> bool:
    normalized = normalize_key(key)
    return normalized.startswith("top") and normalized[3:].isdigit()


def read_topk(path: Path, topk: int = 50) -> Tuple[Dict[str, List[str]], List[str]]:
    rows = read_rows(path)
    warnings: List[str] = []
    if not rows:
        return {}, ["TopK file could not be parsed: {}".format(path)]

    per_user: Dict[str, List[Tuple[int, str]]] = {}
    for row_index, row in enumerate(rows):
        user_id = row_value(row, ("user_id", "client_id", "id", "user")) or str(row_index)
        top_columns = sorted(
            [key for key in row.keys() if is_top_column(str(key))],
            key=lambda key: int(normalize_key(key)[3:] or "0"),
        )
        if top_columns:
            for rank, column in enumerate(top_columns[:topk], start=1):
                item_id = row.get(column)
                if item_id not in (None, ""):
                    per_user.setdefault(str(user_id), []).append((rank, str(item_id)))
            continue

        item_id = row_value(row, ("item_id", "recommended_item_id", "item", "recommended_item"))
        rank_value = row_value(row, ("rank", "position", "top_rank"))
        if item_id is None:
            continue
        try:
            rank = int(float(rank_value)) if rank_value is not None else len(per_user.get(str(user_id), [])) + 1
        except ValueError:
            rank = len(per_user.get(str(user_id), [])) + 1
        per_user.setdefault(str(user_id), []).append((rank, str(item_id)))

    normalized: Dict[str, List[str]] = {}
    for user_id, ranked_items in per_user.items():
        deduped: List[str] = []
        for _, item_id in sorted(ranked_items, key=lambda value: value[0]):
            if item_id not in deduped:
                deduped.append(item_id)
            if len(deduped) >= topk:
                break
        if deduped:
            normalized[user_id] = deduped
    if not normalized:
        warnings.append("TopK file has no readable recommendation rows: {}".format(path))
    return normalized, warnings


def topk_files_from_source(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(
        [
            candidate
            for candidate in path.rglob("*.csv")
            if candidate.is_file()
            and "recommend_topk_manifest" not in candidate.name.lower()
        ],
        key=lambda item: (item.stat().st_mtime, str(item)),
    )


def read_topk_source(path: Path, topk: int = 50) -> Tuple[Dict[str, List[str]], Dict[str, Any], List[str]]:
    files = topk_files_from_source(path)
    warnings: List[str] = []
    if not files:
        return {}, {
            "source": str(path),
            "source_type": "directory" if path.suffix == "" else "file",
            "file_count": 0,
            "user_count": 0,
            "user_ids": [],
        }, ["TopK source has no readable CSV files: {}".format(path)]

    merged: Dict[str, List[str]] = {}
    duplicate_users: Set[str] = set()
    for file_path in files:
        per_file, file_warnings = read_topk(file_path, topk=topk)
        warnings.extend(file_warnings)
        for user_id, items in per_file.items():
            if user_id in merged:
                duplicate_users.add(user_id)
            merged[user_id] = items
    if duplicate_users:
        warnings.append(
            "duplicate user ids found across TopK files; later files were used for users: {}".format(
                ",".join(sorted(duplicate_users))
            )
        )
    metadata = {
        "source": str(path),
        "source_type": "file" if path.is_file() else "directory",
        "file_count": len(files),
        "user_count": len(merged),
        "user_ids": sorted(merged.keys()),
        "files": [str(file_path) for file_path in files],
    }
    if not merged:
        warnings.append("TopK source has no readable recommendation users: {}".format(path))
    return merged, metadata, warnings


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def compare_lists(
    left: Dict[str, List[str]],
    right: Optional[Dict[str, List[str]]],
) -> Tuple[Optional[float], Optional[float]]:
    if right is None:
        return None, None
    common_users = sorted(set(left) & set(right))
    if not common_users:
        return None, None
    overlaps = [len(set(left[user]) & set(right[user])) for user in common_users]
    jaccards = [jaccard(left[user], right[user]) for user in common_users]
    return float(sum(overlaps) / len(overlaps)), float(sum(jaccards) / len(jaccards))


def count_changed_items(
    baseline: Dict[str, List[str]],
    attack: Dict[str, List[str]],
) -> Tuple[int, int, int]:
    changed = 0
    injected = 0
    suppressed = 0
    for user in sorted(set(baseline) & set(attack)):
        base_items = set(baseline[user])
        attack_items = set(attack[user])
        injected += len(attack_items - base_items)
        suppressed += len(base_items - attack_items)
        changed += len(base_items ^ attack_items)
    return changed, injected, suppressed


def changed_users(
    baseline: Dict[str, List[str]],
    attack: Dict[str, List[str]],
    users: Optional[Set[str]] = None,
) -> List[str]:
    common_users = sorted(set(baseline) & set(attack))
    if users is not None:
        common_users = [user for user in common_users if user in users]
    return [user for user in common_users if baseline[user] != attack[user]]


def load_target_items(path: Optional[Path]) -> Tuple[List[str], List[str]]:
    if path is None:
        return [], ["target_items.json not provided"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return [], ["target_items.json could not be parsed: {}".format(path)]
    value = payload.get("target_items") if isinstance(payload, dict) else payload
    if not isinstance(value, list):
        return [], ["target_items field is missing or not a list"]
    items: List[str] = []
    warnings: List[str] = []
    for item in value:
        if isinstance(item, dict):
            item_id = (
                item.get("item_id")
                if item.get("item_id") is not None
                else item.get("itemID", item.get("id", item.get("raw_item_id")))
            )
            if item_id in (None, ""):
                warnings.append("target item object missing item_id: {}".format(item))
                continue
            items.append(str(item_id))
        elif item not in (None, ""):
            items.append(str(item))
    return sorted(set(items), key=lambda value: (len(value), value)), warnings


def load_injected_users(
    injected_users_arg: Optional[str],
    target_interaction_plan_path: Optional[Path],
) -> Tuple[List[str], List[str]]:
    users: List[str] = []
    warnings: List[str] = []
    if injected_users_arg:
        candidate = Path(injected_users_arg)
        if candidate.exists():
            try:
                if candidate.suffix.lower() == ".json":
                    payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
                    if isinstance(payload, dict):
                        value = payload.get("injected_users") or payload.get("user_ids") or payload.get("users")
                        if value is None and isinstance(payload.get("injected_interactions"), list):
                            value = [
                                item.get("user_id")
                                for item in payload.get("injected_interactions", [])
                                if isinstance(item, dict)
                            ]
                    else:
                        value = payload
                    if isinstance(value, list):
                        users.extend(str(item) for item in value if item not in (None, ""))
                else:
                    text = candidate.read_text(encoding="utf-8-sig")
                    for token in text.replace("\n", ",").split(","):
                        token = token.strip()
                        if token:
                            users.append(token)
            except Exception as exc:  # noqa: BLE001
                warnings.append("injected users file could not be parsed: {}".format(exc))
        else:
            users.extend(token.strip() for token in injected_users_arg.split(",") if token.strip())

    if target_interaction_plan_path is not None:
        try:
            payload = json.loads(target_interaction_plan_path.read_text(encoding="utf-8-sig"))
            interactions = payload.get("injected_interactions", []) if isinstance(payload, dict) else []
            if isinstance(interactions, list):
                users.extend(
                    str(item.get("user_id"))
                    for item in interactions
                    if isinstance(item, dict) and item.get("user_id") not in (None, "")
                )
            if not interactions and isinstance(payload, dict):
                for key in ("injected_users", "injected_clients", "malicious_client_ids"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        users.extend(str(item) for item in value if item not in (None, ""))
        except Exception as exc:  # noqa: BLE001
            warnings.append("target_interaction_plan could not be parsed: {}".format(exc))

    return sorted(set(users)), warnings


def target_hit_rate(recommendations: Optional[Dict[str, List[str]]], target_items: Set[str]) -> Optional[float]:
    if recommendations is None or not target_items:
        return None
    if not recommendations:
        return None
    hits = sum(1 for items in recommendations.values() if target_items.intersection(items))
    return float(hits / len(recommendations))


def target_hit_users(
    recommendations: Optional[Dict[str, List[str]]],
    target_items: Set[str],
) -> List[str]:
    if recommendations is None or not target_items:
        return []
    return sorted(
        user_id
        for user_id, items in recommendations.items()
        if target_items.intersection(items)
    )


def subset_recommendations(
    recommendations: Optional[Dict[str, List[str]]],
    users: Set[str],
) -> Optional[Dict[str, List[str]]]:
    if recommendations is None:
        return None
    return {user: items for user, items in recommendations.items() if user in users}


def target_exposure_count(
    recommendations: Optional[Dict[str, List[str]]],
    target_items: Set[str],
) -> Optional[int]:
    if recommendations is None or not target_items:
        return None
    return int(sum(len(target_items.intersection(items)) for items in recommendations.values()))


def best_target_rank(items: Sequence[str], target_items: Set[str], missing_rank: int) -> int:
    ranks = [index + 1 for index, item in enumerate(items) if item in target_items]
    return min(ranks) if ranks else missing_rank


def target_rank_shift(
    left: Optional[Dict[str, List[str]]],
    right: Optional[Dict[str, List[str]]],
    target_items: Set[str],
    topk: int,
) -> Optional[float]:
    if left is None or right is None or not target_items:
        return None
    common_users = sorted(set(left) & set(right))
    if not common_users:
        return None
    missing_rank = topk + 1
    shifts = []
    for user in common_users:
        left_rank = best_target_rank(left[user], target_items, missing_rank)
        right_rank = best_target_rank(right[user], target_items, missing_rank)
        shifts.append(left_rank - right_rank)
    return float(sum(shifts) / len(shifts))


def average_target_rank(
    recommendations: Optional[Dict[str, List[str]]],
    target_items: Set[str],
    topk: int,
) -> Optional[float]:
    if recommendations is None or not target_items:
        return None
    ranks = [
        best_target_rank(items, target_items, topk + 1)
        for items in recommendations.values()
    ]
    if not ranks:
        return None
    return float(sum(ranks) / len(ranks))


def risk_level(
    baseline_attack_jaccard: Optional[float],
    target_hit_baseline: Optional[float],
    target_hit_attack: Optional[float],
) -> str:
    target_gain = None
    if target_hit_baseline is not None and target_hit_attack is not None:
        target_gain = target_hit_attack - target_hit_baseline
    if baseline_attack_jaccard is not None and baseline_attack_jaccard < 0.35:
        return "high"
    if target_gain is not None and target_gain >= 0.20:
        return "high"
    if baseline_attack_jaccard is not None and baseline_attack_jaccard < 0.65:
        return "medium"
    if target_gain is not None and target_gain >= 0.05:
        return "medium"
    return "low"


def compute_recommendation_manipulation(
    baseline_topk: Path,
    attack_topk: Path,
    defense_topk: Optional[Path] = None,
    target_items_path: Optional[Path] = None,
    topk: int = 50,
    target_interaction_plan_path: Optional[Path] = None,
    injected_users_arg: Optional[str] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    baseline, baseline_source, baseline_warnings = read_topk_source(baseline_topk, topk=topk)
    attack, attack_source, attack_warnings = read_topk_source(attack_topk, topk=topk)
    defense = None
    defense_source = None
    defense_warnings: List[str] = []
    if defense_topk is not None:
        defense, defense_source, defense_warnings = read_topk_source(defense_topk, topk=topk)
    warnings.extend("baseline: {}".format(warning) for warning in baseline_warnings)
    warnings.extend("attack: {}".format(warning) for warning in attack_warnings)
    warnings.extend("defense: {}".format(warning) for warning in defense_warnings)

    target_items, target_warnings = load_target_items(target_items_path)
    warnings.extend(target_warnings)
    target_set = set(target_items)
    injected_users, injected_warnings = load_injected_users(
        injected_users_arg,
        target_interaction_plan_path,
    )
    warnings.extend(injected_warnings)
    injected_user_set = set(injected_users)

    baseline_attack_overlap, baseline_attack_jaccard = compare_lists(baseline, attack)
    attack_defense_overlap, attack_defense_jaccard = compare_lists(attack, defense)
    baseline_defense_overlap, baseline_defense_jaccard = compare_lists(baseline, defense)
    changed_count, injected_count, suppressed_count = count_changed_items(baseline, attack)

    target_hit_baseline = target_hit_rate(baseline, target_set)
    target_hit_attack = target_hit_rate(attack, target_set)
    target_hit_defense = target_hit_rate(defense, target_set)
    target_exposure_baseline = target_exposure_count(baseline, target_set)
    target_exposure_attack = target_exposure_count(attack, target_set)
    target_exposure_defense = target_exposure_count(defense, target_set)
    rank_shift_attack = target_rank_shift(baseline, attack, target_set, topk)
    rank_shift_defense = target_rank_shift(attack, defense, target_set, topk)
    exposure_gain = (
        float(target_hit_attack - target_hit_baseline)
        if target_hit_attack is not None and target_hit_baseline is not None
        else None
    )
    exposure_recovery = (
        float(target_hit_defense - target_hit_attack)
        if target_hit_defense is not None and target_hit_attack is not None
        else None
    )
    common_user_set = set(baseline) & set(attack)
    evaluated_user_count = len(common_user_set) if common_user_set else len(attack)
    attack_target_hit_users = target_hit_users(attack, target_set)
    baseline_target_hit_users = target_hit_users(baseline, target_set)
    changed_user_ids = changed_users(baseline, attack)

    injected_baseline = subset_recommendations(baseline, injected_user_set)
    injected_attack = subset_recommendations(attack, injected_user_set)
    injected_common_user_set = set(injected_baseline or {}) & set(injected_attack or {})
    injected_users_with_attack_topk = sorted(injected_user_set & set(attack))
    injected_target_hit_users_attack = target_hit_users(injected_attack, target_set)
    injected_target_hit_users_baseline = target_hit_users(injected_baseline, target_set)
    injected_target_hit_rate_baseline = target_hit_rate(injected_baseline, target_set)
    injected_target_hit_rate_attack = target_hit_rate(injected_attack, target_set)
    if injected_users and not injected_users_with_attack_topk:
        warnings.append("no injected users were found in attack TopK source")
    if target_items and not attack_target_hit_users:
        warnings.append(
            "target items do not appear in attack TopK; ranks beyond TopK require target_rank_summary or score export"
        )

    return {
        "metric_type": "recommendation_manipulation",
        "topk": int(topk),
        "baseline_topk_source": baseline_source,
        "attack_topk_source": attack_source,
        "defense_topk_source": defense_source,
        "baseline_attack_overlap": baseline_attack_overlap,
        "baseline_attack_jaccard": baseline_attack_jaccard,
        "attack_defense_overlap": attack_defense_overlap,
        "attack_defense_jaccard": attack_defense_jaccard,
        "baseline_defense_overlap": baseline_defense_overlap,
        "baseline_defense_jaccard": baseline_defense_jaccard,
        "changed_item_count": int(changed_count),
        "injected_item_count": int(injected_count),
        "suppressed_item_count": int(suppressed_count),
        "target_items_available": bool(target_items),
        "target_items": target_items,
        "baseline_user_count": len(baseline),
        "attack_user_count": len(attack),
        "defense_user_count": len(defense) if defense is not None else None,
        "common_user_count": len(common_user_set),
        "evaluated_user_count": int(evaluated_user_count),
        "changed_user_count": len(changed_user_ids),
        "changed_users": changed_user_ids,
        "target_hit_user_count": len(attack_target_hit_users),
        "target_hit_users": attack_target_hit_users,
        "target_hit_users_baseline": baseline_target_hit_users,
        "target_hit_rate_baseline": target_hit_baseline,
        "target_hit_rate_attack": target_hit_attack,
        "target_hit_rate_defense": target_hit_defense,
        "target_hit_rate_at_k_baseline": target_hit_baseline,
        "target_hit_rate_at_k_attack": target_hit_attack,
        "target_hit_rate_at_k_defense": target_hit_defense,
        "target_exposure_count_baseline": target_exposure_baseline,
        "target_exposure_count_attack": target_exposure_attack,
        "target_exposure_count_defense": target_exposure_defense,
        "target_exposure_gain": exposure_gain,
        "target_exposure_recovery": exposure_recovery,
        "target_average_rank_baseline": average_target_rank(baseline, target_set, topk),
        "target_average_rank_attack": average_target_rank(attack, target_set, topk),
        "target_average_rank_defense": average_target_rank(defense, target_set, topk),
        "target_rank_shift_attack": rank_shift_attack,
        "target_rank_shift_defense": rank_shift_defense,
        "injected_user_count": len(injected_users),
        "injected_users": injected_users,
        "injected_users_with_topk_count": len(injected_users_with_attack_topk),
        "injected_users_with_attack_topk_count": len(injected_users_with_attack_topk),
        "injected_users_with_baseline_topk_count": len(injected_user_set & set(baseline)),
        "injected_users_with_topk": injected_users_with_attack_topk,
        "injected_common_user_count": len(injected_common_user_set),
        "injected_target_hit_rate_baseline": injected_target_hit_rate_baseline,
        "injected_target_hit_rate_attack": injected_target_hit_rate_attack,
        "injected_target_hit_user_count_baseline": len(injected_target_hit_users_baseline),
        "injected_target_hit_user_count_attack": len(injected_target_hit_users_attack),
        "injected_target_hit_users_baseline": injected_target_hit_users_baseline,
        "injected_target_hit_users_attack": injected_target_hit_users_attack,
        "injected_user_changed_count": len(changed_users(baseline, attack, injected_user_set)),
        "injected_changed_users": changed_users(baseline, attack, injected_user_set),
        "manipulation_risk_level": risk_level(
            baseline_attack_jaccard,
            target_hit_baseline,
            target_hit_attack,
        ),
        "warnings": warnings,
        "note": (
            "Lower baseline_attack_jaccard means stronger list shift. Positive "
            "target_rank_shift_attack means target items moved upward under attack."
        ),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_synthetic_smoke() -> Dict[str, Any]:
    baseline = {
        "u1": ["i1", "i2", "i3"],
        "u2": ["i2", "i4", "i5"],
    }
    attack = {
        "u1": ["target", "i2", "i8"],
        "u2": ["target", "i4", "i9"],
    }
    defense = {
        "u1": ["i1", "i2", "target"],
        "u2": ["i2", "i4", "target"],
    }
    base_attack_overlap, base_attack_jaccard = compare_lists(baseline, attack)
    attack_defense_overlap, attack_defense_jaccard = compare_lists(attack, defense)
    baseline_defense_overlap, baseline_defense_jaccard = compare_lists(baseline, defense)
    changed, injected, suppressed = count_changed_items(baseline, attack)
    target_set = {"target"}
    summary = {
        "metric_type": "recommendation_manipulation",
        "topk": 3,
        "baseline_attack_overlap": base_attack_overlap,
        "baseline_attack_jaccard": base_attack_jaccard,
        "attack_defense_overlap": attack_defense_overlap,
        "attack_defense_jaccard": attack_defense_jaccard,
        "baseline_defense_overlap": baseline_defense_overlap,
        "baseline_defense_jaccard": baseline_defense_jaccard,
        "changed_item_count": changed,
        "injected_item_count": injected,
        "suppressed_item_count": suppressed,
        "target_items_available": True,
        "target_hit_rate_baseline": target_hit_rate(baseline, target_set),
        "target_hit_rate_attack": target_hit_rate(attack, target_set),
        "target_hit_rate_defense": target_hit_rate(defense, target_set),
        "target_hit_rate_at_k_baseline": target_hit_rate(baseline, target_set),
        "target_hit_rate_at_k_attack": target_hit_rate(attack, target_set),
        "target_hit_rate_at_k_defense": target_hit_rate(defense, target_set),
        "target_exposure_count_baseline": target_exposure_count(baseline, target_set),
        "target_exposure_count_attack": target_exposure_count(attack, target_set),
        "target_exposure_count_defense": target_exposure_count(defense, target_set),
        "target_exposure_gain": target_hit_rate(attack, target_set) - target_hit_rate(baseline, target_set),
        "target_exposure_recovery": target_hit_rate(defense, target_set) - target_hit_rate(attack, target_set),
        "target_average_rank_baseline": average_target_rank(baseline, target_set, 3),
        "target_average_rank_attack": average_target_rank(attack, target_set, 3),
        "target_average_rank_defense": average_target_rank(defense, target_set, 3),
        "target_rank_shift_attack": target_rank_shift(baseline, attack, target_set, 3),
        "target_rank_shift_defense": target_rank_shift(attack, defense, target_set, 3),
        "manipulation_risk_level": risk_level(
            base_attack_jaccard,
            target_hit_rate(baseline, target_set),
            target_hit_rate(attack, target_set),
        ),
        "warnings": [],
    }
    assert summary["baseline_attack_jaccard"] is not None
    assert summary["target_hit_rate_attack"] == 1.0
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        summary = run_synthetic_smoke()
    else:
        if not args.baseline_topk or not args.attack_topk:
            raise SystemExit("--baseline-topk and --attack-topk are required unless --smoke is used")
        summary = compute_recommendation_manipulation(
            baseline_topk=Path(args.baseline_topk),
            attack_topk=Path(args.attack_topk),
            defense_topk=Path(args.defense_topk) if args.defense_topk else None,
            target_items_path=Path(args.target_items) if args.target_items else None,
            topk=args.topk,
            target_interaction_plan_path=Path(args.target_interaction_plan) if args.target_interaction_plan else None,
            injected_users_arg=args.injected_users,
        )
    write_json(Path(args.output_file), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
