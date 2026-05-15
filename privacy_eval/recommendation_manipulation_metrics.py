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
    return [str(item) for item in value], []


def target_hit_rate(recommendations: Optional[Dict[str, List[str]]], target_items: Set[str]) -> Optional[float]:
    if recommendations is None or not target_items:
        return None
    if not recommendations:
        return None
    hits = sum(1 for items in recommendations.values() if target_items.intersection(items))
    return float(hits / len(recommendations))


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
) -> Dict[str, Any]:
    warnings: List[str] = []
    baseline, baseline_warnings = read_topk(baseline_topk, topk=topk)
    attack, attack_warnings = read_topk(attack_topk, topk=topk)
    defense = None
    defense_warnings: List[str] = []
    if defense_topk is not None:
        defense, defense_warnings = read_topk(defense_topk, topk=topk)
    warnings.extend("baseline: {}".format(warning) for warning in baseline_warnings)
    warnings.extend("attack: {}".format(warning) for warning in attack_warnings)
    warnings.extend("defense: {}".format(warning) for warning in defense_warnings)

    target_items, target_warnings = load_target_items(target_items_path)
    warnings.extend(target_warnings)
    target_set = set(target_items)

    baseline_attack_overlap, baseline_attack_jaccard = compare_lists(baseline, attack)
    attack_defense_overlap, attack_defense_jaccard = compare_lists(attack, defense)
    baseline_defense_overlap, baseline_defense_jaccard = compare_lists(baseline, defense)
    changed_count, injected_count, suppressed_count = count_changed_items(baseline, attack)

    target_hit_baseline = target_hit_rate(baseline, target_set)
    target_hit_attack = target_hit_rate(attack, target_set)
    target_hit_defense = target_hit_rate(defense, target_set)
    rank_shift_attack = target_rank_shift(baseline, attack, target_set, topk)
    rank_shift_defense = target_rank_shift(attack, defense, target_set, topk)

    return {
        "metric_type": "recommendation_manipulation",
        "topk": int(topk),
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
        "target_hit_rate_baseline": target_hit_baseline,
        "target_hit_rate_attack": target_hit_attack,
        "target_hit_rate_defense": target_hit_defense,
        "target_rank_shift_attack": rank_shift_attack,
        "target_rank_shift_defense": rank_shift_defense,
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
        )
    write_json(Path(args.output_file), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
