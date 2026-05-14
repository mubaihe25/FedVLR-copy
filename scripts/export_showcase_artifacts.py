from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARTIFACT_NAMES = {
    "dataset_profile": "dataset_profile.json",
    "metrics_summary": "metrics_summary.json",
    "attack_defense_summary": "attack_defense_summary.json",
    "recommendation_comparison": "recommendation_comparison.json",
    "defense_trace": "defense_trace.json",
    "privacy_risk_summary": "privacy_risk_summary.json",
    "showcase_manifest": "showcase_manifest.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export FedVLR showcase artifacts from existing result files."
    )
    parser.add_argument(
        "--result-dir",
        help="Single experiment result directory. Used for single-run artifacts.",
    )
    parser.add_argument("--baseline-dir", help="Baseline experiment result directory.")
    parser.add_argument("--attack-dir", help="Attack experiment result directory.")
    parser.add_argument("--defense-dir", help="Defense experiment result directory.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where showcase artifact JSON files will be written.",
    )
    parser.add_argument(
        "--include-synthetic-privacy-smoke",
        action="store_true",
        help="Include synthetic privacy probe smoke outputs in privacy_risk_summary.json.",
    )
    return parser


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def newest_file(files: Iterable[Path]) -> Optional[Path]:
    file_list = [path for path in files if path.is_file()]
    if not file_list:
        return None
    return sorted(file_list, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def find_result_file(result_dir: Optional[Path], suffix: str) -> Optional[Path]:
    if result_dir is None or not result_dir.exists():
        return None
    return newest_file(result_dir.rglob("*{}".format(suffix)))


def read_csv_rows(path: Optional[Path], delimiter: str = ",") -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    except Exception:
        return []


def find_round_csv(result_dir: Optional[Path]) -> Optional[Path]:
    if result_dir is None or not result_dir.exists():
        return None
    csv_files = [
        path
        for path in result_dir.rglob("*.csv")
        if "recommend_topk" not in [part.lower() for part in path.parts]
    ]
    for path in sorted(csv_files, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            header = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
        except Exception:
            continue
        if "row_type" in header or "test_recall50" in header or "best_recall50" in header:
            return path
    return newest_file(csv_files)


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def first_non_none(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def find_nested_value(obj: Any, key_names: Sequence[str]) -> Any:
    normalized_names = {normalize_key(name) for name in key_names}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if normalize_key(key) in normalized_names:
                return value
        for value in obj.values():
            found = find_nested_value(value, key_names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_nested_value(value, key_names)
            if found is not None:
                return found
    return None


def collect_nested_values(obj: Any, key_names: Sequence[str]) -> List[Any]:
    normalized_names = {normalize_key(name) for name in key_names}
    values: List[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if normalize_key(key) in normalized_names:
                values.append(value)
            values.extend(collect_nested_values(value, key_names))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(collect_nested_values(value, key_names))
    return values


def find_nested_payload(obj: Any, key_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    normalized_names = {normalize_key(name) for name in key_names}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if normalize_key(key) in normalized_names and isinstance(value, dict):
                return value
        probe_type = normalize_key(obj.get("probe_type", ""))
        if probe_type in normalized_names:
            return obj
        for value in obj.values():
            found = find_nested_payload(value, key_names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_nested_payload(value, key_names)
            if found is not None:
                return found
    return None


def metric_key_names(metric: str, scope: str) -> List[str]:
    base = ["{}50".format(metric), "{}@50".format(metric), "{}_50".format(metric)]
    if scope == "final":
        return base + [
            "test_{}50".format(metric),
            "test_{}@50".format(metric),
            "final_{}50".format(metric),
        ]
    if scope == "best":
        return [
            "best_{}50".format(metric),
            "best_{}@50".format(metric),
            "best_test_{}50".format(metric),
        ]
    if scope == "tail":
        return [
            "tail_mean_{}50".format(metric),
            "tail_{}50".format(metric),
            "tail_test_{}50".format(metric),
        ]
    return base


def read_metric_from_sources(
    sources: Sequence[Any], metric: str, scope: str
) -> Optional[float]:
    key_names = metric_key_names(metric, scope)
    for source in sources:
        value = to_float(find_nested_value(source, key_names))
        if value is not None:
            return value
    return None


def read_metric_from_csv(rows: List[Dict[str, str]], metric: str, scope: str) -> Optional[float]:
    if not rows:
        return None
    if scope == "best":
        for row in rows:
            if row.get("row_type") == "best_summary":
                return first_non_none(
                    [
                        to_float(row.get("best_{}50".format(metric))),
                        to_float(row.get("best_test_{}50".format(metric))),
                    ]
                )
    if scope == "tail":
        for row in rows:
            if row.get("row_type") == "tail_mean_summary":
                return first_non_none(
                    [
                        to_float(row.get("tail_{}50".format(metric))),
                        to_float(row.get("tail_test_{}50".format(metric))),
                    ]
                )
    if scope == "csv":
        for row in reversed(rows):
            if row.get("row_type") and row.get("row_type") != "round":
                continue
            value = first_non_none(
                [
                    to_float(row.get("test_{}50".format(metric))),
                    to_float(row.get("valid_{}50".format(metric))),
                    to_float(row.get("{}50".format(metric))),
                ]
            )
            if value is not None:
                return value
    return None


def has_metric20(sources: Sequence[Any], rows: List[Dict[str, str]], metric: str) -> bool:
    key_names = ["{}20".format(metric), "{}@20".format(metric), "test_{}20".format(metric)]
    for source in sources:
        if to_float(find_nested_value(source, key_names)) is not None:
            return True
    for row in rows:
        if any(to_float(row.get(key)) is not None for key in key_names):
            return True
    return False


class ExperimentBundle:
    def __init__(self, result_dir: Optional[Path], label: str = "result") -> None:
        self.result_dir = result_dir
        self.label = label
        self.summary_path = find_result_file(result_dir, ".experiment_summary.json")
        self.result_path = find_result_file(result_dir, ".experiment_result.json")
        self.csv_path = find_round_csv(result_dir)
        self.summary = read_json(self.summary_path)
        self.result = read_json(self.result_path)
        self.csv_rows = read_csv_rows(self.csv_path)

    @property
    def exists(self) -> bool:
        return self.result_dir is not None and self.result_dir.exists()

    @property
    def sources(self) -> List[Any]:
        return [self.summary, self.result]

    def source_dir_string(self) -> Optional[str]:
        return str(self.result_dir.resolve()) if self.result_dir is not None else None


def extract_dataset_profile(bundle: ExperimentBundle) -> Dict[str, Any]:
    sources = bundle.sources
    user_count = first_non_none(
        to_float(find_nested_value(source, ["user_count", "n_users", "num_users"]))
        for source in sources
    )
    item_count = first_non_none(
        to_float(find_nested_value(source, ["item_count", "n_items", "num_items"]))
        for source in sources
    )
    interaction_count = first_non_none(
        to_float(
            find_nested_value(source, ["interaction_count", "num_interactions", "n_interactions"])
        )
        for source in sources
    )
    sparsity = first_non_none(
        to_float(find_nested_value(source, ["sparsity"])) for source in sources
    )
    if sparsity is None and user_count and item_count and interaction_count is not None:
        denominator = user_count * item_count
        if denominator > 0:
            sparsity = 1.0 - interaction_count / denominator

    dataset = first_non_none(source.get("dataset") for source in sources if isinstance(source, dict))
    model = first_non_none(source.get("model") for source in sources if isinstance(source, dict))
    experiment_type = first_non_none(
        [
            bundle.summary.get("experiment_mode"),
            bundle.result.get("experiment_mode"),
            bundle.summary.get("scenario"),
            bundle.result.get("scenario"),
            find_nested_value(bundle.summary, ["type"]),
            find_nested_value(bundle.result, ["type"]),
        ]
    )
    warnings: List[str] = []
    if user_count is None:
        warnings.append("user_count not found in existing result metadata")
    if item_count is None:
        warnings.append("item_count not found in existing result metadata")
    if interaction_count is None:
        warnings.append("interaction_count not found in existing result metadata")

    return {
        "dataset": dataset,
        "model": model,
        "experiment_type": experiment_type,
        "user_count": int(user_count) if user_count is not None else None,
        "item_count": int(item_count) if item_count is not None else None,
        "interaction_count": int(interaction_count) if interaction_count is not None else None,
        "sparsity": sparsity,
        "source_result_dir": bundle.source_dir_string(),
        "note": "Dataset profile is assembled from existing result metadata; missing fields are left null.",
        "warnings": warnings,
    }


def extract_metrics_summary(bundle: ExperimentBundle) -> Dict[str, Any]:
    final_sources = [
        bundle.summary.get("final_eval", {}),
        bundle.result.get("final_eval", {}),
        bundle.summary.get("metrics", {}),
        bundle.result.get("metrics", {}),
    ]
    best_sources = [
        bundle.summary.get("best_summary", {}),
        bundle.result.get("best_summary", {}),
        bundle.summary.get("metadata", {}).get("best_test_result", {}),
        bundle.result.get("metadata", {}).get("best_test_result", {}),
    ]
    tail_sources = [
        bundle.summary.get("tail_mean_summary", {}),
        bundle.result.get("tail_mean_summary", {}),
        bundle.summary.get("metadata", {}).get("tail_mean_summary", {}),
        bundle.result.get("metadata", {}).get("tail_mean_summary", {}),
    ]

    final_recall = read_metric_from_sources(final_sources, "recall", "final")
    final_ndcg = read_metric_from_sources(final_sources, "ndcg", "final")
    best_recall = first_non_none(
        [
            read_metric_from_csv(bundle.csv_rows, "recall", "best"),
            read_metric_from_sources(best_sources, "recall", "best"),
        ]
    )
    best_ndcg = first_non_none(
        [
            read_metric_from_csv(bundle.csv_rows, "ndcg", "best"),
            read_metric_from_sources(best_sources, "ndcg", "best"),
        ]
    )
    tail_recall = first_non_none(
        [
            read_metric_from_csv(bundle.csv_rows, "recall", "tail"),
            read_metric_from_sources(tail_sources, "recall", "tail"),
        ]
    )
    tail_ndcg = first_non_none(
        [
            read_metric_from_csv(bundle.csv_rows, "ndcg", "tail"),
            read_metric_from_sources(tail_sources, "ndcg", "tail"),
        ]
    )
    csv_recall = read_metric_from_csv(bundle.csv_rows, "recall", "csv")
    csv_ndcg = read_metric_from_csv(bundle.csv_rows, "ndcg", "csv")

    source_pairs = [
        ("final", final_recall, final_ndcg),
        ("tail_mean", tail_recall, tail_ndcg),
        ("best", best_recall, best_ndcg),
        ("csv", csv_recall, csv_ndcg),
    ]
    metric_source = None
    recall50 = None
    ndcg50 = None
    for source_name, recall_value, ndcg_value in source_pairs:
        if recall_value is not None or ndcg_value is not None:
            metric_source = source_name
            recall50 = recall_value
            ndcg50 = ndcg_value
            break

    warnings: List[str] = []
    all_sources = bundle.sources + final_sources + best_sources + tail_sources
    if recall50 is None and has_metric20(all_sources, bundle.csv_rows, "recall"):
        warnings.append("Recall@50 not found; available metric is Recall@20")
    if ndcg50 is None and has_metric20(all_sources, bundle.csv_rows, "ndcg"):
        warnings.append("NDCG@50 not found; available metric is NDCG@20")
    if recall50 is None and ndcg50 is None:
        warnings.append("Recall@50/NDCG@50 not found in result JSON or round CSV")

    return {
        "recall50": recall50,
        "ndcg50": ndcg50,
        "best_recall50": best_recall,
        "best_ndcg50": best_ndcg,
        "tail_mean_recall50": tail_recall,
        "tail_mean_ndcg50": tail_ndcg,
        "metric_source": metric_source,
        "note": "Recall@50/NDCG@50 are never inferred from Recall@20/NDCG@20.",
        "warnings": warnings,
    }


def active_module_type(bundle: ExperimentBundle, field_name: str, active_name: str) -> Any:
    return first_non_none(
        [
            bundle.result.get(field_name),
            bundle.summary.get(field_name),
            bundle.result.get("metadata", {}).get(field_name),
            bundle.summary.get("metadata", {}).get(field_name),
            bundle.result.get(active_name),
            bundle.summary.get(active_name),
            bundle.result.get("metadata", {}).get(active_name),
            bundle.summary.get("metadata", {}).get(active_name),
        ]
    )


def extract_attack_defense_summary(
    baseline_bundle: Optional[ExperimentBundle],
    attack_bundle: Optional[ExperimentBundle],
    defense_bundle: Optional[ExperimentBundle],
) -> Dict[str, Any]:
    warnings: List[str] = []
    if not (baseline_bundle and attack_bundle and defense_bundle):
        return {
            "baseline": None,
            "attack": None,
            "defense": None,
            "recall_drop": None,
            "ndcg_drop": None,
            "recall_recovery_rate": None,
            "ndcg_recovery_rate": None,
            "attack_type": None,
            "defense_type": None,
            "note": "baseline/attack/defense dirs were not all provided; comparison artifact is structural only.",
            "warnings": ["baseline-dir, attack-dir, and defense-dir are required for comparison metrics"],
        }

    baseline_metrics = extract_metrics_summary(baseline_bundle)
    attack_metrics = extract_metrics_summary(attack_bundle)
    defense_metrics = extract_metrics_summary(defense_bundle)

    def pair(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"recall50": metrics.get("recall50"), "ndcg50": metrics.get("ndcg50")}

    baseline = pair(baseline_metrics)
    attack = pair(attack_metrics)
    defense = pair(defense_metrics)

    def drop(base_value: Optional[float], attack_value: Optional[float]) -> Optional[float]:
        if base_value is None or attack_value is None:
            return None
        return base_value - attack_value

    def recovery(
        base_value: Optional[float],
        attack_value: Optional[float],
        defense_value: Optional[float],
    ) -> Optional[float]:
        if base_value is None or attack_value is None or defense_value is None:
            return None
        denominator = base_value - attack_value
        if abs(denominator) <= 1e-12:
            return None
        return (defense_value - attack_value) / denominator

    recall_drop = drop(baseline["recall50"], attack["recall50"])
    ndcg_drop = drop(baseline["ndcg50"], attack["ndcg50"])
    recall_recovery_rate = recovery(
        baseline["recall50"], attack["recall50"], defense["recall50"]
    )
    ndcg_recovery_rate = recovery(
        baseline["ndcg50"], attack["ndcg50"], defense["ndcg50"]
    )
    for label, metrics in (
        ("baseline", baseline_metrics),
        ("attack", attack_metrics),
        ("defense", defense_metrics),
    ):
        warnings.extend("{}: {}".format(label, warning) for warning in metrics.get("warnings", []))

    return {
        "baseline": baseline,
        "attack": attack,
        "defense": defense,
        "recall_drop": recall_drop,
        "ndcg_drop": ndcg_drop,
        "recall_recovery_rate": recall_recovery_rate,
        "ndcg_recovery_rate": ndcg_recovery_rate,
        "attack_type": active_module_type(attack_bundle, "attack_type", "active_attacks"),
        "defense_type": active_module_type(defense_bundle, "defense_type", "active_defenses"),
        "note": "Recovery is computed as (defense - attack) / (baseline - attack) when all Recall@50/NDCG@50 values are available.",
        "warnings": warnings,
    }


def find_topk_file(result_dir: Optional[Path]) -> Optional[Path]:
    if result_dir is None or not result_dir.exists():
        return None
    candidates: List[Path] = []
    direct_dir = result_dir / "recommend_topk"
    if direct_dir.exists():
        candidates.extend(direct_dir.rglob("*.csv"))
    for path in result_dir.rglob("*.csv"):
        lower_path = str(path).lower()
        if "recommend_topk" in lower_path or "top" in path.name.lower():
            candidates.append(path)
    unique_candidates = list(dict.fromkeys(candidates))
    return newest_file(unique_candidates)


def read_recommendations(bundle: Optional[ExperimentBundle]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if bundle is None:
        return [], ["result directory not provided"]
    topk_path = find_topk_file(bundle.result_dir)
    if topk_path is None:
        return [], ["TopK recommendation file not found under {}".format(bundle.source_dir_string())]

    rows = read_csv_rows(topk_path, delimiter="\t")
    if rows and len(rows[0].keys()) <= 1:
        rows = read_csv_rows(topk_path, delimiter=",")
    if not rows:
        return [], ["TopK recommendation file could not be parsed: {}".format(topk_path)]

    row = rows[0]
    top_columns = sorted(
        [key for key in row.keys() if normalize_key(key).startswith("top")],
        key=lambda key: int(re.sub(r"\D", "", key) or "0"),
    )
    recommendations: List[Dict[str, Any]] = []
    for rank, column in enumerate(top_columns, start=1):
        item_id = row.get(column)
        if item_id is None or item_id == "":
            continue
        recommendations.append(
            {
                "rank": rank,
                "item_id": str(item_id),
                "score": None,
                "title": None,
                "reason": None,
                "status": "unknown",
            }
        )
    return recommendations, []


def apply_recommendation_statuses(
    baseline: List[Dict[str, Any]],
    attack: List[Dict[str, Any]],
    defense: List[Dict[str, Any]],
    comparison_mode: bool,
) -> None:
    if not comparison_mode:
        return
    baseline_items = {item["item_id"] for item in baseline}
    attack_items = {item["item_id"] for item in attack}
    defense_items = {item["item_id"] for item in defense}

    for item in baseline:
        item["status"] = "stable" if item["item_id"] in attack_items and item["item_id"] in defense_items else "unknown"
    for item in attack:
        item["status"] = "stable" if item["item_id"] in baseline_items else "injected"
    for item in defense:
        if item["item_id"] in baseline_items and item["item_id"] not in attack_items:
            item["status"] = "recovered"
        elif item["item_id"] in baseline_items:
            item["status"] = "stable"
        else:
            item["status"] = "shifted"


def role_for_single_result(bundle: ExperimentBundle) -> str:
    active_attacks = active_module_type(bundle, "attack_type", "active_attacks")
    active_defenses = active_module_type(bundle, "defense_type", "active_defenses")
    has_attacks = bool(active_attacks)
    has_defenses = bool(active_defenses)
    if has_attacks and has_defenses:
        return "defense"
    if has_attacks:
        return "attack"
    return "baseline"


def extract_recommendation_comparison(
    result_bundle: Optional[ExperimentBundle],
    baseline_bundle: Optional[ExperimentBundle],
    attack_bundle: Optional[ExperimentBundle],
    defense_bundle: Optional[ExperimentBundle],
) -> Dict[str, Any]:
    warnings: List[str] = []
    comparison_mode = bool(baseline_bundle and attack_bundle and defense_bundle)
    if comparison_mode:
        baseline, baseline_warnings = read_recommendations(baseline_bundle)
        attack, attack_warnings = read_recommendations(attack_bundle)
        defense, defense_warnings = read_recommendations(defense_bundle)
        warnings.extend("baseline: {}".format(warning) for warning in baseline_warnings)
        warnings.extend("attack: {}".format(warning) for warning in attack_warnings)
        warnings.extend("defense: {}".format(warning) for warning in defense_warnings)
    else:
        baseline, attack, defense = [], [], []
        if result_bundle is None:
            warnings.append("result-dir not provided; no recommendation source available")
        else:
            recommendations, read_warnings = read_recommendations(result_bundle)
            warnings.extend(read_warnings)
            role = role_for_single_result(result_bundle)
            if role == "attack":
                attack = recommendations
            elif role == "defense":
                defense = recommendations
            else:
                baseline = recommendations

    apply_recommendation_statuses(baseline, attack, defense, comparison_mode)
    return {
        "baseline_recommendations": baseline,
        "attacked_recommendations": attack,
        "defended_recommendations": defense,
        "note": "TopK files contain item ids only; score/title/reason are null unless future exporters add them.",
        "warnings": warnings,
    }


def sum_numeric_values(values: Sequence[Any]) -> Optional[float]:
    numeric_values = [to_float(value) for value in values]
    numeric_values = [value for value in numeric_values if value is not None]
    if not numeric_values:
        return None
    return float(sum(numeric_values))


def first_list(values: Sequence[Any]) -> List[Any]:
    for value in values:
        if isinstance(value, list):
            return value
    return []


def extract_defense_trace(bundle: Optional[ExperimentBundle]) -> Dict[str, Any]:
    if bundle is None:
        return {
            "defense_type": None,
            "aggregation_rule": None,
            "clipped_client_count": None,
            "filtered_client_count": None,
            "trimmed_updates": None,
            "selected_indices": [],
            "rejected_client_count": None,
            "note": "No result directory provided.",
            "warnings": ["defense trace unavailable"],
        }

    metadata_sources = [
        bundle.result.get("metadata", {}),
        bundle.summary.get("metadata", {}),
        bundle.result,
        bundle.summary,
    ]
    defense_sources: List[Any] = []
    for source in metadata_sources:
        if isinstance(source, dict):
            for key in ("defense_summaries", "defense_metrics", "defense_outputs"):
                value = source.get(key)
                if value:
                    defense_sources.append(value)
    for round_metric in bundle.result.get("round_metrics", []) if isinstance(bundle.result, dict) else []:
        extra = round_metric.get("extra", {}) if isinstance(round_metric, dict) else {}
        for key in ("defense_metrics", "defense_outputs"):
            if isinstance(extra, dict) and extra.get(key):
                defense_sources.append(extra[key])

    warnings: List[str] = []
    if not defense_sources:
        warnings.append("metadata.defense_summaries or round defense metrics not found")
        defense_sources = metadata_sources

    defense_type = active_module_type(bundle, "defense_type", "active_defenses")
    aggregation_rule = first_non_none(
        find_nested_value(source, ["aggregation_rule", "defense_rule", "trim_rule", "robust_defense_mode"])
        for source in defense_sources
    )
    clipped_count = sum_numeric_values(
        [value for source in defense_sources for value in collect_nested_values(source, ["clipped_client_count", "total_clipped_clients"])]
    )
    filtered_count = sum_numeric_values(
        [value for source in defense_sources for value in collect_nested_values(source, ["filtered_client_count", "total_filtered_clients"])]
    )
    trimmed_updates = sum_numeric_values(
        [
            value
            for source in defense_sources
            for value in collect_nested_values(
                source,
                [
                    "trimmed_updates",
                    "trimmed_client_count_per_coord",
                    "touched_update_count",
                    "rounds_with_trimmed_mean",
                ],
            )
        ]
    )
    selected_indices = first_list(
        [value for source in defense_sources for value in collect_nested_values(source, ["selected_indices"])]
    )
    rejected_client_count = sum_numeric_values(
        [
            value
            for source in defense_sources
            for value in collect_nested_values(source, ["rejected_client_count", "total_rejected_clients"])
        ]
    )

    return {
        "defense_type": defense_type,
        "aggregation_rule": aggregation_rule,
        "clipped_client_count": int(clipped_count) if clipped_count is not None else None,
        "filtered_client_count": int(filtered_count) if filtered_count is not None else None,
        "trimmed_updates": int(trimmed_updates) if trimmed_updates is not None else None,
        "selected_indices": selected_indices,
        "rejected_client_count": int(rejected_client_count) if rejected_client_count is not None else None,
        "note": "Defense trace is assembled from metadata.defense_summaries and round defense metrics when available.",
        "warnings": warnings,
    }


def not_available_probe(name: str) -> Dict[str, Any]:
    return {
        "status": "not_available",
        "note": "probe module available but not executed in this experiment",
        "probe_type": name,
    }


def overall_risk_level(entries: Sequence[Dict[str, Any]]) -> str:
    levels = [entry.get("risk_level") for entry in entries if isinstance(entry, dict)]
    if "high" in levels:
        return "high"
    if "medium" in levels:
        return "medium"
    if "low" in levels:
        return "low"
    return "not_available"


def extract_privacy_payload(bundle: Optional[ExperimentBundle], include_synthetic: bool) -> Dict[str, Any]:
    warnings: List[str] = []
    sources: List[Any] = []
    if bundle is not None:
        sources.extend([bundle.result, bundle.summary])
        for source in (bundle.result, bundle.summary):
            if isinstance(source, dict):
                metadata = source.get("metadata", {})
                if metadata:
                    sources.append(metadata)
                for key in (
                    "privacy_metric_summaries",
                    "privacy_metric_outputs",
                    "privacy_attack_summaries",
                    "privacy_risk_summary",
                ):
                    if isinstance(metadata, dict) and metadata.get(key):
                        sources.append(metadata[key])
                    if source.get(key):
                        sources.append(source[key])
        for round_metric in bundle.result.get("round_metrics", []) if isinstance(bundle.result, dict) else []:
            extra = round_metric.get("extra", {}) if isinstance(round_metric, dict) else {}
            if isinstance(extra, dict):
                for key in ("privacy_metric_outputs", "privacy_attack_summaries", "privacy_risk_summary"):
                    if extra.get(key):
                        sources.append(extra[key])

    client_update_norm = find_nested_payload(sources, ["client_update_norm", "clientupdatenormmetric"])
    membership = find_nested_payload(
        sources,
        ["membership_inference", "membership_inference_probe"],
    )
    gradient = find_nested_payload(sources, ["gradient_leakage", "gradient_leakage_probe"])

    payload = {
        "client_update_norm": client_update_norm or not_available_probe("client_update_norm"),
        "membership_inference": membership or not_available_probe("membership_inference"),
        "gradient_leakage": gradient or not_available_probe("gradient_leakage"),
        "risk_level": overall_risk_level(
            [
                client_update_norm or {},
                membership or {},
                gradient or {},
            ]
        ),
        "warnings": warnings,
    }
    if client_update_norm is None:
        warnings.append("client_update_norm not found in this experiment")
    if membership is None:
        warnings.append("membership_inference_probe result not found in this experiment")
    if gradient is None:
        warnings.append("gradient_leakage_probe result not found in this experiment")

    if include_synthetic:
        try:
            from privacy_eval.gradient_leakage_probe import (
                run_synthetic_smoke as run_gradient_smoke,
            )
            from privacy_eval.membership_inference_probe import (
                run_synthetic_smoke as run_membership_smoke,
            )

            payload["synthetic_demo"] = {
                "demo_only": True,
                "not_from_real_training": True,
                "membership_inference": run_membership_smoke(),
                "gradient_leakage": run_gradient_smoke(),
            }
        except Exception as exc:
            warnings.append("synthetic privacy smoke failed: {}".format(exc))
            payload["synthetic_demo"] = {
                "demo_only": True,
                "not_from_real_training": True,
                "status": "failed",
                "error": str(exc),
            }
    return payload


def choose_primary_bundle(
    result_bundle: Optional[ExperimentBundle],
    baseline_bundle: Optional[ExperimentBundle],
    attack_bundle: Optional[ExperimentBundle],
    defense_bundle: Optional[ExperimentBundle],
) -> Optional[ExperimentBundle]:
    return first_non_none([result_bundle, defense_bundle, attack_bundle, baseline_bundle])


def export_artifacts(args: argparse.Namespace) -> Dict[str, Path]:
    result_bundle = (
        ExperimentBundle(Path(args.result_dir).resolve(), "result")
        if args.result_dir
        else None
    )
    baseline_bundle = (
        ExperimentBundle(Path(args.baseline_dir).resolve(), "baseline")
        if args.baseline_dir
        else None
    )
    attack_bundle = (
        ExperimentBundle(Path(args.attack_dir).resolve(), "attack")
        if args.attack_dir
        else None
    )
    defense_bundle = (
        ExperimentBundle(Path(args.defense_dir).resolve(), "defense")
        if args.defense_dir
        else None
    )
    primary_bundle = choose_primary_bundle(
        result_bundle, baseline_bundle, attack_bundle, defense_bundle
    )
    if primary_bundle is None:
        raise SystemExit("Provide --result-dir or at least one comparison directory.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Dict[str, Any]] = {
        "dataset_profile": extract_dataset_profile(primary_bundle),
        "metrics_summary": extract_metrics_summary(primary_bundle),
        "attack_defense_summary": extract_attack_defense_summary(
            baseline_bundle, attack_bundle, defense_bundle
        ),
        "recommendation_comparison": extract_recommendation_comparison(
            result_bundle, baseline_bundle, attack_bundle, defense_bundle
        ),
        "defense_trace": extract_defense_trace(defense_bundle or result_bundle or primary_bundle),
        "privacy_risk_summary": extract_privacy_payload(
            result_bundle or defense_bundle or primary_bundle,
            include_synthetic=bool(args.include_synthetic_privacy_smoke),
        ),
    }

    artifact_paths: Dict[str, Path] = {}
    for key, payload in artifacts.items():
        path = output_dir / ARTIFACT_NAMES[key]
        write_json(path, payload)
        artifact_paths[key] = path

    manifest_path = output_dir / ARTIFACT_NAMES["showcase_manifest"]
    source_dirs = {
        "result_dir": str(Path(args.result_dir).resolve()) if args.result_dir else None,
        "baseline_dir": str(Path(args.baseline_dir).resolve()) if args.baseline_dir else None,
        "attack_dir": str(Path(args.attack_dir).resolve()) if args.attack_dir else None,
        "defense_dir": str(Path(args.defense_dir).resolve()) if args.defense_dir else None,
    }
    manifest = {
        "dataset_profile": str(artifact_paths["dataset_profile"]),
        "metrics_summary": str(artifact_paths["metrics_summary"]),
        "attack_defense_summary": str(artifact_paths["attack_defense_summary"]),
        "recommendation_comparison": str(artifact_paths["recommendation_comparison"]),
        "defense_trace": str(artifact_paths["defense_trace"]),
        "privacy_risk_summary": str(artifact_paths["privacy_risk_summary"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dirs": source_dirs,
    }
    write_json(manifest_path, manifest)
    artifact_paths["showcase_manifest"] = manifest_path
    return artifact_paths


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = export_artifacts(args)
    print(
        json.dumps(
            {key: str(path) for key, path in paths.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
