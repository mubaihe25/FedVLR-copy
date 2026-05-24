"""Build a lightweight defense matrix summary from smoke configs/results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a defense_matrix_summary.json artifact.")
    parser.add_argument("--case-config", action="append", default=[])
    parser.add_argument("--result-dir", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    return parser


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def nested_value(payload: Dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def latest_json(directory: Path, suffix: str) -> Optional[Path]:
    files = [path for path in directory.rglob("*{}".format(suffix)) if path.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def config_case(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    defense_params = payload.get("defense_params", {}) if isinstance(payload.get("defense_params"), dict) else {}
    robust = defense_params.get("robust_defense", {}) if isinstance(defense_params.get("robust_defense"), dict) else {}
    attack_params = payload.get("attack_params", {}) if isinstance(payload.get("attack_params"), dict) else {}
    target_injection = (
        attack_params.get("target_interaction_injection", {})
        if isinstance(attack_params.get("target_interaction_injection"), dict)
        else {}
    )
    return {
        "source_type": "config",
        "config_path": str(path),
        "case_name": payload.get("comment") or path.stem,
        "model": payload.get("model"),
        "dataset": payload.get("dataset"),
        "scenario": payload.get("scenario"),
        "enabled_attacks": payload.get("enabled_attacks", []),
        "enabled_defenses": payload.get("enabled_defenses", []),
        "defense_mode": robust.get("robust_defense_mode")
        or (payload.get("enabled_defenses", [None])[0] if payload.get("enabled_defenses") else None),
        "robust_trim_ratio": robust.get("robust_trim_ratio"),
        "target_item_ids": target_injection.get("target_item_ids"),
        "malicious_client_ratio": nested_value(payload, ["malicious_client_config", "ratio"]),
        "epochs": nested_value(payload, ["training_params", "epochs"]),
        "status": "configured",
        "warnings": ["config-only case; no defense outcome metrics are implied"],
    }


def result_case(path: Path) -> Dict[str, Any]:
    summary_path = latest_json(path, ".experiment_summary.json")
    result_path = latest_json(path, ".experiment_result.json")
    summary = read_json(summary_path) if summary_path else {}
    result = read_json(result_path) if result_path else {}
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    defense_summaries = metadata.get("defense_summaries", {})
    return {
        "source_type": "result",
        "result_dir": str(path),
        "summary_path": str(summary_path) if summary_path else None,
        "result_path": str(result_path) if result_path else None,
        "experiment_id": summary.get("experiment_id") or result.get("experiment_id"),
        "model": summary.get("model") or result.get("model"),
        "dataset": summary.get("dataset") or result.get("dataset"),
        "active_attacks": summary.get("active_attacks") or metadata.get("active_attacks"),
        "active_defenses": summary.get("active_defenses") or metadata.get("active_defenses"),
        "defense_summaries": defense_summaries if isinstance(defense_summaries, dict) else {},
        "status": "available" if summary_path or result_path else "not_available",
        "warnings": [] if summary_path or result_path else ["result dir has no experiment JSON outputs"],
    }


def build_summary(configs: Sequence[str], result_dirs: Sequence[str]) -> Dict[str, Any]:
    warnings: List[str] = []
    cases: List[Dict[str, Any]] = []
    for raw_path in configs:
        path = Path(raw_path)
        if path.exists():
            cases.append(config_case(path))
        else:
            warnings.append("case config missing: {}".format(path))
    for raw_path in result_dirs:
        path = Path(raw_path)
        if path.exists():
            cases.append(result_case(path))
        else:
            warnings.append("result dir missing: {}".format(path))
    return {
        "metric_type": "defense_matrix",
        "summary_type": "defense_matrix_summary",
        "case_count": len(cases),
        "cases": cases,
        "matrix_status": "available" if cases else "not_available",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
        "note": (
            "Defense matrix summary is a configuration/result index for smoke comparison. "
            "It does not imply defenses were fully benchmarked unless result metrics are present."
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = build_summary(args.case_config, args.result_dir)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
