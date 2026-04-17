from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_experiment.py"
CONFIG_ROOT = ROOT / "outputs" / "grid_configs"
LOG_ROOT = ROOT / "outputs" / "grid_logs"

LR_GRID: Sequence[tuple[str, float]] = (
    ("5e-4", 5e-4),
    ("1e-3", 1e-3),
    ("2e-3", 2e-3),
)
L2_GRID: Sequence[tuple[str, float]] = (
    ("1e-4", 1e-4),
    ("1e-3", 1e-3),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal6 FedRAP/KU baseline parameter grid through launch_experiment.py."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Generate configs and validate all grid entries without running training.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag used in generated config/log/summary filenames.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke scripts/launch_experiment.py.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Forward --strict-validation to the launcher.",
    )
    return parser


def make_run_tag() -> str:
    return "baseline_minimal6_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))


def safe_token(value: str) -> str:
    return value.replace("+", "").replace("-", "m").replace(".", "p")


def group_slug(index: int, lr_label: str, l2_label: str) -> str:
    return "g{:02d}_lr{}_l2{}".format(index, safe_token(lr_label), safe_token(l2_label))


def build_unified_config(run_tag: str, index: int, lr_label: str, lr: float, l2_label: str, l2_reg: float) -> Dict[str, Any]:
    comment = "baseline_minimal6_g{:02d}_FedRAP_KU_lr{}_l2{}_{}".format(
        index,
        lr_label,
        l2_label,
        run_tag,
    )
    return {
        "model": "FedRAP",
        "dataset": "KU",
        "scenario": "baseline",
        "type": "BaselineSweep",
        "comment": comment,
        "enabled_attacks": [],
        "enabled_defenses": [],
        "enabled_privacy_metrics": [],
        "malicious_client_config": {
            "enabled": False,
            "mode": "none",
            "ratio": 0.0,
            "client_ids": [],
        },
        "training_params": {
            "epochs": 30,
            "local_epochs": 1,
            "clients_sample_ratio": 1.0,
            "learner": "adam",
            "use_gpu": True,
            "eval_step": 1,
            "collect_round_metrics": True,
            "lr": lr,
            "l2_reg": l2_reg,
        },
        "attack_params": {},
        "defense_params": {},
        "privacy_params": {},
        "extra_config": {},
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def extract_launcher_payload(stdout: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    payload: Optional[Dict[str, Any]] = None
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "ok" in candidate:
            payload = candidate
    return payload


def run_launcher(
    *,
    python_exe: str,
    config_path: Path,
    logs_dir: Path,
    slug: str,
    phase: str,
    strict_validation: bool,
) -> Dict[str, Any]:
    command = [
        python_exe,
        str(LAUNCHER),
        "--config",
        str(config_path),
    ]
    if phase == "validate":
        command.append("--validate-only")
    if strict_validation:
        command.append("--strict-validation")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout_path = logs_dir / "{}.{}.stdout.log".format(slug, phase)
    stderr_path = logs_dir / "{}.{}.stderr.log".format(slug, phase)
    payload_path = logs_dir / "{}.{}.launcher.json".format(slug, phase)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    payload = extract_launcher_payload(completed.stdout)
    write_json(payload_path, payload or {"ok": False, "errors": ["launcher_json_not_found"]})

    experiment = payload.get("experiment", {}) if payload else {}
    warnings = payload.get("warnings", []) if payload else []
    errors = payload.get("errors", []) if payload else ["launcher_json_not_found"]

    return {
        "phase": phase,
        "command": command,
        "return_code": completed.returncode,
        "ok": bool(payload and payload.get("ok") and completed.returncode == 0),
        "warnings": warnings,
        "errors": errors,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "launcher_json": str(payload_path),
        "experiment_id": experiment.get("experiment_id"),
        "summary_path": experiment.get("summary_path"),
        "result_path": experiment.get("result_path"),
        "csv_path": experiment.get("csv_path"),
    }


def build_grid(run_tag: str, config_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    index = 1
    for lr_label, lr in LR_GRID:
        for l2_label, l2_reg in L2_GRID:
            slug = group_slug(index, lr_label, l2_label)
            config = build_unified_config(run_tag, index, lr_label, lr, l2_label, l2_reg)
            config_path = config_dir / "{}.json".format(slug)
            write_json(config_path, config)
            entries.append(
                {
                    "index": index,
                    "slug": slug,
                    "lr": lr,
                    "lr_label": lr_label,
                    "l2_reg": l2_reg,
                    "l2_label": l2_label,
                    "config_path": str(config_path),
                    "comment": config["comment"],
                }
            )
            index += 1
    return entries


def write_summary(run_tag: str, summary: Dict[str, Any]) -> tuple[Path, Path]:
    summary_json = LOG_ROOT / "{}_summary.json".format(run_tag)
    summary_md = LOG_ROOT / "{}_summary.md".format(run_tag)
    write_json(summary_json, summary)

    lines = [
        "# Baseline Minimal6 参数网格汇总",
        "",
        "- run_tag: `{}`".format(run_tag),
        "- model: `FedRAP`",
        "- dataset: `KU`",
        "- scenario: `baseline`",
        "- validate_only: `{}`".format(summary.get("validate_only")),
        "",
        "| 序号 | lr | l2_reg | return_code | experiment_id | summary_path | csv_path |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["results"]:
        run_result = row.get("run") or row.get("validate") or {}
        lines.append(
            "| {index} | {lr_label} | {l2_label} | {return_code} | {experiment_id} | {summary_path} | {csv_path} |".format(
                index=row["index"],
                lr_label=row["lr_label"],
                l2_label=row["l2_label"],
                return_code=run_result.get("return_code", ""),
                experiment_id=run_result.get("experiment_id") or "",
                summary_path=run_result.get("summary_path") or "",
                csv_path=run_result.get("csv_path") or "",
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json, summary_md


def main() -> int:
    args = build_parser().parse_args()
    run_tag = args.run_tag or make_run_tag()
    config_dir = CONFIG_ROOT / run_tag
    logs_dir = LOG_ROOT / run_tag
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    entries = build_grid(run_tag, config_dir)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        validate_result = run_launcher(
            python_exe=args.python,
            config_path=Path(entry["config_path"]),
            logs_dir=logs_dir,
            slug=entry["slug"],
            phase="validate",
            strict_validation=args.strict_validation,
        )
        results.append({**entry, "validate": validate_result})

    validate_passed = all(row["validate"]["ok"] for row in results)

    if validate_passed and not args.validate_only:
        for row in results:
            run_result = run_launcher(
                python_exe=args.python,
                config_path=Path(row["config_path"]),
                logs_dir=logs_dir,
                slug=row["slug"],
                phase="run",
                strict_validation=args.strict_validation,
            )
            row["run"] = run_result

    summary: Dict[str, Any] = {
        "run_tag": run_tag,
        "validate_only": args.validate_only,
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "validate_passed": validate_passed,
        "all_runs_passed": all(row.get("run", {}).get("ok", False) for row in results)
        if not args.validate_only and validate_passed
        else None,
        "results": results,
    }
    summary_json, summary_md = write_summary(run_tag, summary)

    print(
        json.dumps(
            {
                "ok": validate_passed if args.validate_only else bool(summary["all_runs_passed"]),
                "run_tag": run_tag,
                "validate_passed": validate_passed,
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.validate_only:
        return 0 if validate_passed else 1
    return 0 if validate_passed and summary["all_runs_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
