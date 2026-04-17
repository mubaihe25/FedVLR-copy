from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_experiment.py"
CONFIG_ROOT = ROOT / "outputs" / "batch_configs"
LOG_ROOT = ROOT / "outputs" / "batch_logs"
RUN_ROOT = ROOT / "outputs" / "batch_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a batch of unified FedVLR experiments through launch_experiment.py."
    )
    parser.add_argument("--batch", required=True, help="Path to a batch config JSON file.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate generated configs.")
    parser.add_argument("--strict-validation", action="store_true", help="Forward strict validation to launcher.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip successful rows recorded in summary JSON.")
    parser.add_argument("--max-runs", type=int, default=None, help="Only consider the first N expanded rows.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed validation or run.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to invoke launcher.")
    return parser


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sanitize_token(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("\\", "_").replace("/", "_").replace(" ", "_")
    for old, new in {
        ".": "p",
        "-": "m",
        "+": "",
        "=": "-",
        ":": "-",
        ",": "_",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "'": "",
        '"': "",
    }.items():
        text = text.replace(old, new)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text) or "value"


def normalize_sweep_value(raw_value: Any) -> Tuple[Any, str]:
    if isinstance(raw_value, dict) and "value" in raw_value:
        value = raw_value["value"]
        label = str(raw_value.get("label", value))
        return value, label
    return raw_value, str(raw_value)


def set_dotted_path(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("empty sweep path")

    current: Dict[str, Any] = target
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError("path {} conflicts at {}".format(dotted_path, part))
        current = child
    current[parts[-1]] = value


def get_dotted_path(source: Dict[str, Any], dotted_path: str) -> Any:
    current: Any = source
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


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


def resolve_existing_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def successful_files_exist(run_result: Dict[str, Any]) -> bool:
    if not run_result.get("ok"):
        return False
    paths = [
        resolve_existing_path(run_result.get("summary_path")),
        resolve_existing_path(run_result.get("result_path")),
        resolve_existing_path(run_result.get("csv_path")),
    ]
    return all(path is not None and path.exists() for path in paths)


def load_previous_rows(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        payload = load_json(summary_path)
    except Exception:
        return {}

    previous: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("results", []):
        slug = row.get("slug")
        if isinstance(slug, str):
            previous[slug] = row
    return previous


def run_launcher(
    *,
    python_exe: str,
    config_path: Path,
    logs_dir: Path,
    slug: str,
    phase: str,
    strict_validation: bool,
) -> Dict[str, Any]:
    command = [python_exe, str(LAUNCHER), "--config", str(config_path)]
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

    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "{}.{}.stdout.log".format(slug, phase)
    stderr_path = logs_dir / "{}.{}.stderr.log".format(slug, phase)
    payload_path = logs_dir / "{}.{}.launcher.json".format(slug, phase)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    payload = extract_launcher_payload(completed.stdout)
    write_json(payload_path, payload or {"ok": False, "errors": ["launcher_json_not_found"]})

    experiment = payload.get("experiment", {}) if payload else {}
    return {
        "phase": phase,
        "command": command,
        "return_code": completed.returncode,
        "ok": bool(payload and payload.get("ok") and completed.returncode == 0),
        "warnings": payload.get("warnings", []) if payload else [],
        "errors": payload.get("errors", []) if payload else ["launcher_json_not_found"],
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "launcher_json": str(payload_path),
        "experiment_id": experiment.get("experiment_id"),
        "summary_path": experiment.get("summary_path"),
        "result_path": experiment.get("result_path"),
        "csv_path": experiment.get("csv_path"),
    }


def sweep_items(sweep_params: Dict[str, List[Any]]) -> List[Tuple[str, List[Tuple[Any, str]]]]:
    items: List[Tuple[str, List[Tuple[Any, str]]]] = []
    for path, values in sweep_params.items():
        if not isinstance(values, list):
            raise ValueError("sweep path {} must be a list".format(path))
        items.append((path, [normalize_sweep_value(value) for value in values]))
    return items


def expand_batch(batch_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_config = batch_config.get("base_config")
    if not isinstance(base_config, dict):
        raise ValueError("batch config must contain object base_config")

    items = sweep_items(batch_config.get("sweep_params") or {})
    if not items:
        combinations: Iterable[Tuple[Tuple[Any, str], ...]] = [tuple()]
    else:
        combinations = itertools.product(*(values for _, values in items))

    batch_name = str(batch_config["batch_name"])
    rows: List[Dict[str, Any]] = []
    for index, values in enumerate(combinations, start=1):
        config = copy.deepcopy(base_config)
        sweep_values: Dict[str, Any] = {}
        sweep_labels: Dict[str, str] = {}
        label_parts: List[str] = []

        for (path, _), (value, label) in zip(items, values):
            set_dotted_path(config, path, value)
            sweep_values[path] = value
            sweep_labels[path] = label
            label_parts.append("{}={}".format(path, label))

        short_label = "__".join("{}-{}".format(path.split(".")[-1], sanitize_token(label)) for path, label in sweep_labels.items())
        slug = "g{:03d}".format(index) if not short_label else "g{:03d}_{}".format(index, short_label)
        base_comment = str(base_config.get("comment", "batch_experiment"))
        config["comment"] = "{}.{}.{}".format(base_comment, batch_name, slug)

        rows.append(
            {
                "index": index,
                "slug": slug,
                "sweep_values": sweep_values,
                "sweep_labels": sweep_labels,
                "sweep_label": ", ".join(label_parts) if label_parts else "single_run",
                "config": config,
            }
        )
    return rows


def write_generated_configs(rows: List[Dict[str, Any]], config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = config_dir / "{}.json".format(row["slug"])
        write_json(path, row["config"])
        row["generated_config_path"] = str(path)


def write_summary(summary_path: Path, md_path: Path, summary: Dict[str, Any]) -> None:
    write_json(summary_path, summary)
    lines = [
        "# Experiment Batch Summary",
        "",
        "- batch_name: `{}`".format(summary.get("batch_name")),
        "- description: {}".format(summary.get("description") or ""),
        "- validate_only: `{}`".format(summary.get("validate_only")),
        "- selected_rows: `{}`".format(len(summary.get("results", []))),
        "",
        "| # | slug | sweep | return_code | experiment_id | summary_path | csv_path |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary.get("results", []):
        run_result = row.get("run") or row.get("validate") or {}
        lines.append(
            "| {index} | {slug} | {sweep} | {return_code} | {experiment_id} | {summary_path} | {csv_path} |".format(
                index=row.get("index", ""),
                slug=row.get("slug", ""),
                sweep=row.get("sweep_label", ""),
                return_code="skipped" if row.get("skipped") else run_result.get("return_code", ""),
                experiment_id=run_result.get("experiment_id") or "",
                summary_path=run_result.get("summary_path") or "",
                csv_path=run_result.get("csv_path") or "",
            )
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    batch_path = Path(args.batch)
    if not batch_path.is_absolute():
        batch_path = ROOT / batch_path

    batch_config = load_json(batch_path)
    batch_name = sanitize_token(batch_config.get("batch_name", batch_path.stem))
    description = str(batch_config.get("description", ""))
    rows = expand_batch({**batch_config, "batch_name": batch_name})
    if args.max_runs is not None:
        rows = rows[: max(0, args.max_runs)]

    config_dir = CONFIG_ROOT / batch_name
    logs_dir = LOG_ROOT / batch_name
    summary_path = RUN_ROOT / "{}_summary.json".format(batch_name)
    md_path = RUN_ROOT / "{}_summary.md".format(batch_name)
    previous_rows = load_previous_rows(summary_path)
    write_generated_configs(rows, config_dir)

    results: List[Dict[str, Any]] = []
    validation_ok = True

    for row in rows:
        previous = previous_rows.get(row["slug"])
        if args.skip_completed and previous and successful_files_exist(previous.get("run", {})):
            row.update(
                {
                    "generated_config_path": row["generated_config_path"],
                    "validate": previous.get("validate"),
                    "run": previous.get("run"),
                    "skipped": True,
                    "skip_reason": "completed_result_files_exist",
                }
            )
            results.append(row)
            continue

        validate_result = run_launcher(
            python_exe=args.python,
            config_path=Path(row["generated_config_path"]),
            logs_dir=logs_dir,
            slug=row["slug"],
            phase="validate",
            strict_validation=args.strict_validation,
        )
        row["validate"] = validate_result
        results.append(row)
        if not validate_result["ok"]:
            validation_ok = False
            if args.fail_fast:
                break

    should_run = validation_ok and not args.validate_only
    run_ok = True
    if should_run:
        for row in results:
            if row.get("skipped") or row.get("run"):
                continue
            if not row.get("validate", {}).get("ok"):
                continue
            run_result = run_launcher(
                python_exe=args.python,
                config_path=Path(row["generated_config_path"]),
                logs_dir=logs_dir,
                slug=row["slug"],
                phase="run",
                strict_validation=args.strict_validation,
            )
            row["run"] = run_result
            if not run_result["ok"]:
                run_ok = False
                if args.fail_fast:
                    break

    summary = {
        "batch_name": batch_name,
        "description": description,
        "batch_path": str(batch_path),
        "validate_only": args.validate_only,
        "strict_validation": args.strict_validation,
        "skip_completed": args.skip_completed,
        "max_runs": args.max_runs,
        "config_dir": str(config_dir),
        "logs_dir": str(logs_dir),
        "summary_json": str(summary_path),
        "summary_md": str(md_path),
        "validation_ok": validation_ok,
        "run_ok": run_ok if should_run else None,
        "results": results,
    }
    write_summary(summary_path, md_path, summary)

    ok = validation_ok if args.validate_only else validation_ok and run_ok
    print(
        json.dumps(
            {
                "ok": ok,
                "batch_name": batch_name,
                "expanded_count": len(expand_batch({**batch_config, "batch_name": batch_name})),
                "selected_count": len(rows),
                "summary_json": str(summary_path),
                "summary_md": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
