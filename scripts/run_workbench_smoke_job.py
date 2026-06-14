from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "workbench_jobs"
SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_workbench_smoke_config as generator  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def safe_job_dir(job_id: str, output_dir: Path) -> Path:
    if not SAFE_JOB_ID_RE.fullmatch(job_id):
        raise ValueError("invalid_job_id")
    output_dir = output_dir.resolve()
    default_root = DEFAULT_OUTPUT_ROOT.resolve()
    if output_dir.name == job_id:
        job_dir = output_dir
    else:
        job_dir = (output_dir / job_id).resolve()
    if default_root not in [job_dir, *job_dir.parents]:
        raise ValueError("job_path_outside_workbench_root")
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def append_log(job_dir: Path, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    with (job_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def update_status(job_dir: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    current = read_json(job_dir / "status.json", {}) or {}
    current.update(patch)
    current["updated_at"] = utc_now()
    write_json(job_dir / "status.json", current)
    return current


def normalize_input(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    if "normalized_config" in payload and isinstance(payload["normalized_config"], dict):
        return payload["normalized_config"], list(payload.get("warnings", [])), list(payload.get("errors", []))
    if "training" in payload and "unified_experiment_config" in payload:
        return payload, [], []
    normalized, warnings, errors = generator.normalize_workbench_payload(payload)
    return normalized, warnings, errors


def build_full_train_launcher_config(job_id: str, normalized: Dict[str, Any]) -> Dict[str, Any]:
    unified = dict(normalized.get("unified_experiment_config") or {})
    training = dict(unified.get("training_params") or normalized.get("training") or {})
    client_ratio = training.get("client_sampling_ratio", training.get("clients_sample_ratio"))
    if client_ratio is not None:
        training["client_sampling_ratio"] = client_ratio
        training["clients_sample_ratio"] = client_ratio
    training["top_k"] = 50
    training["topk"] = [50]
    training["recommendation_topk"] = 50
    unified["training_params"] = training
    unified["type"] = "WorkbenchFullTrain"
    unified["comment"] = f"workbench_full_train_{job_id}"
    extra_config = dict(unified.get("extra_config") or {})
    extra_config.update(
        {
            "workbench_job_id": job_id,
            "workbench_source": "full_train",
            "workbench_direction": normalized.get("direction"),
        }
    )
    unified["extra_config"] = extra_config
    unified["workbench_runtime_policy"] = {
        "no_arbitrary_command": True,
        "no_output_deletion": True,
    }
    return unified




def parse_first_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_subprocess_error(output: str, return_code: int) -> Tuple[str, str]:
    detail = output.strip()
    summary = f"训练子进程退出码 {return_code}"
    traceback_lines = [line.strip() for line in detail.splitlines() if line.strip()]
    exception_line = next(
        (
            line
            for line in reversed(traceback_lines)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):", line)
        ),
        None,
    )
    if exception_line:
        summary = f"{summary}：{exception_line}"
    return summary, detail


def read_csv_summary_metrics(csv_path: Path | None) -> Dict[str, Any]:
    if csv_path is None or not csv_path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    preferred = next((row for row in rows if row.get("row_type") == "best_summary"), None)
    if preferred is None:
        preferred = rows[-1] if rows else {}
    return {
        "recall_at_50": preferred.get("best_recall50") or preferred.get("test_recall50") or preferred.get("tail_test_recall50"),
        "ndcg_at_50": preferred.get("best_ndcg50") or preferred.get("test_ndcg50") or preferred.get("tail_test_ndcg50"),
        "round_id": preferred.get("best_round_for_recall50") or preferred.get("round_id"),
        "row_type": preferred.get("row_type"),
    }


def run_full_train(
    job_id: str,
    job_dir: Path,
    launcher_config: Dict[str, Any],
    *,
    config_name: str = "launcher_config.json",
    phase: str = "training",
    progress: int = 42,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], str | None]:
    launcher_path = ROOT / "scripts" / "launch_experiment.py"
    if not launcher_path.exists():
        raise FileNotFoundError("launch_experiment.py")
    config_path = job_dir / config_name
    write_json(config_path, launcher_config)
    command = [sys.executable, str(launcher_path), "--config", str(config_path)]
    command_display = subprocess.list2cmdline(command)
    runtime_metadata = {
        "subprocess_command": command_display,
        "python_path": str(Path(sys.executable).resolve()),
        "cwd": str(ROOT),
        "launcher_path": str(launcher_path),
        "return_code": None,
        "phase": phase,
    }

    update_status(
        job_dir,
        {
            "status": "running",
            "stage": phase,
            "progress": progress,
            "source": "full_train",
            "message": "正在执行 FedVLR 真实训练入口。",
            **runtime_metadata,
        },
    )
    append_log(job_dir, f"[{phase}] command={command_display}")
    append_log(job_dir, f"[{phase}] python={runtime_metadata['python_path']}")
    append_log(job_dir, f"[{phase}] cwd={runtime_metadata['cwd']}")
    started = time.time()
    process = subprocess.Popen(  # noqa: S603 - fixed Python executable and whitelisted launcher path.
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    update_status(job_dir, {"pid": process.pid, "subprocess_pid": process.pid})
    append_log(job_dir, f"[{phase}] pid={process.pid}")
    output_lines: List[str] = []
    if process.stdout is not None:
        with (job_dir / "run.log").open("a", encoding="utf-8") as handle:
            for line in process.stdout:
                output_lines.append(line)
                handle.write(line)
                handle.flush()
        process.stdout.close()
    return_code = process.wait()
    elapsed = round(time.time() - started, 3)
    process_output = "".join(output_lines)
    update_status(job_dir, {"return_code": return_code, "subprocess_return_code": return_code})
    append_log(job_dir, f"[{phase}] Launcher exited with code={return_code}; elapsed={elapsed}s.")
    status_payload = read_json(job_dir / "status.json", {}) or {}
    subprocess_runs = list(status_payload.get("subprocess_runs") or [])
    subprocess_runs.append(
        {
            **runtime_metadata,
            "pid": process.pid,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
        }
    )
    update_status(job_dir, {"subprocess_runs": subprocess_runs})

    launch_payload = parse_first_json_object(process_output)
    if return_code != 0 or not launch_payload.get("ok"):
        error_summary, error_detail = extract_subprocess_error(process_output, return_code)
        if launch_payload.get("errors"):
            error_summary = f"{error_summary}：{launch_payload['errors']}"
        update_status(
            job_dir,
            {
                "failure_stage": phase,
                "error_summary": error_summary,
                "error_detail": error_detail,
            },
        )
        return "failed", {}, {"launch_payload": launch_payload, **runtime_metadata, "return_code": return_code}, error_summary

    experiment = launch_payload.get("experiment") if isinstance(launch_payload.get("experiment"), dict) else {}
    csv_path = ROOT / str(experiment.get("csv_path")) if experiment.get("csv_path") else None
    summary_path = ROOT / str(experiment.get("summary_path")) if experiment.get("summary_path") else None
    csv_metrics = read_csv_summary_metrics(csv_path)
    summary = read_json(summary_path, {}) if summary_path else {}
    training_config = summary.get("training_config") if isinstance(summary, dict) else {}
    final_eval = summary.get("final_eval") if isinstance(summary, dict) else {}
    metrics = {
        "recall_at_50": _number_or_raw(csv_metrics.get("recall_at_50")),
        "ndcg_at_50": _number_or_raw(csv_metrics.get("ndcg_at_50")),
        "direction": (launcher_config.get("extra_config") or {}).get("workbench_direction") or "aggregation_defense",
        "model": experiment.get("model") or launcher_config.get("model"),
        "dataset": experiment.get("dataset") or launcher_config.get("dataset"),
        "source": "full_train",
        "active_attacks": experiment.get("active_attacks", []),
        "active_defenses": experiment.get("active_defenses", []),
        "loss": final_eval.get("loss") if isinstance(final_eval, dict) else None,
        "epochs": training_config.get("epochs") if isinstance(training_config, dict) else None,
        "local_epochs": training_config.get("local_epochs") if isinstance(training_config, dict) else None,
        "client_sampling_ratio": training_config.get("clients_sample_ratio") if isinstance(training_config, dict) else None,
    }
    pointer = {
        "launch_payload": launch_payload,
        "result_dir": repo_relative(ROOT / str(experiment.get("result_dir"))) if experiment.get("result_dir") else None,
        "summary_path": repo_relative(summary_path),
        "result_path": repo_relative(ROOT / str(experiment.get("result_path"))) if experiment.get("result_path") else None,
        "csv_path": repo_relative(csv_path),
        "recommend_topk_dir": repo_relative(ROOT / str(experiment.get("recommend_topk_dir"))) if experiment.get("recommend_topk_dir") else None,
        **runtime_metadata,
        "return_code": return_code,
    }
    target_rank_source = csv_path.parent / "target_rank_summary.json" if csv_path else None
    if target_rank_source and target_rank_source.exists():
        target_rank_copy = job_dir / f"{phase}_target_rank_summary.json"
        shutil.copyfile(target_rank_source, target_rank_copy)
        pointer["target_rank_summary"] = repo_relative(target_rank_copy)
    warnings = list(launch_payload.get("warnings", []))
    status = "completed"
    partial_reason = None
    if warnings:
        partial_reason = "full training completed with launcher warnings"
    return status, metrics, pointer, partial_reason


def build_recommendation_baseline_config(job_id: str, attack_config: Dict[str, Any]) -> Dict[str, Any]:
    baseline = copy.deepcopy(attack_config)
    baseline["scenario"] = "baseline"
    baseline["comment"] = f"workbench_full_train_{job_id}_baseline"
    baseline["enabled_attacks"] = []
    baseline["enabled_defenses"] = []
    baseline["enabled_privacy_metrics"] = []
    baseline["malicious_client_config"] = {
        "enabled": False,
        "mode": "none",
        "ratio": 0.0,
        "client_ids": [],
    }
    target_params = dict((attack_config.get("attack_params") or {}).get("target_interaction_injection") or {})
    attack_params = copy.deepcopy(baseline.get("attack_params") or {})
    baseline_target_params = dict(attack_params.get("target_interaction_injection") or {})
    baseline_target_params["enabled"] = False
    attack_params["target_interaction_injection"] = baseline_target_params
    baseline["attack_params"] = attack_params
    extra_config = dict(baseline.get("extra_config") or {})
    extra_config.update(
        {
            "workbench_phase": "baseline",
            "target_item_ids": list(target_params.get("target_item_ids") or []),
            "target_item_title": target_params.get("target_item_title"),
        }
    )
    baseline["extra_config"] = extra_config
    return baseline


def read_topk_preview(path_value: Any, limit: int = 50) -> Dict[str, Any]:
    if not path_value:
        return {}
    directory = ROOT / str(path_value)
    manifest = read_json(directory / "recommend_topk_manifest.json", {}) or {}
    entries = list(manifest.get("topk_files") or []) if isinstance(manifest, dict) else []
    file_name = entries[0].get("file") if entries and isinstance(entries[0], dict) else None
    if not file_name:
        candidates = sorted(directory.glob("*.csv"))
        file_name = candidates[0].name if candidates else None
    if not file_name:
        return {}
    csv_path = directory / str(file_name)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"), None)
    if not row:
        return {}
    ordered_items = [
        row.get(f"top_{index}")
        for index in range(limit)
        if row.get(f"top_{index}") not in {None, ""}
    ]
    return {
        "user_id": str(row.get("id") or ""),
        "items": ordered_items,
        "count": len(ordered_items),
        "source_file": repo_relative(csv_path),
    }


def build_target_rank_comparison(
    baseline_path_value: Any,
    attack_path_value: Any,
    target_item_ids: List[str],
) -> Dict[str, Any]:
    baseline = read_json(ROOT / str(baseline_path_value), {}) if baseline_path_value else {}
    attack = read_json(ROOT / str(attack_path_value), {}) if attack_path_value else {}
    target_id = str(target_item_ids[0]) if target_item_ids else ""
    baseline_summary = ((baseline.get("target_summaries") or {}).get(target_id) or {}) if isinstance(baseline, dict) else {}
    attack_summary = ((attack.get("target_summaries") or {}).get(target_id) or {}) if isinstance(attack, dict) else {}
    before_rank = _number_or_raw(baseline_summary.get("average_unmasked_rank"))
    after_rank = _number_or_raw(attack_summary.get("average_unmasked_rank"))
    attack_records = list(attack.get("user_target_records") or []) if isinstance(attack, dict) else []
    masked_hits = sum(1 for record in attack_records if record.get("masked_rank") is not None and int(record["masked_rank"]) <= 50)
    evaluated_count = int(attack.get("evaluated_user_count") or len(attack_records) or 0) if isinstance(attack, dict) else 0
    return {
        "target_item_id": target_id or None,
        "target_rank_before": before_rank,
        "target_rank_after": after_rank,
        "rank_gain": (float(before_rank) - float(after_rank)) if isinstance(before_rank, (int, float)) and isinstance(after_rank, (int, float)) else None,
        "masked_top50_hit": masked_hits > 0,
        "masked_top50_hit_count": masked_hits,
        "masked_top50_hit_rate": (masked_hits / evaluated_count) if evaluated_count else None,
        "evaluated_user_count": evaluated_count,
    }


def _number_or_raw(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def run_job(job_id: str, payload: Dict[str, Any], job_dir: Path) -> int:
    existing_status = read_json(job_dir / "status.json", {}) or {}
    created_at = existing_status.get("created_at") or utc_now()
    started_at = existing_status.get("started_at") or created_at
    (job_dir / "run.log").write_text("", encoding="utf-8")
    update_status(
        job_dir,
        {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": None,
            "error_message": None,
            "source": None,
            "result_dir": None,
            "artifact_dir": None,
        },
    )
    append_log(job_dir, f"Workbench job {job_id} queued.")

    update_status(job_dir, {"status": "running", "stage": "preparing_config", "progress": 12, "started_at": started_at})
    append_log(job_dir, "[prepare] Validating full-training payload.")

    normalized, warnings, errors = normalize_input(payload)
    if errors:
        raise ValueError(";".join(str(item) for item in errors))
    if str(normalized.get("execution_mode")) != "full_train":
        raise ValueError("full_train_required")
    robust_aggregators = list(normalized.get("robust_aggregators") or [])
    if len(robust_aggregators) > 1:
        raise ValueError("multiple_robust_aggregators_not_supported")
    if str(normalized.get("direction")) == "aggregation_defense":
        base_attack = str((normalized.get("defense") or {}).get("base_attack") or "")
        if base_attack not in {"none", "malicious_update"}:
            raise ValueError(f"aggregation_defense_invalid_base_attack:{base_attack}")
    normalized_training = dict(normalized.get("training") or {})
    normalized_training["top_k"] = 50
    normalized["training"] = normalized_training
    write_json(job_dir / "config.json", normalized)
    direction = str(normalized.get("direction") or "recommendation_manipulation")
    launcher_config = build_full_train_launcher_config(job_id, normalized)
    write_json(job_dir / "launcher_config.json", launcher_config)
    append_log(job_dir, "[prepare] Wrote launcher_config.json with submitted training parameters.")

    update_status(
        job_dir,
        {
            "stage": "preparing_config",
            "progress": 24,
            "direction": direction,
            "scenario_id": normalized.get("scenario_id"),
            "source": "full_train",
            "warnings": warnings,
            "errors": [],
        },
    )
    launcher_warnings: List[str] = []
    if direction == "recommendation_manipulation":
        baseline_config = build_recommendation_baseline_config(job_id, launcher_config)
        write_json(job_dir / "baseline_launcher_config.json", baseline_config)
        append_log(job_dir, "[prepare] Wrote baseline_launcher_config.json for real baseline training.")
        baseline_status, baseline_metrics, baseline_pointer, baseline_reason = run_full_train(
            job_id,
            job_dir,
            baseline_config,
            config_name="baseline_launcher_config.json",
            phase="baseline_training",
            progress=32,
        )
        baseline_launch_payload = baseline_pointer.get("launch_payload") if isinstance(baseline_pointer, dict) else {}
        if isinstance(baseline_launch_payload, dict):
            launcher_warnings.extend(str(item) for item in baseline_launch_payload.get("warnings", []))
        if baseline_status == "failed":
            final_status = "failed"
            metrics = baseline_metrics
            pointer_extra = {
                "baseline": baseline_pointer,
                "attack": None,
                "result_dir": baseline_pointer.get("result_dir"),
                "launch_payload": baseline_launch_payload,
            }
            partial_reason = baseline_reason
        else:
            attack_extra = dict(launcher_config.get("extra_config") or {})
            attack_extra["workbench_phase"] = "attack"
            launcher_config["extra_config"] = attack_extra
            write_json(job_dir / "launcher_config.json", launcher_config)
            append_log(job_dir, "[prepare] Starting real attack training after baseline completed.")
            attack_status, attack_metrics, attack_pointer, attack_reason = run_full_train(
                job_id,
                job_dir,
                launcher_config,
                config_name="launcher_config.json",
                phase="attack_training",
                progress=58,
            )
            attack_launch_payload = attack_pointer.get("launch_payload") if isinstance(attack_pointer, dict) else {}
            if isinstance(attack_launch_payload, dict):
                launcher_warnings.extend(str(item) for item in attack_launch_payload.get("warnings", []))
            target_params = dict((launcher_config.get("attack_params") or {}).get("target_interaction_injection") or {})
            target_item_ids = [str(item) for item in target_params.get("target_item_ids") or []]
            target_comparison = build_target_rank_comparison(
                baseline_pointer.get("target_rank_summary"),
                attack_pointer.get("target_rank_summary"),
                target_item_ids,
            )
            baseline_top50 = read_topk_preview(baseline_pointer.get("recommend_topk_dir"))
            attack_top50 = read_topk_preview(attack_pointer.get("recommend_topk_dir"))
            final_status = attack_status
            metrics = {
                **attack_metrics,
                "baseline_loss": baseline_metrics.get("loss"),
                "baseline_recall_at_50": baseline_metrics.get("recall_at_50"),
                "baseline_ndcg_at_50": baseline_metrics.get("ndcg_at_50"),
                "attack_loss": attack_metrics.get("loss"),
                "attack_recall_at_50": attack_metrics.get("recall_at_50"),
                "attack_ndcg_at_50": attack_metrics.get("ndcg_at_50"),
                **target_comparison,
                "baseline_top50": baseline_top50,
                "attack_top50": attack_top50,
            }
            pointer_extra = {
                **attack_pointer,
                "baseline": baseline_pointer,
                "attack": attack_pointer,
                "baseline_recommend_topk_dir": baseline_pointer.get("recommend_topk_dir"),
                "attack_recommend_topk_dir": attack_pointer.get("recommend_topk_dir"),
                "baseline_target_rank_summary": baseline_pointer.get("target_rank_summary"),
                "attack_target_rank_summary": attack_pointer.get("target_rank_summary"),
                "target_rank_comparison": target_comparison,
                "recommendation_previews": {
                    "baseline": baseline_top50,
                    "attack": attack_top50,
                },
            }
            partial_reason = attack_reason or baseline_reason
    else:
        final_status, metrics, pointer_extra, partial_reason = run_full_train(job_id, job_dir, launcher_config)
        launch_payload = pointer_extra.get("launch_payload") if isinstance(pointer_extra, dict) else {}
        if isinstance(launch_payload, dict):
            launcher_warnings.extend(str(item) for item in launch_payload.get("warnings", []))
    launcher_warnings = list(dict.fromkeys(launcher_warnings))
    all_warnings = [*warnings, *launcher_warnings]

    update_status(job_dir, {"stage": "exporting_artifacts", "progress": 86, "warnings": all_warnings})
    append_log(job_dir, "[export] Writing full-training metrics_summary and result_pointer.")
    result_dir = pointer_extra.get("result_dir") if isinstance(pointer_extra, dict) else None

    result_pointer = {
        "job_id": job_id,
        "status": final_status,
        "source": "full_train",
        "config": "config.json",
        "launcher_config": "launcher_config.json",
        "status_file": "status.json",
        "metrics_summary": "metrics_summary.json",
        "log": "run.log",
        "result_dir": result_dir,
        "artifact_dir": None,
        "showcase_scenario_id": normalized.get("scenario_id"),
        **(pointer_extra if isinstance(pointer_extra, dict) else {}),
    }
    metrics_summary = {
        "job_id": job_id,
        "status": final_status,
        "source": "full_train",
        "direction": direction,
        "scenario_id": normalized.get("scenario_id"),
        "model": metrics.get("model"),
        "dataset": metrics.get("dataset"),
        "metrics": metrics,
        "warnings": all_warnings,
        "message": "真实全量训练任务已执行，训练参数来自工作台提交配置。",
        "partial_reason": partial_reason,
    }
    write_json(job_dir / "result_pointer.json", result_pointer)
    write_json(job_dir / "metrics_summary.json", metrics_summary)

    finished_at = utc_now()
    update_status(
        job_dir,
        {
            "status": final_status,
            "stage": final_status,
            "progress": 100,
            "finished_at": finished_at,
            "error_message": partial_reason if final_status in {"partial", "failed"} else None,
            "message": "真实训练已完成。" if final_status == "completed" else "真实训练执行失败。",
            "result_dir": result_dir,
            "artifact_dir": None,
            "source": "full_train",
            "warnings": all_warnings,
        },
    )
    append_log(job_dir, f"[done] status={final_status}; source=full_train; result_dir={result_dir}.")
    if partial_reason:
        append_log(job_dir, f"[warning] {partial_reason}.")
    return 0 if final_status in {"completed", "partial"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FedVLR workbench full-training job.")
    parser.add_argument("--job-id", required=True, help="Safe workbench job id.")
    parser.add_argument("--payload-file", help="Raw workbench payload JSON file.")
    parser.add_argument("--config", help="Normalized workbench config JSON file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Workbench output root or job directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job_id = generator.safe_job_id(args.job_id)
    job_dir = safe_job_dir(job_id, Path(args.output_dir))
    try:
        payload_path = Path(args.payload_file or args.config) if (args.payload_file or args.config) else None
        payload = read_json(payload_path, {}) if payload_path else {}
        if not isinstance(payload, dict):
            raise ValueError("payload_must_be_json_object")
        return run_job(job_id, payload, job_dir)
    except Exception as exc:  # noqa: BLE001
        now = utc_now()
        append_log(job_dir, f"[failed] {exc}")
        append_log(job_dir, traceback.format_exc().rstrip())
        update_status(
            job_dir,
            {
                "job_id": job_id,
                "status": "failed",
                "stage": "failed",
                "progress": 100,
                "updated_at": now,
                "finished_at": now,
                "error_message": str(exc),
            },
        )
        write_json(
            job_dir / "metrics_summary.json",
            {
                "job_id": job_id,
                "status": "failed",
                "source": None,
                "metrics": {},
                "message": "真实全量训练任务执行失败。",
                "error_message": str(exc),
            },
        )
        write_json(
            job_dir / "result_pointer.json",
            {
                "job_id": job_id,
                "status": "failed",
                "config": "config.json",
                "status_file": "status.json",
                "metrics_summary": "metrics_summary.json",
                "log": "run.log",
                "result_dir": None,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
