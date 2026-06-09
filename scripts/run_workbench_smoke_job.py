from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "workbench_jobs"
DEFAULT_SCENARIO_ID = "amazon_beauty_poc_security_v3"
SHOWCASE_ROOT = ROOT / "outputs" / "showcase_artifacts"
SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
REAL_SMOKE_MODELS = {"FedAvg", "FedRAP"}
RECOMMENDATION_REAL_SMOKE_TEMPLATE = ROOT / "configs" / "experiment_smoke" / "amazon_beauty_poc_target_injection_smoke.json"
REAL_SMOKE_TEMPLATE_BY_ALGORITHM = {
    "krum": ROOT / "configs" / "experiment_smoke" / "security_matrix" / "poisoning_krum_topk_smoke.json",
    "median": ROOT / "configs" / "experiment_smoke" / "security_matrix" / "poisoning_median_topk_smoke.json",
    "trimmedmean": ROOT / "configs" / "experiment_smoke" / "security_matrix" / "poisoning_trimmed_mean_topk_smoke.json",
    "trimmed_mean": ROOT / "configs" / "experiment_smoke" / "security_matrix" / "poisoning_trimmed_mean_topk_smoke.json",
    "bulyan": ROOT / "configs" / "experiment_smoke" / "security_matrix" / "poisoning_bulyan_topk_smoke.json",
}

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


def cap_int(value: Any, default: int, upper: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, upper))


def cap_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(parsed, upper))


def build_launcher_config(normalized: Dict[str, Any], raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    unified = dict(normalized.get("unified_experiment_config") or {})
    training = dict(unified.get("training_params") or normalized.get("training") or {})

    allow_extended = bool(raw_payload.get("allow_extended_smoke") or raw_payload.get("allow_runtime_overrides"))
    max_epochs = 5 if allow_extended else 2
    max_client_ratio = 0.1 if raw_payload.get("explicit_large_client_ratio") or raw_payload.get("allow_large_client_ratio") else 0.05

    epochs = cap_int(training.get("epochs") or training.get("total_rounds"), 1, max_epochs)
    total_rounds = cap_int(training.get("total_rounds") or epochs, epochs, max_epochs)
    local_epochs = 1
    client_ratio = cap_float(training.get("client_sampling_ratio"), 0.05, 0.01, max_client_ratio)

    training.update(
        {
            "epochs": epochs,
            "total_rounds": total_rounds,
            "local_epochs": local_epochs,
            "client_sampling_ratio": client_ratio,
            "bounded_by_workbench_runner": True,
        }
    )
    unified["training_params"] = training
    unified["workbench_runtime_limits"] = {
        "max_epochs": max_epochs,
        "max_local_epochs": 1,
        "max_client_sampling_ratio": max_client_ratio,
        "no_arbitrary_command": True,
        "no_output_deletion": True,
    }
    return unified


def robust_algorithm_key(normalized: Dict[str, Any]) -> str:
    values = normalized.get("robust_aggregators")
    if isinstance(values, str):
        values = [values]
    if isinstance(values, list) and values:
        first = str(values[0]).strip()
    else:
        first = "Krum"
    return first.replace("-", "_").replace(" ", "").lower()


def can_run_real_smoke(normalized: Dict[str, Any]) -> bool:
    if str(normalized.get("execution_mode")) != "real_smoke":
        return False
    direction = str(normalized.get("direction"))
    dataset = str(normalized.get("dataset"))
    model = str(normalized.get("model"))
    if direction == "recommendation_manipulation":
        return dataset == "AMAZON_BEAUTY_POC" and model == "FedAvg"
    return direction == "aggregation_defense" and dataset == "KU" and model in REAL_SMOKE_MODELS


def build_real_smoke_launcher_config(job_id: str, normalized: Dict[str, Any]) -> Tuple[Dict[str, Any], Path, List[str]]:
    model = str(normalized.get("model") or "FedAvg")
    direction = str(normalized.get("direction") or "aggregation_defense")
    algorithm_key = robust_algorithm_key(normalized)
    if direction == "recommendation_manipulation":
        template_path = RECOMMENDATION_REAL_SMOKE_TEMPLATE
    else:
        template_path = REAL_SMOKE_TEMPLATE_BY_ALGORITHM.get(algorithm_key) or REAL_SMOKE_TEMPLATE_BY_ALGORITHM["krum"]
    config = read_json(template_path, {}) or {}
    if not isinstance(config, dict):
        raise ValueError("invalid_real_smoke_template")

    warnings: List[str] = []
    if direction == "aggregation_defense" and algorithm_key not in REAL_SMOKE_TEMPLATE_BY_ALGORITHM:
        warnings.append(f"unsupported_robust_algorithm_for_real_smoke:{algorithm_key};fallback=krum")

    training = dict(config.get("training_params") or {})
    training.update(
        {
            "epochs": 1,
            "total_rounds": 1,
            "local_epochs": 1,
            "clients_sample_ratio": 0.05,
            "client_sampling_ratio": 0.05,
            "use_gpu": False,
            "eval_step": 1,
            "collect_round_metrics": True,
            "save_recommended_topk": bool((normalized.get("training") or {}).get("save_topk", True)),
        }
    )
    malicious_ratio = (normalized.get("attack") or {}).get("malicious_client_ratio", 0.2)
    try:
        malicious_ratio = max(0.0, min(float(malicious_ratio), 0.2))
    except (TypeError, ValueError):
        malicious_ratio = 0.2

    if direction == "recommendation_manipulation":
        attack = dict(normalized.get("attack") or {})
        target_item_id = str(attack.get("target_item_id") or "0")
        try:
            injection_ratio = max(0.0, min(float(attack.get("injection_ratio", 0.2)), 1.0))
        except (TypeError, ValueError):
            injection_ratio = 0.2
        try:
            max_injections = max(1, min(int(float(attack.get("max_injections_per_client", 10))), 100))
        except (TypeError, ValueError):
            max_injections = 10
        injection = dict((config.get("attack_params") or {}).get("target_interaction_injection") or {})
        injection.update(
            {
                "enabled": True,
                "target_item_ids": [target_item_id],
                "target_item_title": attack.get("target_item_title"),
                "injection_ratio": injection_ratio,
                "malicious_client_ratio": malicious_ratio,
                "max_injections_per_client": max_injections,
                "planner_only": False,
            }
        )
        config.update(
            {
                "model": "FedAvg",
                "dataset": "AMAZON_BEAUTY_POC",
                "scenario": "attack_only",
                "type": "WorkbenchRecommendationManipulationSmoke",
                "comment": f"workbench_real_smoke_{job_id}",
                "enabled_attacks": ["target_interaction_injection"],
                "enabled_defenses": [],
                "enabled_privacy_metrics": [],
                "training_params": training,
                "malicious_client_config": {"enabled": malicious_ratio > 0, "mode": "ratio", "ratio": malicious_ratio, "client_ids": []},
                "attack_params": {"target_interaction_injection": injection},
            }
        )
    else:
        config.update(
            {
                "model": model,
                "dataset": "KU",
                "scenario": "attack_and_defense",
                "type": "WorkbenchAggregationDefenseSmoke",
                "comment": f"workbench_real_smoke_{job_id}",
                "training_params": training,
                "malicious_client_config": {"enabled": malicious_ratio > 0, "mode": "ratio", "ratio": malicious_ratio, "client_ids": []},
            }
        )
    extra_config = dict(config.get("extra_config") or {})
    extra_config.update(
        {
            "smoke_only": True,
            "workbench_job_id": job_id,
            "workbench_source": "real_smoke",
            "workbench_direction": direction,
            "runtime_limits": {
                "max_epochs": 1,
                "max_local_epochs": 1,
                "max_client_sampling_ratio": 0.05,
                "no_arbitrary_command": True,
                "no_output_deletion": True,
            },
        }
    )
    config["extra_config"] = extra_config
    return config, template_path, warnings


def parse_first_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def run_real_smoke(job_id: str, job_dir: Path, launcher_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any], str | None]:
    launcher_path = ROOT / "scripts" / "launch_experiment.py"
    if not launcher_path.exists():
        raise FileNotFoundError("launch_experiment.py")
    config_path = job_dir / "launcher_config.json"
    write_json(config_path, launcher_config)
    command = [sys.executable, str(launcher_path), "--config", str(config_path)]

    update_status(job_dir, {"status": "running", "stage": "running", "progress": 42, "source": "real_smoke"})
    append_log(job_dir, f"[run] Starting real bounded smoke via {repo_relative(launcher_path)}.")
    started = time.time()
    process = subprocess.run(  # noqa: S603 - fixed Python executable and whitelisted launcher path.
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    elapsed = round(time.time() - started, 3)
    with (job_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(process.stdout or "")
        if process.stdout and not process.stdout.endswith("\n"):
            handle.write("\n")
    append_log(job_dir, f"[run] Launcher exited with code={process.returncode}; elapsed={elapsed}s.")

    launch_payload = parse_first_json_object(process.stdout or "")
    if process.returncode != 0 or not launch_payload.get("ok"):
        error_message = launch_payload.get("errors") or f"launcher_failed:{process.returncode}"
        return "failed", {}, {"launch_payload": launch_payload}, str(error_message)

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
        "source": "real_smoke",
        "active_attacks": experiment.get("active_attacks", []),
        "active_defenses": experiment.get("active_defenses", []),
        "loss": final_eval.get("loss") if isinstance(final_eval, dict) else None,
        "epochs": training_config.get("epochs") if isinstance(training_config, dict) else 1,
        "local_epochs": training_config.get("local_epochs") if isinstance(training_config, dict) else 1,
        "client_sampling_ratio": training_config.get("clients_sample_ratio") if isinstance(training_config, dict) else 0.05,
    }
    pointer = {
        "launch_payload": launch_payload,
        "result_dir": repo_relative(ROOT / str(experiment.get("result_dir"))) if experiment.get("result_dir") else None,
        "summary_path": repo_relative(summary_path),
        "result_path": repo_relative(ROOT / str(experiment.get("result_path"))) if experiment.get("result_path") else None,
        "csv_path": repo_relative(csv_path),
        "recommend_topk_dir": repo_relative(ROOT / str(experiment.get("recommend_topk_dir"))) if experiment.get("recommend_topk_dir") else None,
    }
    warnings = list(launch_payload.get("warnings", []))
    status = "completed"
    partial_reason = None
    if warnings:
        partial_reason = "real smoke completed with launcher warnings; defense effect is smoke-level only"
    return status, metrics, pointer, partial_reason


def _number_or_raw(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def direction_panel(direction: str) -> Tuple[str, str]:
    if direction == "recommendation_manipulation":
        return "target_manipulation_metrics.json", "推荐操纵证据"
    if direction == "membership_inference":
        return "membership_inference_panel.json", "成员推断证据"
    if direction == "update_leakage":
        return "update_leakage_panel.json", "更新泄露证据"
    if direction == "aggregation_defense":
        return "aggregation_defense_panel.json", "聚合防御证据"
    return "frontend_summary.json", "工作台证据"


def compact_metric_items(metrics: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
    return compact


def extract_metrics(direction: str, panel: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str | None]:
    if not panel:
        return "partial", {}, "未找到可复用的 V3 证据面板"
    panel_status = str(panel.get("status") or "available")
    if direction == "recommendation_manipulation":
        metrics = compact_metric_items(
            panel,
            [
                "baseline_unmasked_rank",
                "attack_unmasked_rank",
                "rank_gain",
                "normalized_rank_gain",
                "reciprocal_rank_gain",
                "attack_topk_hit",
                "target_manipulation_index",
                "recommendation_jaccard",
                "changed_user_count",
                "changed_item_count",
            ],
        )
        return "completed" if panel_status == "available" else "partial", metrics, None
    if direction == "membership_inference":
        metrics = compact_metric_items(
            panel,
            [
                "auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "score_gap",
                "member_count",
                "non_member_count",
                "evidence_type",
            ],
        )
        return "completed" if panel.get("auc") is not None or panel.get("accuracy") is not None else "partial", metrics, None
    if direction == "update_leakage":
        hit = panel.get("hit_at_k") if isinstance(panel.get("hit_at_k"), dict) else {}
        metrics = {
            "hit_at_10": panel.get("hit_at_10", hit.get("10") or hit.get(10)),
            "hit_at_20": panel.get("hit_at_20", hit.get("20") or hit.get(20)),
            "hit_at_50": panel.get("hit_at_50", hit.get("50") or hit.get(50)),
            "highest_risk_modality": panel.get("highest_risk_modality"),
            "candidate_item_count": panel.get("candidate_item_count"),
        }
        return "completed" if any(value is not None for value in metrics.values()) else "partial", metrics, None
    if direction == "aggregation_defense":
        recovery = panel.get("recovery_rate") if isinstance(panel.get("recovery_rate"), dict) else {}
        metrics = {
            "defense_algorithm": panel.get("defense_algorithm"),
            "aggregation_visibility": panel.get("aggregation_visibility"),
            "recall_before": panel.get("recall_before"),
            "recall_after": panel.get("recall_after"),
            "ndcg_before": panel.get("ndcg_before"),
            "ndcg_after": panel.get("ndcg_after"),
            "recovery_rate_recall": recovery.get("recall"),
            "recovery_rate_ndcg": recovery.get("ndcg"),
            "selected_client_count": len(panel.get("selected_clients") or []),
            "rejected_client_count": len(panel.get("rejected_clients") or []),
        }
        if panel_status == "configured_only":
            return "partial", metrics, "已配置，未形成完整 benchmark"
        return "completed" if panel_status in {"available", "completed"} else "partial", metrics, None
    return "completed", {}, None


def run_job(job_id: str, payload: Dict[str, Any], job_dir: Path) -> int:
    created_at = utc_now()
    (job_dir / "run.log").write_text("", encoding="utf-8")
    update_status(
        job_dir,
        {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "error_message": None,
            "source": None,
            "result_dir": None,
            "artifact_dir": None,
        },
    )
    append_log(job_dir, f"Workbench job {job_id} queued.")
    time.sleep(0.15)

    update_status(job_dir, {"status": "running", "stage": "preparing_config", "progress": 12, "started_at": utc_now()})
    append_log(job_dir, "[prepare] Validating bounded smoke payload.")

    normalized, warnings, errors = normalize_input(payload)
    if errors:
        raise ValueError(";".join(str(item) for item in errors))
    write_json(job_dir / "config.json", normalized)
    direction = str(normalized.get("direction") or "recommendation_manipulation")
    if can_run_real_smoke(normalized):
        launcher_config, template_path, real_warnings = build_real_smoke_launcher_config(job_id, normalized)
        warnings = [*warnings, *real_warnings]
        write_json(job_dir / "launcher_config.json", launcher_config)
        append_log(job_dir, f"[prepare] Wrote real smoke launcher_config.json from {repo_relative(template_path)}.")
        update_status(
            job_dir,
            {
                "stage": "preparing_config",
                "progress": 24,
                "direction": direction,
                "scenario_id": normalized.get("scenario_id"),
                "source": "real_smoke",
                "warnings": warnings,
                "errors": [],
            },
        )
        final_status, metrics, pointer_extra, partial_reason = run_real_smoke(job_id, job_dir, launcher_config)
        launcher_warnings = []
        launch_payload = pointer_extra.get("launch_payload") if isinstance(pointer_extra, dict) else {}
        if isinstance(launch_payload, dict):
            launcher_warnings = [str(item) for item in launch_payload.get("warnings", [])]
        all_warnings = [*warnings, *launcher_warnings]

        update_status(job_dir, {"stage": "exporting_artifacts", "progress": 86, "warnings": all_warnings})
        append_log(job_dir, "[export] Writing real smoke metrics_summary and result_pointer.")
        result_dir = pointer_extra.get("result_dir") if isinstance(pointer_extra, dict) else None
        result_pointer = {
            "job_id": job_id,
            "status": final_status,
            "source": "real_smoke",
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
            "source": "real_smoke",
            "direction": direction,
            "scenario_id": normalized.get("scenario_id"),
            "model": metrics.get("model"),
            "dataset": metrics.get("dataset"),
            "metrics": metrics,
            "warnings": all_warnings,
            "message": "真实受限 smoke job 已执行；该结果只代表 1 epoch 小规模链路验证。",
            "partial_reason": partial_reason,
            "runtime_limits": launcher_config.get("extra_config", {}).get("runtime_limits"),
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
                "result_dir": result_dir,
                "artifact_dir": None,
                "source": "real_smoke",
                "warnings": all_warnings,
            },
        )
        append_log(job_dir, f"[done] status={final_status}; source=real_smoke; result_dir={result_dir}.")
        if partial_reason:
            append_log(job_dir, f"[boundary] {partial_reason}.")
        return 0 if final_status in {"completed", "partial"} else 1

    launcher_config = build_launcher_config(normalized, payload)
    write_json(job_dir / "launcher_config.json", launcher_config)
    append_log(job_dir, "[prepare] Wrote config.json and launcher_config.json with runtime limits.")
    time.sleep(0.2)
    fallback_source = "probe_smoke" if str(normalized.get("execution_mode")) == "probe_smoke" else "existing_artifact"

    scenario_id = str(normalized.get("scenario_id") or DEFAULT_SCENARIO_ID)
    artifact_dir = (SHOWCASE_ROOT / scenario_id).resolve()
    if not artifact_dir.exists():
        artifact_dir = (SHOWCASE_ROOT / DEFAULT_SCENARIO_ID).resolve()
        scenario_id = DEFAULT_SCENARIO_ID
    if SHOWCASE_ROOT.resolve() not in [artifact_dir, *artifact_dir.parents]:
        raise ValueError("artifact_path_outside_showcase_root")

    update_status(
        job_dir,
        {
            "stage": "running",
            "progress": 45,
            "direction": direction,
            "scenario_id": scenario_id,
            "source": fallback_source,
            "artifact_dir": repo_relative(artifact_dir),
            "warnings": warnings,
            "errors": [],
        },
    )
    append_log(job_dir, f"[run] Direction={direction}; source={fallback_source}; using bounded exported evidence/probe envelope.")
    append_log(job_dir, "[run] No arbitrary shell command or long training is executed by this runner.")
    time.sleep(0.25)

    panel_file, evidence_label = direction_panel(direction)
    panel_path = artifact_dir / panel_file
    panel = read_json(panel_path, {}) or {}
    final_status, metrics, partial_reason = extract_metrics(direction, panel)

    update_status(job_dir, {"stage": "exporting_artifacts", "progress": 78})
    append_log(job_dir, f"[export] Reading {evidence_label} from {repo_relative(panel_path) or panel_file}.")
    time.sleep(0.2)

    result_pointer = {
        "job_id": job_id,
        "status": final_status,
        "source": fallback_source,
        "config": "config.json",
        "launcher_config": "launcher_config.json",
        "status_file": "status.json",
        "metrics_summary": "metrics_summary.json",
        "log": "run.log",
        "result_dir": None,
        "artifact_dir": repo_relative(artifact_dir),
        "showcase_scenario_id": scenario_id,
        "generated_panels": {direction: repo_relative(panel_path)},
    }
    metrics_summary = {
        "job_id": job_id,
        "status": final_status,
        "source": fallback_source,
        "direction": direction,
        "scenario_id": scenario_id,
        "metrics": metrics,
        "message": "运行轻量 probe，复用导出的安全证据生成结果摘要；不是完整训练 benchmark。" if fallback_source == "probe_smoke" else "复用已导出的安全证据；不是本次新训练生成的完整结果。",
        "partial_reason": partial_reason,
        "runtime_limits": launcher_config.get("workbench_runtime_limits"),
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
            "error_message": partial_reason,
            "result_dir": None,
            "artifact_dir": repo_relative(artifact_dir),
            "source": fallback_source,
        },
    )
    append_log(job_dir, f"[done] status={final_status}; source={fallback_source}.")
    if partial_reason:
        append_log(job_dir, f"[boundary] {partial_reason}.")
    return 0 if final_status in {"completed", "partial"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded FedVLR workbench smoke job.")
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
                "message": "受限 smoke job 执行失败。",
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
