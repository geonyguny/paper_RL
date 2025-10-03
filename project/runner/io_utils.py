# project/runner/io_utils.py
from __future__ import annotations
import os
import csv
import json
import datetime
from typing import Any, Dict, List


# =========================
# Constants / Basics
# =========================
FIELDNAMES: List[str] = [
    "ts", "asset", "method", "baseline", "es_mode", "tag",
    "EW", "ES95", "Ruin", "mean_WT",
    "w_max", "fee_annual", "lambda_term", "alpha", "F_target",
    "n_paths",
    "args_json",
]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _now_ts_utc() -> str:
    """YYMMDDTHHMMSS (UTC) short timestamp."""
    return datetime.datetime.utcnow().strftime("%y%m%dT%H%M%S")

def _metrics_log_dir(outputs_dir: str) -> str:
    return os.path.join(outputs_dir, "_logs")

def _metrics_log_path(outputs_dir: str) -> str:
    return os.path.join(_metrics_log_dir(outputs_dir), "metrics.csv")


# =========================
# Slim args
# =========================
def slim_args(args) -> dict:
    """CSV에는 핵심 인자를 JSON으로 보존."""
    keys = [
        "asset", "method", "baseline", "w_max", "fee_annual", "horizon_years",
        "alpha", "lambda_term", "F_target", "p_annual", "g_real_annual",
        "w_fixed", "floor_on", "f_min_real", "es_mode", "outputs",
        "hjb_W_grid", "hjb_Nshock", "hjb_eta_n",
        "hedge", "hedge_mode", "hedge_cost", "hedge_sigma_k", "hedge_tx",
        "market_mode", "market_csv", "bootstrap_block", "use_real_rf",
        "mortality", "mort_table", "age0", "sex", "bequest_kappa", "bequest_gamma",
        "cvar_stage", "alpha_stage", "lambda_stage", "cstar_mode", "cstar_m",
        "rl_q_cap", "teacher_eps0", "teacher_decay", "lw_scale", "survive_bonus",
        "crra_gamma", "u_scale", "xai_on",
        "seeds", "n_paths",
        "rl_epochs", "rl_steps_per_epoch", "rl_n_paths_eval", "gae_lambda",
        "entropy_coef", "value_coef", "lr", "max_grad_norm",
        "q_floor", "beta", "quiet",
        "ann_on", "ann_alpha", "ann_L", "ann_d", "ann_index",
        # 추가 메타
        "bands", "data_window", "data_profile", "tag",
    ]
    return {k: getattr(args, k, None) for k in keys}


# =========================
# CSV header migration
# =========================
def _first_line(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except Exception:
        return ""

def _needs_rotation(existing_header: str) -> bool:
    if not existing_header:
        return False
    # 간단 비교: 콤마로 split 후 필드명이 기대 스키마와 다르면 회전
    current = [h.strip() for h in existing_header.split(",") if h.strip()]
    return current != FIELDNAMES

def _ensure_metrics_csv(outputs_dir: str) -> str:
    """헤더가 다르면 기존 파일을 회전(.old_YYMMDDTHHMMSS)하고 새 헤더 생성."""
    logs_dir = _metrics_log_dir(outputs_dir)
    ensure_dir(logs_dir)
    csv_path = _metrics_log_path(outputs_dir)

    if os.path.exists(csv_path):
        header = _first_line(csv_path)
        if _needs_rotation(header):
            ts = _now_ts_utc()
            rotated = os.path.join(logs_dir, f"metrics_OLD_{ts}.csv")
            os.replace(csv_path, rotated)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

    return csv_path


# =========================
# Autosave
# =========================
def do_autosave(metrics: Dict[str, Any], cfg, args, out_payload: Dict[str, Any]) -> None:
    """
    metrics.csv에 안전하게 append.
    - 스키마 자동 정합성 검사/회전
    - 고정 필드명/순서
    - JSON 직렬화로 전체 args 보존
    """
    try:
        csv_path = _ensure_metrics_csv(args.outputs)

        row = {
            "ts": _now_ts_utc(),
            "asset": out_payload.get("asset", getattr(cfg, "asset", None)),
            "method": out_payload.get("method", getattr(args, "method", None)),
            "baseline": out_payload.get("baseline", getattr(args, "baseline", None)),
            "es_mode": out_payload.get("es_mode", getattr(args, "es_mode", None)),
            "tag": getattr(cfg, "tag", None) or getattr(args, "tag", None) or "",
            "EW": (metrics or {}).get("EW"),
            "ES95": (metrics or {}).get("ES95"),
            "Ruin": (metrics or {}).get("Ruin"),
            "mean_WT": (metrics or {}).get("mean_WT"),
            "w_max": getattr(cfg, "w_max", None),
            "fee_annual": out_payload.get(
                "fee_annual",
                getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
            ),
            "lambda_term": out_payload.get("lambda_term", getattr(cfg, "lambda_term", None)),
            "alpha": out_payload.get("alpha", getattr(cfg, "alpha", None)),
            "F_target": out_payload.get("F_target", getattr(cfg, "F_target", None)),
            "n_paths": out_payload.get("n_paths"),
            "args_json": json.dumps(slim_args(args), ensure_ascii=False),
        }

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)

        print(f"[autosave] metrics -> {csv_path}")

    except Exception as e:
        print(f"[autosave] skipped: {e}")
