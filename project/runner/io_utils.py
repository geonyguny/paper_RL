# project/runner/io_utils.py
from __future__ import annotations
import os, csv, datetime
from typing import Any, Dict

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def now_iso() -> str:
    # 파일명·정렬 친화적 타임스탬프(로컬)
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

SCHEMA_VERSION = "v2"

# 고정 헤더(열 순서 보장)
CSV_FIELDS = [
    "ts", "schema", "asset", "method", "es_mode", "tag",
    "alpha", "lambda", "F_target",
    "EW", "ES95", "Ruin", "mean_WT",
    "best_epoch", "train_time_s", "eval_time_s",
    "y_ann", "a_factor", "P",
    "fee_annual", "w_max", "horizon_years",
    "market_mode", "market_csv", "data_window", "use_real_rf",
    "bands", "outputs",
    "seeds", "n_paths",
    # RL 하이퍼들
    "rl_epochs", "rl_steps_per_epoch", "rl_n_paths_eval",
    "entropy_coef", "value_coef", "gae_lambda", "lr", "max_grad_norm",
    "rl_q_cap", "teacher_eps0", "teacher_decay",
    "survive_bonus", "u_scale", "lw_scale",
    # Hedge / mortality / ann
    "hedge", "hedge_mode", "hedge_sigma_k", "hedge_cost", "hedge_tx",
    "mortality", "ann_on", "ann_alpha", "ann_L", "ann_d", "ann_index",
    # 참조 경로
    "ckpt_path",
]

def _s(v: Any) -> Any:
    # CSV 안정화를 위해 리스트/튜플은 공백 구분 문자열로, None은 빈칸으로
    if v is None: return ""
    if isinstance(v, (list, tuple)): return " ".join(str(x) for x in v)
    return v

def slim_args(args) -> dict:
    # 기존 함수 유지(외부 사용)
    keys = [
        "asset","method","baseline","w_max","fee_annual","horizon_years",
        "alpha","lambda_term","F_target","p_annual","g_real_annual",
        "w_fixed","floor_on","f_min_real","es_mode","outputs",
        "hjb_W_grid","hjb_Nshock","hjb_eta_n",
        "hedge","hedge_mode","hedge_cost","hedge_sigma_k","hedge_tx",
        "market_mode","market_csv","bootstrap_block","use_real_rf",
        "mortality","mort_table","age0","sex","bequest_kappa","bequest_gamma",
        "cvar_stage","alpha_stage","lambda_stage","cstar_mode","cstar_m",
        "rl_q_cap","teacher_eps0","teacher_decay","lw_scale","survive_bonus",
        "crra_gamma","u_scale","xai_on",
        "seeds","n_paths",
        "rl_epochs","rl_steps_per_epoch","rl_n_paths_eval","gae_lambda",
        "entropy_coef","value_coef","lr","max_grad_norm",
        "q_floor","beta","quiet",
        "ann_on","ann_alpha","ann_L","ann_d","ann_index",
        "bands","data_window","data_profile","tag",
    ]
    return {k: getattr(args, k, None) for k in keys}

def append_metrics_csv(path: str, payload: Dict[str, Any]):
    args = payload.get("args") or {}
    metrics = payload.get("metrics") or {}

    row = {
        "ts": now_iso(),
        "schema": SCHEMA_VERSION,
        "asset": payload.get("asset"),
        "method": payload.get("method"),
        "es_mode": payload.get("es_mode"),
        "tag": (args.get("tag") if isinstance(args, dict) else None) or "",

        "alpha": payload.get("alpha"),
        "lambda": payload.get("lambda_term"),
        "F_target": payload.get("F_target"),

        "EW": metrics.get("EW"),
        "ES95": metrics.get("ES95"),
        "Ruin": metrics.get("Ruin"),
        "mean_WT": metrics.get("mean_WT"),
        "best_epoch": metrics.get("best_epoch"),
        "train_time_s": metrics.get("train_time_s"),
        "eval_time_s": metrics.get("eval_time_s"),

        "y_ann": metrics.get("y_ann"),
        "a_factor": metrics.get("a_factor"),
        "P": metrics.get("P"),

        "fee_annual": payload.get("fee_annual"),
        "w_max": payload.get("w_max"),
        "horizon_years": payload.get("horizon_years"),

        "market_mode": (args.get("market_mode") if isinstance(args, dict) else None),
        "market_csv": (args.get("market_csv") if isinstance(args, dict) else None),
        "data_window": (args.get("data_window") if isinstance(args, dict) else None),
        "use_real_rf": (args.get("use_real_rf") if isinstance(args, dict) else None),

        "bands": (args.get("bands") if isinstance(args, dict) else None),
        "outputs": payload.get("args", {}).get("outputs") if isinstance(args, dict) else None,

        "seeds": (args.get("seeds") if isinstance(args, dict) else None),
        "n_paths": payload.get("n_paths"),

        "rl_epochs": (args.get("rl_epochs") if isinstance(args, dict) else None),
        "rl_steps_per_epoch": (args.get("rl_steps_per_epoch") if isinstance(args, dict) else None),
        "rl_n_paths_eval": (args.get("rl_n_paths_eval") if isinstance(args, dict) else None),
        "entropy_coef": (args.get("entropy_coef") if isinstance(args, dict) else None),
        "value_coef": (args.get("value_coef") if isinstance(args, dict) else None),
        "gae_lambda": (args.get("gae_lambda") if isinstance(args, dict) else None),
        "lr": (args.get("lr") if isinstance(args, dict) else None),
        "max_grad_norm": (args.get("max_grad_norm") if isinstance(args, dict) else None),

        "rl_q_cap": (args.get("rl_q_cap") if isinstance(args, dict) else None),
        "teacher_eps0": (args.get("teacher_eps0") if isinstance(args, dict) else None),
        "teacher_decay": (args.get("teacher_decay") if isinstance(args, dict) else None),
        "survive_bonus": (args.get("survive_bonus") if isinstance(args, dict) else None),
        "u_scale": (args.get("u_scale") if isinstance(args, dict) else None),
        "lw_scale": (args.get("lw_scale") if isinstance(args, dict) else None),

        "hedge": (args.get("hedge") if isinstance(args, dict) else None),
        "hedge_mode": (args.get("hedge_mode") if isinstance(args, dict) else None),
        "hedge_sigma_k": (args.get("hedge_sigma_k") if isinstance(args, dict) else None),
        "hedge_cost": (args.get("hedge_cost") if isinstance(args, dict) else None),
        "hedge_tx": (args.get("hedge_tx") if isinstance(args, dict) else None),

        "mortality": (args.get("mortality") if isinstance(args, dict) else None),
        "ann_on": (args.get("ann_on") if isinstance(args, dict) else None),
        "ann_alpha": (args.get("ann_alpha") if isinstance(args, dict) else None),
        "ann_L": (args.get("ann_L") if isinstance(args, dict) else None),
        "ann_d": (args.get("ann_d") if isinstance(args, dict) else None),
        "ann_index": (args.get("ann_index") if isinstance(args, dict) else None),

        "ckpt_path": payload.get("ckpt_path"),
    }

    # 타입/표기 정규화
    for k in list(row.keys()):
        row[k] = _s(row[k])

    write_header = not os.path.exists(path)
    ensure_dir(os.path.dirname(path))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)

def do_autosave(metrics: dict, cfg, args, out_payload: dict):
    """
    우선 eval.save_metrics_autocsv()가 있으면 그대로 사용.
    없으면 고정 스키마 CSV로 폴백.
    """
    try:
        try:
            from ..eval import save_metrics_autocsv  # optional
            csv_path = save_metrics_autocsv(metrics, cfg, outputs=cfg.outputs)
            print(f"[autosave] metrics -> {csv_path}")
        except Exception:
            csv_path = os.path.join(cfg.outputs, "_logs", "metrics.csv")
            append_metrics_csv(csv_path, out_payload)
            print(f"[autosave] metrics -> {csv_path}")
    except Exception as e:
        print(f"[autosave] skipped: {e}")
