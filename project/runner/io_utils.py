# project/runner/io_utils.py
from __future__ import annotations
import os, csv, datetime
from typing import Any, Dict

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def now_iso() -> str: return datetime.datetime.now().isoformat(timespec="seconds")

def slim_args(args) -> dict:
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
    ]
    return {k: getattr(args, k, None) for k in keys}

def append_metrics_csv(path: str, payload: Dict[str, Any]):
    row = {
        'ts': now_iso(),
        'asset': payload.get('asset'),
        'method': payload.get('method'),
        'lambda': payload.get('lambda_term'),
        'F_target': payload.get('F_target'),
        'alpha': payload.get('alpha'),
        'ES95': (payload.get('metrics') or {}).get('ES95'),
        'EW': (payload.get('metrics') or {}).get('EW'),
        'Ruin': (payload.get('metrics') or {}).get('Ruin'),
        'mean_WT': (payload.get('metrics') or {}).get('mean_WT'),
        'hedge_on': (payload.get('args') or {}).get('hedge') == 'on',
        'hedge_mode': (payload.get('args') or {}).get('hedge_mode'),
        'fee_annual': payload.get('fee_annual'),
        'w_max': payload.get('w_max'),
        'horizon_years': payload.get('horizon_years'),
        'seeds': (payload.get('args') or {}).get('seeds'),
        'n_paths': (payload.get('args') or {}).get('n_paths'),
        'mortality_on': (payload.get('args') or {}).get('mortality') == 'on',
        'market_mode': (payload.get('args') or {}).get('market_mode'),
        'ann_on': (payload.get('args') or {}).get('ann_on'),
        'ann_alpha': (payload.get('args') or {}).get('ann_alpha'),
        'ann_L': (payload.get('args') or {}).get('ann_L'),
        'ann_d': (payload.get('args') or {}).get('ann_d'),
        'ann_index': (payload.get('args') or {}).get('ann_index'),
        'y_ann': (payload.get('metrics') or {}).get('y_ann'),
        'a_factor': (payload.get('metrics') or {}).get('a_factor'),
        'P': (payload.get('metrics') or {}).get('P'),
    }
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

def do_autosave(metrics: dict, cfg, args, out_payload: dict):
    try:
        try:
            from ..eval import save_metrics_autocsv  # optional
            csv_path = save_metrics_autocsv(metrics, cfg, outputs=cfg.outputs)
            print(f"[autosave] metrics -> {csv_path}")
        except Exception:
            ensure_dir(os.path.join(cfg.outputs, "_logs"))
            csv_path = os.path.join(cfg.outputs, "_logs", "metrics.csv")
            append_metrics_csv(csv_path, out_payload)
            print(f"[autosave:fallback] metrics -> {csv_path}")
    except Exception as e:
        print(f"[autosave] skipped: {e}")
