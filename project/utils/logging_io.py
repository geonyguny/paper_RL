# project/utils/logging_io.py
import os, csv, json, time
from datetime import datetime

METRIC_HEADER = [
    "ts","asset","method","baseline","es_mode","alpha","lambda_term","F_target",
    "EW","EL","ES95","Ruin","mean_WT",
    "HedgeHit","HedgeKMean","HedgeActiveW",
    "fee_annual","w_max","floor_on","f_min_real",
    "hedge_on","hedge_mode","hedge_sigma_k","hedge_cost","hedge_tx",
    "horizon_years","steps_per_year","seeds","n_paths_eval",
    "market_mode","market_csv","bootstrap_block","use_real_rf","tag"
]

def save_metrics_autocsv(csv_path: str, args: dict, metrics: dict, tag: str=""):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    row = [
        datetime.utcnow().isoformat(timespec="seconds"),
        args.get("asset"), args.get("method"), args.get("baseline"),
        args.get("es_mode"), args.get("alpha"), args.get("lambda_term"), args.get("F_target"),
        round(float(metrics.get("EW", float("nan"))), 6),
        round(float(metrics.get("EL", float("nan"))), 6),
        round(float(metrics.get("ES95", float("nan"))), 6),
        round(float(metrics.get("Ruin", float("nan"))), 6),
        round(float(metrics.get("mean_WT", float("nan"))), 6),
        round(float(metrics.get("HedgeHit", float("nan"))), 6),
        round(float(metrics.get("HedgeKMean", float("nan"))), 6),
        round(float(metrics.get("HedgeActiveW", float("nan"))), 6),
        args.get("fee_annual"), args.get("w_max"),
        args.get("floor_on", False), args.get("f_min_real"),
        args.get("hedge_on", False), args.get("hedge_mode"), args.get("hedge_sigma_k"),
        args.get("hedge_cost"), args.get("hedge_tx"),
        args.get("horizon_years"), args.get("steps_per_year"),
        args.get("seeds"), args.get("n_paths_eval"),
        args.get("market_mode"), args.get("market_csv"),
        args.get("bootstrap_block"), args.get("use_real_rf"), tag
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(METRIC_HEADER)
        w.writerow(row)

def dump_result_json(path: str, args: dict, metrics: dict, extras: dict=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "ts": int(time.time()),
        "args": args,           # hedge_cost/hedge_tx 이미 args에 포함
        "metrics": metrics,
        "extras": extras or {}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
