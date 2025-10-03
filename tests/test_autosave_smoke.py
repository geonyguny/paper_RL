# tests/test_autosave_smoke.py
import os, tempfile
from project.runner.io_utils import append_metrics_csv, CSV_FIELDS, SCHEMA_VERSION

def test_autosave_csv_schema_and_row():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "metrics.csv")

    payload = {
        "asset": "US",
        "method": "rl",
        "baseline": "",
        "metrics": {"EW": 0.1, "ES95": 0.0, "Ruin": 0.0, "mean_WT": 0.1, "best_epoch": 3},
        "w_max": 0.7,
        "fee_annual": 0.004,
        "lambda_term": 0.0,
        "alpha": 0.95,
        "F_target": 0.0,
        "es_mode": "wealth",
        "n_paths": 1500,
        "args": {
            "outputs": "./outputs",
            "market_mode": "bootstrap",
            "market_csv": "X.csv",
            "data_window": "2005-01:2020-12",
            "use_real_rf": "on",
            "bands": "on",
            "seeds": [0,1,2],
            "rl_epochs": 1,
            "rl_steps_per_epoch": 128,
            "rl_n_paths_eval": 64,
            "entropy_coef": 0.001,
            "value_coef": 0.5,
            "gae_lambda": 0.95,
            "lr": 1e-4,
            "max_grad_norm": 0.5,
            "rl_q_cap": 0.005,
            "teacher_eps0": 1.0,
            "teacher_decay": 0.999,
            "survive_bonus": 0.0,
            "u_scale": 0.0,
            "lw_scale": 0.0,
            "hedge": "off",
            "hedge_mode": "sigma",
            "hedge_sigma_k": 0.2,
            "hedge_cost": 0.005,
            "hedge_tx": 0.0,
            "mortality": "off",
            "ann_on": "off",
            "ann_alpha": 0.0,
            "ann_L": 0.0,
            "ann_d": 0,
            "ann_index": "real",
            "tag": "smoke",
        },
        "ckpt_path": "./outputs/smoke/policy.pt",
    }

    append_metrics_csv(path, payload)

    assert os.path.exists(path)
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    # 첫 줄: 헤더, 둘째 줄: 데이터
    assert len(lines) >= 2
    header = lines[0].split(",")
    assert header == CSV_FIELDS
    # 스키마 버전 체크 (둘째 줄 두 번째 컬럼)
    assert lines[1].split(",")[1] == SCHEMA_VERSION
