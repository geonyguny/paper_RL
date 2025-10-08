# tests/test_hjb_calib.py
import json
import subprocess
import sys

def run(cmd):
    out = subprocess.check_output(cmd, shell=True, text=True)
    return json.loads(out)

def test_hjb_cvar_target():
    cmd = (
        "python -m project.runner.cli "
        "--method hjb "
        "--market_mode bootstrap "
        "--data_profile full "
        "--es_mode loss "
        "--F_target 0.60 "
        "--cvar_target 0.40 "
        "--cvar_tol 0.01 "
        "--lambda_min 0.0 "
        "--lambda_max 5.0 "
        "--calib_fast on "
        "--calib_max_iter 8 "
        "--print_mode summary "
        "--metrics_keys EW,ES95,mean_WT,es95_source "
        "--tag hjb_calib_check_ci"
    )
    data = run(cmd)
    es = data["metrics"]["ES95"]
    assert 0.39 <= es <= 0.41
