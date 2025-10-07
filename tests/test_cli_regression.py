# tests/test_cli_regression.py
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(args):
    """
    project.runner.cli를 서브프로세스로 실행해 JSON을 파싱해 반환
    """
    cmd = [sys.executable, "-m", "project.runner.cli"] + list(map(str, args))
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def es95_interp(losses, alpha):
    """
    Acerbi–Tasche 표본식 (보간 포함 CVaR_α)
    L_(1) <= ... <= L_(n), j=floor(n*alpha), theta=n*alpha-j
    ES = ((1-theta)*L_(j+1) + sum_{i=j+2}^n L_(i)) / (n*(1-alpha))
    """
    L = sorted(float(x) for x in losses)
    n = len(L)
    assert n > 0
    a = max(min(float(alpha), 1.0 - 1e-12), 1e-12)
    j = int(math.floor(n * a))
    if j >= n:
        j = n - 1
    theta = n * a - j
    Lj1 = L[j]
    tail_sum = sum(L[j + 1 :])
    return ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))


# 공통 실행 인자(빠른 테스트를 위해 소형 데이터/짧은 러닝)
COMMON = [
    "--method", "rl",
    "--market_mode", "bootstrap",
    "--data_profile", "dev",
    "--es_mode", "loss",
    "--F_target", "0.60",
    "--rl_epochs", "1",
    "--rl_steps_per_epoch", "128",
    "--rl_n_paths_eval", "100",
    "--seeds", "0",
    "--alpha_mix", "equal",
    "--h_FX", "1",
    "--return_actor", "on",
]


def test_cvar_matches_acerbi_tasche():
    # paths 포함(full)로 실행하여 eval_WT 확보
    args = COMMON + ["--print_mode", "full", "--tag", "test_cvar_matches_acerbi_tasche"]
    res = run_cli(args)

    assert "metrics" in res and "extra" in res and "eval_WT" in res["extra"]
    wt = list(map(float, res["extra"]["eval_WT"]))
    assert len(wt) == 100  # rl_n_paths_eval=100, seed 1개

    # 손실 L = max(F - W_T, 0)
    F = 0.60
    L = [max(F - w, 0.0) for w in wt]

    es_cli = float(res["metrics"]["ES95"])
    es_ref = es95_interp(L, alpha=0.95)

    # 수치 오차는 충분히 엄격하게
    assert es_cli == pytest.approx(es_ref, rel=1e-9, abs=1e-12)


def test_print_mode_metrics_keys_and_shape():
    args = COMMON + [
        "--print_mode", "metrics",
        "--metrics_keys", "EW,ES95,Ruin,es95_source",
        "--tag", "test_print_mode_metrics",
    ]
    res = run_cli(args)

    # metrics 모드는 납작한 딕셔너리: 선택 키 + 메타(tag,asset,method,n_paths)
    expected_present = {"EW", "ES95", "Ruin", "es95_source", "tag", "asset", "method", "n_paths"}
    for k in expected_present:
        assert k in res

    # 불필요한 중첩 키가 없어야 함
    assert "metrics" not in res
    assert "extra" not in res


def test_print_mode_summary_fields():
    args = COMMON + [
        "--print_mode", "summary",
        "--metrics_keys", "EW,ES95,Ruin,mean_WT,es95_source",
        "--tag", "test_print_mode_summary",
    ]
    res = run_cli(args)

    # summary 모드 스키마 검증
    for k in ("tag", "asset", "method", "age0", "sex", "metrics", "n_paths"):
        assert k in res

    # n_paths 추정 로직: rl_n_paths_eval(100) * seeds(1) = 100
    assert res["n_paths"] == 100

    # metrics 내부 선택 키 노출 확인
    for k in ("EW", "ES95", "Ruin", "mean_WT", "es95_source"):
        assert k in res["metrics"]


def test_no_paths_strips_large_arrays_in_full():
    # 대용량 경로 제거(no_paths) 확인
    args = COMMON + [
        "--print_mode", "full",
        "--no_paths",
        "--tag", "test_no_paths_full",
    ]
    res = run_cli(args)

    assert "extra" in res
    # 배열이 제거되고 *_n 카운트만 남아야 함
    assert "eval_WT" not in res["extra"]
    assert "eval_WT_n" in res["extra"]
    assert res["extra"]["eval_WT_n"] == 100
