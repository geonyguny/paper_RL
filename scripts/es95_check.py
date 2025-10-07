# scripts/es95_check.py
from __future__ import annotations
import json
import math
import subprocess
import sys
from typing import List, Dict, Any, Optional

def var_alpha(losses: List[float], alpha: float) -> float:
    """VaR_α: 오름차순 정렬, (n-1)*α 위치 선형보간."""
    L = sorted(float(x) for x in losses)
    n = len(L)
    if n == 0:
        return 0.0
    idx = (n - 1) * alpha
    m = int(math.floor(idx))
    gamma = idx - m
    if m >= n - 1:
        return L[-1]
    return L[m] + gamma * (L[m + 1] - L[m])

def es_alpha_interp(losses: List[float], alpha: float) -> float:
    """
    ES_α (보간 포함 표본 추정, ascending 정렬 기준):
      j = floor(n*alpha), θ = n*alpha - j
      ES = ((1-θ)*L_(j+1) + sum_{i=j+2}^n L_(i)) / (n*(1-α))
    """
    L = sorted(float(x) for x in losses)  # ascending
    n = len(L)
    if n == 0:
        return 0.0
    j = int(math.floor(n * alpha))
    theta = n * alpha - j
    # 0-based: L_(j+1) == L[j]
    Lj1 = L[j]
    tail_sum = sum(L[j+1:])  # L_(j+2)..L_(n)
    return ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - alpha))

def es_alpha_simple(losses: List[float], alpha: float) -> float:
    """ES_α: 단순 상위 k=ceil(n*(1-α))개 평균(보간 없음, 참고용)."""
    L = sorted((float(x) for x in losses), reverse=True)
    n = len(L)
    if n == 0:
        return 0.0
    k = int(math.ceil(n * (1.0 - alpha)))
    k = max(1, min(k, n))
    return sum(L[:k]) / k

def run_cli_and_get_json(tag: str) -> Dict[str, Any]:
    """지정된 파라미터로 CLI 실행 후 JSON 파싱."""
    cmd = [
        sys.executable, "-m", "project.runner.cli",
        "--method", "rl",
        "--market_mode", "bootstrap",
        "--data_profile", "full",
        "--es_mode", "loss",
        "--F_target", "0.60",
        "--rl_q_cap", "0.0042",
        "--rl_epochs", "1",
        "--rl_steps_per_epoch", "512",
        "--rl_n_paths_eval", "200",
        "--seeds", "0",
        "--alpha_mix", "equal",
        "--h_FX", "1",
        "--return_actor", "on",           # eval_WT를 extra로 받기 위함
        "--print_mode", "full",
        "--tag", tag,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)

def compute_from_json_blob(blob: Dict[str, Any], F_target: float = 0.60, alpha: float = 0.95) -> Dict[str, Any]:
    """CLI JSON blob에서 eval_WT를 꺼내 지표 재계산."""
    extra = blob.get("extra") or {}
    wt = extra.get("eval_WT")
    if not isinstance(wt, list) or len(wt) == 0:
        raise SystemExit("extra.eval_WT 가 비어있습니다. CLI를 --return_actor on --print_mode full 로 실행했는지 확인하세요.")

    n = len(wt)
    EW = sum(wt) / n
    ruin = sum(1 for x in wt if x <= 0.0) / n

    # 손실 L = max(F - W_T, 0)
    losses = [max(F_target - float(x), 0.0) for x in wt]

    # 재계산 지표
    var95  = var_alpha(losses, alpha)
    es95_i = es_alpha_interp(losses, alpha)   # CLI와 거의 일치
    es95_s = es_alpha_simple(losses, alpha)   # 참고용(보간 없음)

    # CLI metrics 비교(있으면)
    cli_metrics = blob.get("metrics") or {}
    return {
        "tag": blob.get("tag"),
        "asset": blob.get("asset"),
        "method": blob.get("method"),
        "n_paths_from_json": blob.get("n_paths"),
        "n_paths_from_eval_WT": n,
        "WT_mean": EW,
        "WT_min": min(wt),
        "WT_max": max(wt),
        "Ruin": ruin,
        "F_target": F_target,
        "alpha": alpha,
        "VaR_alpha": var95,
        "ES95_interp": es95_i,
        "ES95_simple": es95_s,
        "CLI_EW": cli_metrics.get("EW"),
        "CLI_ES95": cli_metrics.get("ES95"),
        "CLI_es95_source": cli_metrics.get("es95_source"),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run_cli", "from_json"], default="run_cli",
                    help="run_cli: CLI 실행 후 계산, from_json: 저장된 JSON 파일에서 계산")
    ap.add_argument("--json_path", type=str, default=None, help="--mode from_json 일 때 읽을 JSON 파일 경로")
    ap.add_argument("--tag", type=str, default="cvar_interp_check", help="run_cli 모드에서 CLI --tag")
    ap.add_argument("--F_target", type=float, default=0.60)
    ap.add_argument("--alpha", type=float, default=0.95)
    args = ap.parse_args()

    if args.mode == "run_cli":
        blob = run_cli_and_get_json(tag=args.tag)
    else:
        if not args.json_path:
            raise SystemExit("--mode from_json 인 경우 --json_path 를 지정해야 합니다.")
        with open(args.json_path, "r", encoding="utf-8") as f:
            blob = json.load(f)

    out = compute_from_json_blob(blob, F_target=args.F_target, alpha=args.alpha)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
