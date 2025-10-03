# project/runner/cli.py
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional, Any, Dict, Tuple

from ..config import (
    CVAR_TARGET_DEFAULT,
    CVAR_TOL_DEFAULT,
    LAMBDA_MIN_DEFAULT,
    LAMBDA_MAX_DEFAULT,
)
from .run import run_once, run_rl
from .calibrate import calibrate_lambda

# evaluate 위치가 프로젝트마다 다를 수 있어 유연하게 import 시도
try:  # 권장 경로
    from .evaluate import evaluate  # type: ignore
except Exception:
    try:  # 대안 경로
        from .eval import evaluate  # type: ignore
    except Exception:
        evaluate = None  # evaluate가 없으면 런타임에 dict 반환만 신뢰


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--asset", type=str, default="KR")
    p.add_argument("--method", type=str, default="hjb", choices=["hjb", "rl", "rule"])
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--w_max", type=float, default=0.70)
    p.add_argument("--fee_annual", type=float, default=0.004)
    p.add_argument("--horizon_years", type=int, default=35)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--lambda_term", type=float, default=0.0)
    p.add_argument("--F_target", type=float, default=0.0)
    p.add_argument("--p_annual", type=float, default=0.04)
    p.add_argument("--g_real_annual", type=float, default=0.02)
    p.add_argument("--w_fixed", type=float, default=0.60)
    p.add_argument("--floor_on", action="store_true")
    p.add_argument("--f_min_real", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--es_mode", type=str, default="wealth", choices=["wealth", "loss"])
    p.add_argument("--outputs", type=str, default="./outputs")

    # HJB
    p.add_argument("--hjb_W_grid", type=int, default=None)
    p.add_argument("--hjb_Nshock", type=int, default=None)
    p.add_argument("--hjb_eta_n", type=int, default=None)

    # Hedge
    p.add_argument("--hedge", choices=["on", "off"], default="off")
    p.add_argument("--hedge_mode", choices=["mu", "sigma", "downside"], default="sigma")
    p.add_argument("--hedge_cost", type=float, default=0.005)
    p.add_argument("--hedge_sigma_k", type=float, default=0.20)
    p.add_argument("--hedge_tx", type=float, default=0.0)

    # Market
    p.add_argument("--market_mode", choices=["iid", "bootstrap"], default="iid")
    p.add_argument("--market_csv", type=str, default=None)
    p.add_argument("--bootstrap_block", type=int, default=24)
    p.add_argument("--use_real_rf", choices=["on", "off"], default="on")

    # Mortality
    p.add_argument("--mortality", choices=["on", "off"], default="off")
    p.add_argument("--mort_table", type=str, default=None)
    p.add_argument("--age0", type=int, default=65)
    p.add_argument("--sex", choices=["M", "F"], default="M")
    p.add_argument("--bequest_kappa", type=float, default=0.0)
    p.add_argument("--bequest_gamma", type=float, default=1.0)

    # CVaR calibration
    p.add_argument("--cvar_target", type=float, default=CVAR_TARGET_DEFAULT)
    p.add_argument("--cvar_tol", type=float, default=CVAR_TOL_DEFAULT)
    p.add_argument("--lambda_min", type=float, default=LAMBDA_MIN_DEFAULT)
    p.add_argument("--lambda_max", type=float, default=LAMBDA_MAX_DEFAULT)
    p.add_argument("--calib_fast", choices=["on", "off"], default="on")
    p.add_argument("--calib_max_iter", type=int, default=8)

    # autosave
    p.add_argument("--autosave", choices=["on", "off"], default="off")

    # RL
    p.add_argument("--rl_epochs", type=int, default=60)
    p.add_argument("--rl_steps_per_epoch", type=int, default=2048)
    p.add_argument("--rl_n_paths_eval", type=int, default=300)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--rl_q_cap", type=float, default=0.0)
    p.add_argument("--teacher_eps0", type=float, default=0.0)
    p.add_argument("--teacher_decay", type=float, default=1.0)
    p.add_argument("--lw_scale", type=float, default=0.0)
    p.add_argument("--survive_bonus", type=float, default=0.0)
    p.add_argument("--crra_gamma", type=float, default=3.0)
    p.add_argument("--u_scale", type=float, default=0.0)

    # Lite overrides
    p.add_argument("--q_floor", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)

    # Stage-wise CVaR
    p.add_argument("--cvar_stage", choices=["on", "off"], default="off")
    p.add_argument("--alpha_stage", type=float, default=0.95)
    p.add_argument("--lambda_stage", type=float, default=0.0)
    p.add_argument("--cstar_mode", choices=["fixed", "annuity", "vpw"], default="annuity")
    p.add_argument("--cstar_m", type=float, default=0.04 / 12)

    # XAI
    p.add_argument("--xai_on", choices=["on", "off"], default="on")

    # QUIET
    p.add_argument("--quiet", choices=["on", "off"], default="on")

    # [ANN] Overlay (MVP)
    p.add_argument("--ann_on", choices=["on", "off"], default="off")
    p.add_argument("--ann_alpha", type=float, default=0.0)
    p.add_argument("--ann_L", type=float, default=0.0)
    p.add_argument("--ann_d", type=int, default=0)
    p.add_argument("--ann_index", choices=["real", "nominal"], default="real")

    # New: Bands 저장 토글 / 데이터 윈도우 / 프로파일 / 태그
    p.add_argument(
        "--bands", choices=["on", "off"], default="on",
        help="consumption bands 저장(on/off). off면 _bands 파일 미생성",
    )
    p.add_argument(
        "--data_window", type=str, default=None,
        help="YYYY-MM:YYYY-MM 형식 기간 슬라이스 (예: 2005-01:2020-12)",
    )
    p.add_argument(
        "--data_profile", choices=["dev", "full"], default=None,
        help="market_csv 미지정 시 기본 CSV 경로를 프로파일로 자동 설정(dev/full)",
    )
    p.add_argument(
        "--tag", type=str, default=None,
        help="실험 태그(로그/metrics에 기록)",
    )

    return p


# --------------------------
# Helpers
# --------------------------

_WINDOW_RE = re.compile(r"^\d{4}-\d{2}:\d{4}-\d{2}$")


def _apply_data_profile_defaults(args: argparse.Namespace) -> None:
    """
    --data_profile 지정 & --market_csv 미지정 시,
    project/data/market 하위의 기본 CSV 경로를 자동 설정.
    """
    if getattr(args, "data_profile", None) and not getattr(args, "market_csv", None):
        base = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "market")
        )
        if args.data_profile == "dev":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_mini.csv"))
        elif args.data_profile == "full":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_full.csv"))


def _validate_args(args: argparse.Namespace) -> None:
    # data_window 형식 체크
    if args.data_window is not None:
        s = str(args.data_window).strip()
        if s and not _WINDOW_RE.match(s):
            raise SystemExit(
                f"--data_window 형식 오류: '{args.data_window}'. "
                "예: 2005-01:2020-12"
            )
    # bootstrap인데 csv 미지정이면 사용자 친화적 에러
    if args.method in ("hjb", "rule", "rl") and args.market_mode == "bootstrap":
        if not args.market_csv and not args.data_profile:
            raise SystemExit(
                "market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full) 중 하나는 필수입니다."
            )


def _maybe_evaluate_with_es_mode(
    res: Any,
    es_mode: str,
) -> Dict[str, Any]:
    """
    run_once/run_rl 결과가 dict면 그대로,
    (cfg, actor) 형태면 evaluate(cfg, actor, es_mode=...)를 호출해 metrics dict로 변환.
    evaluate를 import하지 못한 경우엔 가능한 정보를 dict로 래핑.
    """
    # 이미 최종 아웃풋(dict)인 경우 그대로 반환
    if isinstance(res, dict):
        # 기록 상 es_mode가 누락/불일치하면 표시만 보정
        if "es_mode" not in res:
            res["es_mode"] = es_mode
        return res

    # evaluate 사용 가능한 경우만 처리
    if evaluate is not None:
        # (cfg, actor) or 객체 형태 유연 처리
        cfg: Optional[Any] = None
        actor: Optional[Any] = None

        if isinstance(res, tuple) and len(res) >= 2:
            cfg, actor = res[0], res[1]
        else:
            # res가 네임드 객체라면 속성 추출 시도
            cfg = getattr(res, "cfg", None)
            actor = getattr(res, "actor", None)

        if cfg is not None and actor is not None:
            metrics = evaluate(cfg, actor, es_mode=str(es_mode).lower())
            # evaluate가 metrics dict를 반환한다고 가정
            if isinstance(metrics, dict):
                if "es_mode" not in metrics:
                    metrics["es_mode"] = str(es_mode).lower()
                return metrics

    # 여기까지 왔다면 evaluate를 못 쓰는 상황 -> 정보 보존용 래핑
    return {
        "result": "ok",
        "note": "evaluate not executed in cli (no evaluate import or unexpected return type).",
        "es_mode": str(es_mode).lower(),
    }


# --------------------------
# Main
# --------------------------

def main():
    p = _build_arg_parser()
    args = p.parse_args()

    # Default market_csv from data_profile when not provided
    _apply_data_profile_defaults(args)
    _validate_args(args)

    # Route by method
    if args.method == "rl":
        res = run_rl(args)
        out = _maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"))
    elif args.method == "hjb" and (args.cvar_target is not None):
        out = calibrate_lambda(args)
        # 보정: HJB 경로도 출력 dict에 es_mode 표기 일관화
        if isinstance(out, dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(args, "es_mode", "wealth")).lower()
    else:
        res = run_once(args)
        out = _maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"))

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
