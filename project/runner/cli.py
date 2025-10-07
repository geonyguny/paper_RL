# project/runner/cli.py
from __future__ import annotations

import argparse
import json
import os
import re
import math
from typing import Optional, Any, Dict, Iterable

from ..config import (
    CVAR_TARGET_DEFAULT,
    CVAR_TOL_DEFAULT,
    LAMBDA_MIN_DEFAULT,
    LAMBDA_MAX_DEFAULT,
)
from .run import run_once, run_rl
from .calibrate import calibrate_lambda

# --- evaluate: 다양한 배치에 대응(절대 경로 우선) ---
def _import_evaluate():
    candidates = [
        "project.runner.evaluate",  # project/runner/evaluate.py
        "project.evaluate",         # project/evaluate.py
        "project.runner.eval",      # project/runner/eval.py
        "project.eval",             # project/eval.py
    ]
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["evaluate"])
            return getattr(mod, "evaluate")
        except Exception:
            continue
    return None

evaluate = _import_evaluate()  # type: ignore

# --- CVaR 유틸: 있으면 사용, 없으면 내장 폴백 ---
try:
    from ..utils.metrics_utils import terminal_losses, cvar_alpha  # type: ignore
except Exception:
    terminal_losses = None  # type: ignore
    cvar_alpha = None       # type: ignore


def _cvar_fallback(losses: Iterable[float], alpha: float) -> float:
    """보간 포함 CVaR_α (Acerbi–Tasche 표본식).
    손실 L_(1) <= ... <= L_(n)에 대해
      j = floor(n*α), θ = n*α - j
      ES = ((1-θ)*L_(j+1) + sum_{i=j+2}^n L_(i)) / (n*(1-α))
    """
    import numpy as np
    L = np.asarray(list(losses), dtype=float)
    n = L.size
    if n == 0:
        return 0.0

    a = float(alpha)
    # 안전 가드: 0 < α < 1
    a = max(min(a, 1.0 - 1e-12), 1e-12)

    L.sort()                     # 오름차순
    j = int(math.floor(n * a))   # 0-베이스 index
    if j >= n:
        j = n - 1
    theta = n * a - j            # [0,1)
    Lj1 = float(L[j])            # L_(j+1)
    tail_sum = float(L[j + 1 :].sum())  # L_(j+2) .. L_(n)
    ES = ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))
    return float(ES)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--asset", type=str, default="KR")
    p.add_argument("--method", type=str, default="hjb", choices=["hjb", "rl", "rule"])
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--w_max", type=float, default=0.70)
    p.add_argument("--fee_annual", type=float, default=0.004)
    p.add_argument("--horizon_years", type=int, default=35)
    p.add_argument("--alpha", type=float, default=0.95)  # CVaR level
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

    # Hedge (legacy)
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

    # New: Bands / data window / profile / tag
    p.add_argument("--bands", choices=["on", "off"], default="on",
                   help="consumption bands 저장(on/off). off면 _bands 파일 미생성")
    p.add_argument("--data_window", type=str, default=None,
                   help="YYYY-MM:YYYY-MM 형식 (예: 2005-01:2020-12)")
    p.add_argument("--data_profile", choices=["dev", "full"], default=None,
                   help="market_csv 미지정 시 프로파일(dev/full)로 자동 설정")
    p.add_argument("--tag", type=str, default=None,
                   help="실험 태그(로그/metrics에 기록)")

    # === NEW === 자산배분 & 환헤지 옵션 (run.py가 읽어 처리)
    p.add_argument("--alpha_mix", type=str, default=None,
                   help="자산배분 α=(KR,US,Au). 예: 'equal' 또는 '0.33,0.33,0.34'")
    p.add_argument("--alpha_kr", type=float, default=None, help="KR 가중치(개별 지정 시)")
    p.add_argument("--alpha_us", type=float, default=None, help="US 가중치(개별 지정 시)")
    p.add_argument("--alpha_au", type=float, default=None, help="Gold 가중치(개별 지정 시)")
    p.add_argument("--h_FX", type=float, default=None,
                   help="US 주식 환헤지 비율 h∈[0,1] (예: 1=전헤지)")
    p.add_argument("--fx_hedge_cost", type=float, default=None,
                   help="연 환헤지 비용(기본 0.002=0.2%)")

    # --- NEW: 표준출력 제어 ---
    p.add_argument("--print_mode", choices=["full", "metrics", "summary"], default="full",
                   help="stdout 출력 형태: full(원본), metrics(선택 지표만), summary(요약)")
    p.add_argument("--metrics_keys", type=str, default="EW,ES95,Ruin,mean_WT,es_mode",
                   help="print_mode=metrics|summary 에서 노출할 metrics 키(콤마구분)")
    p.add_argument("--no_paths", action="store_true",
                   help="stdout에서 extra.eval_WT/ruin_flags 등 대용량 배열 제거(길이만 남김)")

    p.add_argument("--return_actor", choices=["on", "off"], default="off",
                   help="on이면 run_rl이 (cfg, actor)를 반환하고 CLI가 재평가(CVaR 재계산) 수행")
    return p


# --------------------------
# Helpers
# --------------------------

_WINDOW_RE = re.compile(r"^\d{4}-\d{2}:\d{4}-\d{2}$")


def _apply_data_profile_defaults(args: argparse.Namespace) -> None:
    """--data_profile 지정 & --market_csv 미지정 시 기본 CSV 경로 자동 설정."""
    if getattr(args, "data_profile", None) and not getattr(args, "market_csv", None):
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "market"))
        if args.data_profile == "dev":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_mini.csv"))
        elif args.data_profile == "full":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_full.csv"))


def _validate_args(args: argparse.Namespace) -> None:
    if args.data_window is not None:
        s = str(args.data_window).strip()
        if s and not _WINDOW_RE.match(s):
            raise SystemExit(f"--data_window 형식 오류: '{args.data_window}'. 예: 2005-01:2020-12")
    if args.method in ("hjb", "rule", "rl") and args.market_mode == "bootstrap":
        if not args.market_csv and not args.data_profile:
            raise SystemExit("market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full) 필요.")


# --- 다양한 결과물에서 W_T 추출 ---
def _maybe_extract_WT(candidate: Any) -> Optional[Iterable[float]]:
    if candidate is None:
        return None
    # numpy array 지원
    try:
        import numpy as _np
        if isinstance(candidate, _np.ndarray):
            return candidate.tolist()
    except Exception:
        pass
    # list/tuple
    if isinstance(candidate, (list, tuple)):
        if len(candidate) >= 2 and isinstance(candidate[1], dict):
            wt = _maybe_extract_WT(candidate[1])
            if wt is not None:
                return wt
        if candidate and all(isinstance(x, (int, float)) for x in candidate):
            return candidate  # type: ignore
    # dict
    if isinstance(candidate, dict):
        for k in [
            "eval_WT", "W_T", "WT", "terminal_wealth", "terminal_wealths",
            "paths_WT", "wt_paths", "wealth_terminal", "wealth_T",
        ]:
            if k in candidate and candidate[k] is not None:
                return candidate[k]  # type: ignore
        for k in ["metrics", "extra", "extras", "eval", "payload", "data"]:
            if k in candidate and isinstance(candidate[k], (dict, list, tuple)):
                wt = _maybe_extract_WT(candidate[k])
                if wt is not None:
                    return wt
    # attrs
    for attr in ["eval_WT", "W_T", "WT", "terminal_wealth", "paths_WT", "wealth_T", "terminal_wealths"]:
        try:
            wt = getattr(candidate, attr)
            if wt is not None:
                return wt  # type: ignore
        except Exception:
            pass
    return None


# --- ES95(CVaR) 보정 ---
def _fixup_metrics_with_cvar(args: argparse.Namespace, out: Dict[str, Any]) -> Dict[str, Any]:
    # metrics dict 위치 결정
    metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    F_target = float(getattr(args, "F_target", 0.0) or 0.0)
    alpha_v = float(getattr(args, "alpha", 0.95) or 0.95)

    # wealth 모드에서는 재계산 대상 아님
    if es_mode != "loss" or F_target <= 0.0:
        metrics["es_mode"] = es_mode
        return out

    WT = None
    for cand in (out, metrics):
        WT = _maybe_extract_WT(cand)
        if WT is not None:
            break

    if WT is None:
        # 의심 패턴(EW+ES95 ≈ F_target) 감지 시 메시지
        try:
            EW = float(metrics.get("EW", 0.0) or metrics.get("mean_WT", 0.0) or 0.0)
            ES_old = float(metrics.get("ES95", 0.0) or 0.0)
            if abs((EW + ES_old) - F_target) < 1e-9 and ES_old > 0:
                metrics["es95_note"] = "ES95 looks like (F_target - EW). No W_T to recompute; please expose path-level W_T from evaluate."
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        return out

    # CVaR 재계산
    try:
        import numpy as np
        WT_arr = np.asarray(list(WT), dtype=float)
        if terminal_losses is not None and cvar_alpha is not None:
            L = terminal_losses(WT_arr, F_target)
            ES = cvar_alpha(L, alpha=alpha_v)
        else:
            L = np.maximum(F_target - WT_arr, 0.0)
            ES = _cvar_fallback(L, alpha=alpha_v)

        EW = float(metrics.get("EW", 0.0))
        if EW == 0.0:
            EW = float(metrics.get("mean_WT", 0.0) or float(np.mean(WT_arr)))
        metrics["EW"] = EW
        metrics["mean_WT"] = EW
        metrics["ES95"] = float(ES)
        try:
            metrics["Ruin"] = float((WT_arr <= 0.0).mean())
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        metrics["es95_source"] = "path_level_cvar"
    except Exception as e:
        metrics["es_mode"] = es_mode
        metrics["es95_note"] = f"failed to recompute ES95: {type(e).__name__}"
    return out


# --- n_paths 추정 (결과에 없을 때) ---
def _estimate_n_paths(args: argparse.Namespace, out: Dict[str, Any]) -> Optional[int]:
    try:
        if isinstance(out, dict):
            np_exist = out.get("n_paths")
            if isinstance(np_exist, (int, float)) and int(np_exist) > 0:
                return int(np_exist)
        # seeds 처리
        seeds = getattr(args, "seeds", [])
        if isinstance(seeds, int):
            n_seeds = 1
        elif isinstance(seeds, (list, tuple)):
            n_seeds = max(1, len(seeds))
        else:
            n_seeds = 1
        # 방법별 평가 경로
        if str(getattr(args, "method", "hjb")).lower() == "rl":
            n_eval = int(getattr(args, "rl_n_paths_eval", 0) or 0)
            if n_eval > 0:
                return n_eval * n_seeds
        # 기본 경로
        n_base = int(getattr(args, "n_paths", 0) or 0)
        if n_base > 0:
            return n_base * n_seeds
    except Exception:
        pass
    return None


# --- 표준출력 축소/요약 도우미 ---
def _prune_for_stdout(args: argparse.Namespace, out: Dict[str, Any]) -> Any:
    def _sel_metrics(md: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
        return {k: md[k] for k in keys if k in md}

    # 대용량 경로 제거 옵션
    if args.no_paths and isinstance(out, dict):
        out = dict(out)  # shallow copy
        extra = out.get("extra")
        if isinstance(extra, dict):
            for k in ("eval_WT", "ruin_flags"):
                if k in extra and isinstance(extra[k], (list, tuple)):
                    try:
                        extra[k + "_n"] = len(extra[k])
                    except Exception:
                        pass
                    del extra[k]
            out["extra"] = extra

    mode = getattr(args, "print_mode", "full")
    if mode == "full":
        return out

    # metrics 사전 찾기
    metrics = out["metrics"] if isinstance(out, dict) and isinstance(out.get("metrics"), dict) else out
    keys = [s.strip() for s in str(getattr(args, "metrics_keys", "")).split(",") if s.strip()]

    # 공통 메타 필드 추정
    n_paths_guess = _estimate_n_paths(args, out)
    age0_guess = None
    sex_guess = None
    try:
        if isinstance(out, dict):
            age0_guess = out.get("age0")
            sex_guess = out.get("sex")
        if age0_guess is None:
            age0_guess = getattr(args, "age0", None)
        if sex_guess is None:
            sex_guess = getattr(args, "sex", None)
    except Exception:
        pass

    if mode == "metrics":
        mini = _sel_metrics(metrics, keys)
        mini.update({
            "tag": out.get("tag") if isinstance(out, dict) else None,
            "asset": out.get("asset") if isinstance(out, dict) else None,
            "method": out.get("method") if isinstance(out, dict) else None,
            "n_paths": n_paths_guess,
        })
        return mini

    if mode == "summary":
        args_dict = out.get("args", {}) if isinstance(out, dict) else {}
        return {
            "tag": out.get("tag") if isinstance(out, dict) else None,
            "asset": out.get("asset") if isinstance(out, dict) else None,
            "method": out.get("method") if isinstance(out, dict) else None,
            "age0": age0_guess if age0_guess is not None else (args_dict or {}).get("age0"),
            "sex": sex_guess if sex_guess is not None else (args_dict or {}).get("sex"),
            "metrics": _sel_metrics(metrics, keys),
            "n_paths": n_paths_guess,
            "T": (out.get("extra") or {}).get("T") if isinstance(out, dict) else None,
        }

    return out  # 안전망


def _maybe_evaluate_with_es_mode(res: Any, es_mode: str) -> Dict[str, Any]:
    """run_* 결과가 dict면 그대로 사용.
    (cfg, actor) 형태면 evaluate(cfg, actor, es_mode=...) 호출 → dict로 변환.
    """
    # 이미 최종 dict
    if isinstance(res, dict):
        if "es_mode" not in res:
            res["es_mode"] = es_mode
        return res

    # evaluate 경로
    if evaluate is not None:
        cfg: Optional[Any] = None
        actor: Optional[Any] = None

        if isinstance(res, tuple) and len(res) >= 2:
            cfg, actor = res[0], res[1]
        else:
            cfg = getattr(res, "cfg", None)
            actor = getattr(res, "actor", None)

        if cfg is not None and actor is not None:
            try:
                m = evaluate(cfg, actor, es_mode=str(es_mode).lower(), return_paths=True)  # type: ignore
            except TypeError:
                m = evaluate(cfg, actor, es_mode=str(es_mode).lower())  # type: ignore

            # (metrics, extras) or dict
            if isinstance(m, tuple) and len(m) >= 1 and isinstance(m[0], dict):
                pack: Dict[str, Any] = {"metrics": m[0]}
                if len(m) >= 2:
                    pack["extra"] = m[1]  # extras['eval_WT'] 등 보관
                if "es_mode" not in pack["metrics"]:
                    pack["metrics"]["es_mode"] = str(es_mode).lower()
                # cfg에서 몇 가지 편의 필드 복사(있을 때만)
                for k in ("asset", "method", "w_max", "fee_annual", "lambda_term", "alpha", "F_target", "outputs", "tag"):
                    try:
                        pack[k] = getattr(cfg, k)
                    except Exception:
                        pass
                return pack

            if isinstance(m, dict):
                if "es_mode" not in m:
                    m["es_mode"] = str(es_mode).lower()
                return m

    # evaluate 사용 불가 → 최소 정보 래핑
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

    _apply_data_profile_defaults(args)
    _validate_args(args)

    # route
    if args.method == "rl":
        res = run_rl(args)
        out = _maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"))
    elif args.method == "hjb" and (args.cvar_target is not None):
        out = calibrate_lambda(args)
        if isinstance(out, dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(args, "es_mode", "wealth")).lower()
    else:
        res = run_once(args)
        out = _maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"))

    # ES95(CVaR) 보정
    try:
        if isinstance(out, dict):
            out = _fixup_metrics_with_cvar(args, out)
    except Exception as _e:
        try:
            if isinstance(out, dict):
                tgt = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
                tgt["es95_note"] = f"post-fixup failed: {type(_e).__name__}"
        except Exception:
            pass

    # 표준출력 제어 적용
    to_print = _prune_for_stdout(args, out) if isinstance(out, dict) else out
    print(json.dumps(to_print, ensure_ascii=False))


if __name__ == "__main__":
    main()
