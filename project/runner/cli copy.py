# project/runner/cli.py
from __future__ import annotations

import argparse
import json
import os
import re
import math
import time
import sys
from typing import Optional, Any, Dict, Iterable, Tuple, List

from ..config import (
    CVAR_TARGET_DEFAULT,
    CVAR_TOL_DEFAULT,
    LAMBDA_MIN_DEFAULT,
    LAMBDA_MAX_DEFAULT,
)
from .run import run_once, run_rl
from .calibrate import calibrate_lambda  # ← 캘리브레이션 엔트리포인트

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
    """보간 포함 CVaR_α (Acerbi–Tasche 표본식)."""
    import numpy as np
    L = np.asarray(list(losses), dtype=float)
    n = L.size
    if n == 0:
        return 0.0
    a = float(alpha)
    a = max(min(a, 1.0 - 1e-12), 1e-12)  # 0<α<1 가드
    L.sort()                     # 오름차순
    j = int(math.floor(n * a))   # 0-베이스 index
    if j >= n:
        j = n - 1
    theta = n * a - j            # [0,1)
    Lj1 = float(L[j])            # L_(j+1)
    tail_sum = float(L[j + 1 :].sum())  # L_(j+2) .. L_(n)
    ES = ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))
    return float(ES)


def _fmt_hms(sec: float) -> str:
    try:
        total = int(round(float(sec)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"


def _parse_hms_to_seconds(hms: Optional[str]) -> Optional[float]:
    if not hms:
        return None
    s = str(hms).strip()
    try:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = map(int, parts)
            return float(h) * 3600 + float(m) * 60 + float(sec)
        if len(parts) == 2:  # mm:ss
            m, sec = map(int, parts)
            return float(m) * 60 + float(sec)
        # 숫자만 주면 초로 해석
        return float(s)
    except Exception:
        return None


# ==========================
# ETA 히스토리/추정 유틸
# ==========================
def _eta_db_path(args: argparse.Namespace) -> str:
    default = os.path.join(getattr(args, "outputs", "./outputs"), ".eta_history.json")
    return getattr(args, "eta_db", default) or default

def _eta_load_db(path: str) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def _eta_save_db(path: str, rows: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows[-500:], f, ensure_ascii=False, indent=0)  # 최근 500건만 유지
    except Exception:
        pass

def _seeds_count(args: argparse.Namespace) -> int:
    s = getattr(args, "seeds", [])
    if isinstance(s, int):
        return 1
    if isinstance(s, (list, tuple)):
        return max(1, len(s))
    return 1

def _eta_signature(args: argparse.Namespace) -> Dict[str, Any]:
    # 예측에 중요할만한 항목만 선택
    return {
        "method": getattr(args, "method", None),
        "market_mode": getattr(args, "market_mode", None),
        "data_profile": getattr(args, "data_profile", None),
        "asset": getattr(args, "asset", None),
        "es_mode": getattr(args, "es_mode", None),

        # 공통 규모지표
        "n_paths": getattr(args, "n_paths", None),
        "seeds": _seeds_count(args),

        # RL 규모지표
        "rl_epochs": getattr(args, "rl_epochs", None),
        "rl_steps_per_epoch": getattr(args, "rl_steps_per_epoch", None),
        "rl_n_paths_eval": getattr(args, "rl_n_paths_eval", None),

        # HJB 프리셋(있으면)
        "hjb_W_grid": getattr(args, "hjb_W_grid", None),
        "hjb_Nshock": getattr(args, "hjb_Nshock", None),
        "hjb_eta_n": getattr(args, "hjb_eta_n", None),
    }

def _eta_match_score(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    # 간단 가중치 매칭 점수(완전 동일 필드 수)
    keys = ["method", "market_mode", "data_profile", "asset", "es_mode"]
    score = sum(1 for k in keys if a.get(k) == b.get(k))
    return score

def _eta_predict_from_history(args: argparse.Namespace, db: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    sig = _eta_signature(args)
    # 후보: 동일 메소드 + 같은 데이터 프로파일 중심으로 최근 50건
    candidates: List[Dict[str, Any]] = []
    for row in reversed(db):
        r_sig = row.get("sig", {})
        if r_sig.get("method") != sig.get("method"):
            continue
        if r_sig.get("data_profile") != sig.get("data_profile"):
            continue
        candidates.append(row)
        if len(candidates) >= 50:
            break

    if not candidates:
        return None, "no_history"

    # 최근 최고 매칭을 하나 고르고, 규모 비율로 스케일
    # 휴리스틱: RL은 step/epoch/seed 합성 규모, HJB는 n_paths*seeds
    best = max(candidates, key=lambda r: _eta_match_score(sig, r.get("sig", {})))
    base_time = float(best.get("time_total_s", 0.0) or 0.0)
    if base_time <= 0:
        return None, "bad_history_base"

    ref = best.get("sig", {})
    method = sig.get("method")

    try:
        if method == "rl":
            seeds = max(1, int(sig.get("seeds") or 1))
            ref_seeds = max(1, int(ref.get("seeds") or 1))

            steps = max(1, int(sig.get("rl_epochs") or 1) * int(sig.get("rl_steps_per_epoch") or 1))
            ref_steps = max(1, int(ref.get("rl_epochs") or 1) * int(ref.get("rl_steps_per_epoch") or 1))

            paths = max(1, int(sig.get("rl_n_paths_eval") or 1) * seeds)
            ref_paths = max(1, int(ref.get("rl_n_paths_eval") or 1) * ref_seeds)

            # 단순 선형 스케일 + 혼합(학습 80%, 평가 20%)
            scale_train = (steps * seeds) / max(1.0, ref_steps * ref_seeds)
            scale_eval  = paths / max(1.0, ref_paths)
            eta = base_time * (0.8 * scale_train + 0.2 * scale_eval)
            return max(1.0, float(eta)), "history_rl"

        # HJB / rule
        seeds = max(1, int(sig.get("seeds") or 1))
        ref_seeds = max(1, int(ref.get("seeds") or 1))
        n_paths = int(sig.get("n_paths") or 0)
        ref_paths = int(ref.get("n_paths") or 0)
        scale = 1.0
        if n_paths > 0 and ref_paths > 0:
            scale = (n_paths * seeds) / (ref_paths * ref_seeds)
        eta = base_time * scale
        return max(1.0, float(eta)), "history_hjb"
    except Exception:
        return None, "scale_error"

def _eta_record(args: argparse.Namespace, elapsed_s: float) -> None:
    try:
        path = _eta_db_path(args)
        db = _eta_load_db(path)
        db.append({
            "ts": time.time(),
            "time_total_s": float(elapsed_s),
            "sig": _eta_signature(args),
        })
        _eta_save_db(path, db)
    except Exception:
        pass


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

    # === CVaR Calibration ===
    p.add_argument("--calib", choices=["on", "off"], default="off",
                   help="on이면 캘리브레이션 모드로 진입(calibrate_lambda).")
    p.add_argument("--calib_param", choices=["lambda", "F"], default="lambda",
                   help="캘리브레이션 대상 파라미터 선택.")
    p.add_argument("--cvar_target", type=float, default=CVAR_TARGET_DEFAULT)
    p.add_argument("--cvar_tol", type=float, default=CVAR_TOL_DEFAULT)
    p.add_argument("--lambda_min", type=float, default=LAMBDA_MIN_DEFAULT)
    p.add_argument("--lambda_max", type=float, default=LAMBDA_MAX_DEFAULT)
    p.add_argument("--calib_fast", choices=["on", "off"], default="on")
    p.add_argument("--calib_max_iter", type=int, default=8)
    p.add_argument("--F_min", type=float, default=None)
    p.add_argument("--F_max", type=float, default=None)

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
    p.add_argument("--bands", choices=["on", "off"], default="on")
    p.add_argument("--data_window", type=str, default=None)
    p.add_argument("--data_profile", choices=["dev", "full"], default=None)
    p.add_argument("--tag", type=str, default=None)

    # === 자산배분 & 환헤지 옵션
    p.add_argument("--alpha_mix", type=str, default=None)
    p.add_argument("--alpha_kr", type=float, default=None)
    p.add_argument("--alpha_us", type=float, default=None)
    p.add_argument("--alpha_au", type=float, default=None)
    p.add_argument("--h_FX", type=float, default=None)
    p.add_argument("--fx_hedge_cost", type=float, default=None)

    # --- 표준출력 제어 ---
    p.add_argument("--print_mode", choices=["full", "metrics", "summary"], default="full")
    p.add_argument("--metrics_keys", type=str, default="EW,ES95,Ruin,mean_WT,es_mode")
    p.add_argument("--no_paths", action="store_true")

    p.add_argument("--return_actor", choices=["on", "off"], default="off")

    # --- ETA 옵션 ---
    p.add_argument("--eta_mode", choices=["off", "history"], default="history",
                   help="실행 전 예상시간(ETA) 추정 모드. off면 비활성화")
    p.add_argument("--eta_budget_hms", type=str, default=None,
                   help="예산 시간(HH:MM:SS). 예측 ETA가 이를 넘으면 실행 중단(기본 hard stop).")
    p.add_argument("--eta_budget_s", type=float, default=None,
                   help="예산 시간(초). eta_budget_hms보다 우선.")
    p.add_argument("--eta_hard_stop", choices=["on", "off"], default="on",
                   help="on=예산 초과 시 즉시 종료, off=경고만")
    p.add_argument("--eta_db", type=str, default=None,
                   help="ETA 히스토리 DB 경로(기본 outputs/.eta_history.json)")

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
        if len(candidate) >= 2 and isinstance(candidate[1], (dict, list, tuple)):
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
        for k in ["metrics", "extra", "extras", "eval", "payload", "data", "result"]:
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


# --- F_target 결정(캘리브 결과 우선) ---
def _resolve_F_for_cvar(args: argparse.Namespace, out: Dict[str, Any]) -> float:
    # 1) 캘리브 선택 F (필드명 양쪽 호환)
    try:
        cc = out.get("cvar_calibration", {}) if isinstance(out, dict) else {}
        sf = cc.get("selected_F_target")
        if sf is None:
            sf = cc.get("F_selected")
        if isinstance(sf, (int, float)):
            return float(sf)
    except Exception:
        pass
    # 2) 결과 상단의 F_target
    try:
        ft = out.get("F_target") if isinstance(out, dict) else None
        if isinstance(ft, (int, float)):
            return float(ft)
    except Exception:
        pass
    # 3) CLI 인자 최후
    try:
        return float(getattr(args, "F_target", 0.0) or 0.0)
    except Exception:
        return 0.0


# --- ES95(CVaR) 보정 ---
def _fixup_metrics_with_cvar(args: argparse.Namespace, out: Dict[str, Any]) -> Dict[str, Any]:
    metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    F_target = _resolve_F_for_cvar(args, out if isinstance(out, dict) else {})
    alpha_v = float(getattr(args, "alpha", 0.95) or 0.95)

    if es_mode != "loss" or F_target <= 0.0:
        metrics["es_mode"] = es_mode
        return out

    WT = None
    for cand in (out, metrics):
        WT = _maybe_extract_WT(cand)
        if WT is not None:
            break

    if WT is None:
        try:
            EW = float(metrics.get("EW", 0.0) or metrics.get("mean_WT", 0.0) or 0.0)
            ES_old = float(metrics.get("ES95", 0.0) or 0.0)
            if abs((EW + ES_old) - F_target) < 1e-9 and ES_old > 0:
                metrics["es95_note"] = "ES95 looks like (F_target - EW). No W_T to recompute; please expose path-level W_T from evaluate."
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        return out

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
        seeds = getattr(args, "seeds", [])
        if isinstance(seeds, int):
            n_seeds = 1
        elif isinstance(seeds, (list, tuple)):
            n_seeds = max(1, len(seeds))
        else:
            n_seeds = 1
        if str(getattr(args, "method", "hjb")).lower() == "rl":
            n_eval = int(getattr(args, "rl_n_paths_eval", 0) or 0)
            if n_eval > 0:
                return n_eval * n_seeds
        n_base = int(getattr(args, "n_paths", 0) or 0)
        if n_base > 0:
            return n_base * n_seeds
    except Exception:
        pass
    return None


# --- 표준출력 축소/요약 도우미 ---
def _prune_for_stdout(args: argparse.Namespace, out: Dict[str, Any]) -> Any:
    def _sel_metrics(md: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
        return {k: md[k] for k in md if k in keys}

    if args.no_paths and isinstance(out, dict):
        out = dict(out)
        extra = out.get("extra")
        if isinstance(extra, dict):
            for k in ("eval_WT", "ruin_flags"):
                if k in extra and isinstance(extra[k], (list, tuple)):
                    try:
                        extra[k] = []  # 실제 배열을 제거(요약엔 길이만 표시됨)
                        extra[k + "_n"] = int(extra.get(k + "_n", 0))
                    except Exception:
                        pass
            out["extra"] = extra

    mode = getattr(args, "print_mode", "full")
    if mode == "full":
        return out

    metrics = out["metrics"] if isinstance(out, dict) and isinstance(out.get("metrics"), dict) else out
    keys = [s.strip() for s in str(getattr(args, "metrics_keys", "")).split(",") if s.strip()]

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
        if isinstance(out, dict):
            mini["time_total_s"] = out.get("time_total_s")
            mini["time_total_hms"] = out.get("time_total_hms")
        mini.update({
            "tag": out.get("tag") if isinstance(out, dict) else getattr(args, "tag", None),
            "asset": out.get("asset") if isinstance(out, dict) else getattr(args, "asset", None),
            "method": out.get("method") if isinstance(out, dict) else getattr(args, "method", None),
            "n_paths": n_paths_guess,
        })
        return mini

    if mode == "summary":
        args_dict = out.get("args", {}) if isinstance(out, dict) else {}
        top_tag    = (out.get("tag") if isinstance(out, dict) else None) or getattr(args, "tag", None)
        top_method = (out.get("method") if isinstance(out, dict) else None) or getattr(args, "method", None)
        top_asset  = (out.get("asset") if isinstance(out, dict) else None) or getattr(args, "asset", None)

        summary_obj = {
            "tag": top_tag,
            "asset": top_asset,
            "method": top_method,
            "age0": age0_guess if age0_guess is not None else (args_dict or {}).get("age0"),
            "sex": sex_guess if sex_guess is not None else (args_dict or {}).get("sex"),
            "metrics": _sel_metrics(metrics, keys),
            "n_paths": n_paths_guess,
            "T": (out.get("extra") or {}).get("T") if isinstance(out, dict) else None,
            "time_total_s": out.get("time_total_s") if isinstance(out, dict) else None,
            "time_total_hms": out.get("time_total_hms") if isinstance(out, dict) else None,
        }
        # 요약에도 캘리브레이션 섹션 포함(있을 때만)
        if isinstance(out, dict) and isinstance(out.get("cvar_calibration"), dict):
            summary_obj["cvar_calibration"] = out["cvar_calibration"]
        return summary_obj

    return out  # 안전망


# ---------- cfg/actor 추출 (경로 재평가용) ----------
def _try_extract_cfg_actor(res: Any) -> Tuple[Optional[Any], Optional[Any]]:
    # tuple (cfg, actor) 스타일
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    # 객체 속성
    cfg = getattr(res, "cfg", None)
    actor = getattr(res, "actor", None) or getattr(res, "policy", None)
    if cfg is not None or actor is not None:
        return cfg, actor
    # dict 스타일
    if isinstance(res, dict):
        cfg = res.get("args") or res.get("cfg")
        actor = res.get("actor") or res.get("policy") or res.get("pi")
        if cfg is not None or actor is not None:
            return cfg, actor
    return None, None


# --- 평가 결과 패킹(+필요 시 경로 재평가) ---
def _maybe_evaluate_with_es_mode(res: Any, es_mode: str, want_paths: bool = False) -> Dict[str, Any]:
    """
    기본 패킹 + 필요시 evaluate(...) 재호출하여 paths 생성.
    want_paths=True 이면 extras.eval_WT를 가능하면 채움.
    """
    # 1) (metrics[, extra]) 튜플
    if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], dict):
        pack: Dict[str, Any] = {"metrics": dict(res[0])}
        if len(res) >= 2 and isinstance(res[1], dict):
            pack["extra"] = dict(res[1])
        if "es_mode" not in pack["metrics"]:
            pack["metrics"]["es_mode"] = str(es_mode).lower()
    # 2) 최종 dict
    elif isinstance(res, dict):
        if "metrics" in res and isinstance(res["metrics"], dict):
            pack = {"metrics": dict(res["metrics"])}
            if isinstance(res.get("extra"), dict):
                pack["extra"] = dict(res["extra"])
            # 기타 상위 메타 필드 유지
            for k in ("asset","method","w_max","fee_annual","lambda_term","alpha","F_target","outputs","tag","n_paths","args"):
                if k in res:
                    pack[k] = res[k]
            if "es_mode" not in pack["metrics"]:
                pack["metrics"]["es_mode"] = str(es_mode).lower()
        else:
            # 평탄 dict인 경우 metrics로 승격
            pack = {"metrics": dict(res)}
            if "es_mode" not in pack["metrics"]:
                pack["metrics"]["es_mode"] = str(es_mode).lower()
    else:
        pack = {
            "result": "ok",
            "note": "evaluate not executed in cli (no evaluate import or unexpected return type).",
            "es_mode": str(es_mode).lower(),
        }

    # 이미 WT가 들어있으면 경로 충분
    have_paths = False
    try:
        wt0 = _maybe_extract_WT({"metrics": pack.get("metrics", {}), "extra": pack.get("extra", {})})
        have_paths = wt0 is not None and len(list(wt0)) > 0
    except Exception:
        have_paths = False

    # --- 경로 필요하고 없을 때만 evaluate 시도 ---
    if want_paths and not have_paths and evaluate is not None:
        cfg, actor = _try_extract_cfg_actor(res)
        if cfg is None:
            cfg, actor = _try_extract_cfg_actor(pack)
        if cfg is not None:
            try:
                m = None
                try:
                    m = evaluate(cfg, actor, es_mode=str(es_mode).lower(), return_paths=True)  # type: ignore
                except TypeError:
                    m = evaluate(cfg, actor, es_mode=str(es_mode).lower())  # type: ignore

                if isinstance(m, tuple) and len(m) >= 1 and isinstance(m[0], dict):
                    pack["metrics"] = m[0]
                    if len(m) >= 2 and isinstance(m[1], dict):
                        pack["extra"] = m[1]
                elif isinstance(m, dict):
                    pack["metrics"] = m

                if "es_mode" not in pack.get("metrics", {}):
                    pack["metrics"]["es_mode"] = str(es_mode).lower()

                wt_paths = _maybe_extract_WT({"metrics": pack.get("metrics", {}), "extra": pack.get("extra", {})})
                if wt_paths is not None:
                    if "extra" not in pack or not isinstance(pack["extra"], dict):
                        pack["extra"] = {}
                    pack["extra"]["eval_WT"] = list(wt_paths)
            except Exception:
                # evaluate 실패시 조용히 폴백
                pass

    return pack


# --------------------------
# Main
# --------------------------

def main():
    # 전체 실행시간(벽시계) 측정 시작
    t0 = time.perf_counter()

    p = _build_arg_parser()
    args = p.parse_args()

    _apply_data_profile_defaults(args)
    _validate_args(args)

    # === ETA: 실행 전 예측 & 예산 검사 (STDERR로만 출력) ===
    try:
        if getattr(args, "eta_mode", "history") == "history":
            db = _eta_load_db(_eta_db_path(args))
            eta_s, src = _eta_predict_from_history(args, db)
            if eta_s is not None:
                print(f"[ETA] ~{_fmt_hms(eta_s)} (source={src}) … starting", file=sys.stderr, flush=True)

                # 예산 체크
                budget = getattr(args, "eta_budget_s", None)
                if budget is None:
                    budget = _parse_hms_to_seconds(getattr(args, "eta_budget_hms", None))
                if budget is not None and eta_s > float(budget):
                    if str(getattr(args, "eta_hard_stop", "on")).lower() == "on":
                        print(f"[ETA] exceeds budget {_fmt_hms(budget)} → abort.", file=sys.stderr, flush=True)
                        sys.exit(3)
                    else:
                        print(f"[ETA] exceeds budget {_fmt_hms(budget)} → continue (soft-warn).",
                              file=sys.stderr, flush=True)
            # 히스토리 없으면 조용히 진행
    except Exception as _e:
        # ETA는 기능 보조이므로 에러 무시
        print(f"[ETA] predictor skipped ({type(_e).__name__})", file=sys.stderr, flush=True)

    # 경로 재평가 필요 여부: full 출력 + no_paths 미사용
    want_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))

    # === 캘리브레이션 분기 ===
    if getattr(args, "calib", "off") == "on":
        out = calibrate_lambda(args)
        if isinstance(out, dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(args, "es_mode", "wealth")).lower()
    else:
        # === 일반 실행 경로 ===
        if args.method == "rl":
            res = run_rl(args)
            out = _maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"), want_paths=want_paths)
        else:
            res = run_once(args)
            out = _maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"), want_paths=want_paths)

    # ★ 메타 필드 보강(요약/표 출력에서 비지 않도록)
    if isinstance(out, dict):
        out.setdefault("tag", getattr(args, "tag", None))
        out.setdefault("method", getattr(args, "method", None))
        out.setdefault("asset", getattr(args, "asset", None))

    # ES95(CVaR) 보정(wealth→loss 경로레벨 재계산; WT 없으면 스킵)
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

    # (선택) extras.eval_WT_n 편의 필드
    try:
        if isinstance(out, dict) and isinstance(out.get("extra"), dict):
            ew = out["extra"].get("eval_WT")
            if isinstance(ew, (list, tuple)):
                out["extra"]["eval_WT_n"] = len(ew)
    except Exception:
        pass

    # 전체 실행시간 기록
    elapsed = time.perf_counter() - t0
    if isinstance(out, dict):
        out["time_total_s"] = round(elapsed, 3)
        out["time_total_hms"] = _fmt_hms(elapsed)

    # ETA 히스토리 갱신
    try:
        _eta_record(args, elapsed)
    except Exception:
        pass

    # 표준출력 제어
    to_print = _prune_for_stdout(args, out) if isinstance(out, dict) else out
    print(json.dumps(to_print, ensure_ascii=False))


if __name__ == "__main__":
    main()
