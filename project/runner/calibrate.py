from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable, Any
from argparse import Namespace
import time
import math

from .run import run_once

# ---------- 공용 유틸 (WT, ES 계산) ----------

def _maybe_extract_WT(candidate: Any) -> Optional[Iterable[float]]:
    if candidate is None:
        return None
    try:
        import numpy as _np
        if isinstance(candidate, _np.ndarray):
            return candidate.tolist()
    except Exception:
        pass
    if isinstance(candidate, (list, tuple)):
        if len(candidate) >= 2 and isinstance(candidate[1], (dict, list, tuple)):
            wt = _maybe_extract_WT(candidate[1])
            if wt is not None:
                return wt
        if candidate and all(isinstance(x, (int, float)) for x in candidate):
            return candidate  # type: ignore
    if isinstance(candidate, dict):
        for k in ["eval_WT","W_T","WT","terminal_wealth","terminal_wealths",
                  "paths_WT","wt_paths","wealth_terminal","wealth_T"]:
            if k in candidate and candidate[k] is not None:
                return candidate[k]  # type: ignore
        for k in ["metrics","extra","extras","eval","payload","data","result"]:
            if k in candidate and isinstance(candidate[k], (dict, list, tuple)):
                wt = _maybe_extract_WT(candidate[k])
                if wt is not None:
                    return wt
    for attr in ["eval_WT","W_T","WT","terminal_wealth","paths_WT","wealth_T","terminal_wealths"]:
        try:
            wt = getattr(candidate, attr)
            if wt is not None:
                return wt  # type: ignore
        except Exception:
            pass
    return None

def _es_acerbi_tasche(losses: Iterable[float], alpha: float) -> float:
    import numpy as np
    L = np.asarray(list(losses), dtype=float)
    n = L.size
    if n == 0:
        return 0.0
    a = float(alpha)
    a = max(min(a, 1.0 - 1e-12), 1e-12)
    L.sort()
    j = int(math.floor(n * a))
    if j >= n:
        j = n - 1
    theta = n * a - j
    Lj1 = float(L[j])
    tail_sum = float(L[j+1:].sum())
    ES = ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))
    return float(ES)

def _compute_es_loss(res: Dict[str, Any], F_target: float, alpha: float) -> Tuple[Optional[float], Dict[str, float]]:
    import numpy as np
    WT = None
    for cand in (res.get("extra"), res, res.get("metrics")):
        if cand is not None:
            WT = _maybe_extract_WT(cand)
            if WT is not None:
                break
    if WT is None:
        return None, {}
    W = np.asarray(list(WT), dtype=float)
    L = np.maximum(F_target - W, 0.0)
    ES = _es_acerbi_tasche(L, alpha=alpha)
    out = {
        "EW": float(W.mean()) if W.size > 0 else 0.0,
        "mean_WT": float(W.mean()) if W.size > 0 else 0.0,
        "ES95": float(ES),
        "Ruin": float((W <= 0.0).mean()) if W.size > 0 else 0.0,
    }
    return float(ES), out

def _copy_args(args: Namespace, **overrides) -> Namespace:
    d = vars(args).copy()
    d.update(overrides)
    return Namespace(**d)

def _first_seed_list(seeds_val: Any) -> list[int]:
    if isinstance(seeds_val, int):
        return [int(seeds_val)]
    if isinstance(seeds_val, (list, tuple)) and len(seeds_val) > 0:
        try:
            return [int(seeds_val[0])]
        except Exception:
            return [0]
    return [0]

def _fmt_hms(sec: float) -> str:
    try:
        total = int(round(float(sec)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"

# ---------- 내부: 한 점 평가 ----------

def _eval_point(args: Namespace, lambda_term: Optional[float] = None,
                F_target: Optional[float] = None, fast: bool = True) -> Tuple[Dict[str, Any], Optional[float]]:
    overrides = dict(es_mode="loss")
    if lambda_term is not None:
        overrides["lambda_term"] = float(lambda_term)
    if F_target is not None:
        overrides["F_target"] = float(F_target)

    if fast and str(getattr(args, "calib_fast","on")).lower() == "on":
        seeds1 = _first_seed_list(getattr(args, "seeds", [0]))
        overrides.update(dict(
            hjb_W_grid=81, hjb_Nshock=128, hjb_eta_n=41,
            n_paths=200, seeds=seeds1
        ))

    local = _copy_args(args, **overrides)
    res = run_once(local)

    # path-level로 ES 재계산(wealth→loss 변환)
    F = float(getattr(local, "F_target", getattr(args, "F_target", 0.0)) or 0.0)
    alpha = float(getattr(local, "alpha", getattr(args, "alpha", 0.95)) or 0.95)
    es_paths, patch = _compute_es_loss(res, F_target=F, alpha=alpha)

    if es_paths is not None:
        m = res.get("metrics") or {}
        m.update(patch)
        m["es95_source"] = "path_level_cvar"
        res["metrics"] = m
        return res, float(es_paths)
    else:
        es0 = (res.get("metrics") or {}).get("ES95")
        return res, float(es0) if es0 is not None else None

# ---------- 메인: 캘리브레이션 ----------

def calibrate_lambda(args: Namespace) -> Dict[str, Any]:
    """
    calib_param=lambda(기본): λ를 이분법으로 조정해 ES(loss)≈cvar_target
    calib_param=F: F_target을 이분법으로 조정해 ES(loss)≈cvar_target
    - 브래킷 불가능시(unattainable_*): 상태/경계값을 결과에 기록
    - 총 소요시간/fast/slow 분해시간/평가회수 포함
    """
    target = float(args.cvar_target)
    tol = float(args.cvar_tol)
    max_iter = int(getattr(args, "calib_max_iter", 8))
    mode = str(getattr(args, "calib_param", "lambda"))

    # 시간 집계
    t0_total = time.perf_counter()
    n_fast = n_slow = 0
    t_fast = t_slow = 0.0

    def wrap_eval(lambda_term: Optional[float], F_target: Optional[float], fast: bool):
        nonlocal n_fast, n_slow, t_fast, t_slow
        t1 = time.perf_counter()
        res, es = _eval_point(args, lambda_term=lambda_term, F_target=F_target, fast=fast)
        dt = time.perf_counter() - t1
        if fast: n_fast += 1; t_fast += dt
        else:    n_slow += 1; t_slow += dt
        return res, es

    history: list[Dict[str, float]] = []

    if mode == "lambda":
        lo = float(args.lambda_min); hi = float(args.lambda_max)
        # 초기
        res_lo, es_lo = wrap_eval(lambda_term=lo, F_target=None, fast=True)
        res_hi, es_hi = wrap_eval(lambda_term=hi, F_target=None, fast=True)

        # 불가능 판정(둘 다 아래 / 둘 다 위)
        if (es_lo is not None) and (es_hi is not None):
            if (es_lo < target) and (es_hi < target):
                # 가장 가까운 쪽을 우선 선택
                cand_lambda = lo if abs(es_lo - target) <= abs(es_hi - target) else hi
                final_res, final_es = wrap_eval(lambda_term=cand_lambda, F_target=None, fast=False)
                total = time.perf_counter() - t0_total
                final_res.setdefault("cvar_calibration", {})
                final_res["cvar_calibration"].update({
                    "mode": "lambda",
                    "status": "unattainable_below_range",
                    "selected_lambda": float(cand_lambda),
                    "selected_ES95": float(final_es) if final_es is not None else None,
                    "cvar_target": target, "cvar_tol": tol,
                    "lambda_min": float(args.lambda_min), "lambda_max": float(args.lambda_max),
                    "es_bounds": {"lo": float(es_lo), "hi": float(es_hi)},
                    "history_tail": [],
                    "time_total_s": round(total, 3), "time_total_hms": _fmt_hms(total),
                    "evals_fast": n_fast, "evals_slow": n_slow,
                    "time_fast_s": round(t_fast, 3), "time_slow_s": round(t_slow, 3),
                })
                return final_res
            if (es_lo > target) and (es_hi > target):
                cand_lambda = lo if abs(es_lo - target) <= abs(es_hi - target) else hi
                final_res, final_es = wrap_eval(lambda_term=cand_lambda, F_target=None, fast=False)
                total = time.perf_counter() - t0_total
                final_res.setdefault("cvar_calibration", {})
                final_res["cvar_calibration"].update({
                    "mode": "lambda",
                    "status": "unattainable_above_range",
                    "selected_lambda": float(cand_lambda),
                    "selected_ES95": float(final_es) if final_es is not None else None,
                    "cvar_target": target, "cvar_tol": tol,
                    "lambda_min": float(args.lambda_min), "lambda_max": float(args.lambda_max),
                    "es_bounds": {"lo": float(es_lo), "hi": float(es_hi)},
                    "history_tail": [],
                    "time_total_s": round(total, 3), "time_total_hms": _fmt_hms(total),
                    "evals_fast": n_fast, "evals_slow": n_slow,
                    "time_fast_s": round(t_fast, 3), "time_slow_s": round(t_slow, 3),
                })
                return final_res

        # 이분 탐색
        best_lmbd, best_res, best_es = lo, res_lo, es_lo
        prev_es: Optional[float] = None
        status = "ok"
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            res_mid, es_mid = wrap_eval(lambda_term=mid, F_target=None, fast=True)
            history.append({"lambda": float(mid), "ES95": float(es_mid) if es_mid is not None else None})

            if prev_es is not None and es_mid is not None and abs(es_mid - prev_es) < 1e-4:
                best_lmbd, best_res, best_es = mid, res_mid, es_mid; status = "plateau"; break
            prev_es = es_mid

            if es_mid is None:
                lo = mid; best_lmbd, best_res, best_es = mid, res_mid, es_mid; status = "incomplete"; continue
            if abs(es_mid - target) <= tol:
                best_lmbd, best_res, best_es = mid, res_mid, es_mid; status = "ok"; break

            # ES(loss)가 λ에 대해 단조 감소한다는 가정
            if es_mid > target: lo = mid
            else:               hi = mid

            best_lmbd, best_res, best_es = mid, res_mid, es_mid

        final_res, final_es = wrap_eval(lambda_term=best_lmbd, F_target=None, fast=False)
        total = time.perf_counter() - t0_total
        final_res.setdefault("cvar_calibration", {})
        final_res["cvar_calibration"].update({
            "mode": "lambda",
            "status": status,
            "selected_lambda": float(best_lmbd),
            "selected_ES95": float(final_es) if final_es is not None else None,
            "cvar_target": target, "cvar_tol": tol,
            "lambda_min": float(args.lambda_min), "lambda_max": float(args.lambda_max),
            "iterations": len(history), "history_tail": history[-5:],
            "time_total_s": round(total, 3), "time_total_hms": _fmt_hms(total),
            "evals_fast": n_fast, "evals_slow": n_slow,
            "time_fast_s": round(t_fast, 3), "time_slow_s": round(t_slow, 3),
        })
        return final_res

    # -------- calib_param == "F": F_target 보정 --------
    F0 = float(getattr(args, "F_target", 0.0))
    F_min = float(getattr(args, "F_min", F0 - 0.5))
    F_max = float(getattr(args, "F_max", F0 + 0.5))

    res_lo, es_lo = wrap_eval(lambda_term=None, F_target=F_min, fast=True)
    res_hi, es_hi = wrap_eval(lambda_term=None, F_target=F_max, fast=True)

    # F는 ES(loss)에 대해 단조 증가(큰 F일수록 손실 증가) — 일반적으로 브래킷을 만들기 쉬움
    if (es_lo is not None) and (es_hi is not None) and (es_lo > target) and (es_hi > target):
        # 둘 다 위면 범위를 더 아래로 내림
        F_min = F_min - 0.5
        res_lo, es_lo = wrap_eval(lambda_term=None, F_target=F_min, fast=True)
    if (es_lo is not None) and (es_hi is not None) and (es_lo < target) and (es_hi < target):
        # 둘 다 아래면 범위를 더 위로 올림
        F_max = F_max + 0.5
        res_hi, es_hi = wrap_eval(lambda_term=None, F_target=F_max, fast=True)

    best_F, best_res, best_es = F0, res_lo, es_lo
    status = "ok"
    prev_es: Optional[float] = None
    for _ in range(max_iter):
        F_mid = 0.5 * (F_min + F_max)
        res_mid, es_mid = wrap_eval(lambda_term=None, F_target=F_mid, fast=True)
        history.append({"F_target": float(F_mid), "ES95": float(es_mid) if es_mid is not None else None})

        if prev_es is not None and es_mid is not None and abs(es_mid - prev_es) < 1e-4:
            best_F, best_res, best_es = F_mid, res_mid, es_mid; status = "plateau"; break
        prev_es = es_mid

        if es_mid is None:
            F_min = F_mid; best_F, best_res, best_es = F_mid, res_mid, es_mid; status = "incomplete"; continue
        if abs(es_mid - target) <= tol:
            best_F, best_res, best_es = F_mid, res_mid, es_mid; status = "ok"; break

        # ES(loss)는 F에 대해 단조 증가
        if es_mid < target: F_min = F_mid
        else:               F_max = F_mid

        best_F, best_res, best_es = F_mid, res_mid, es_mid

    final_res, final_es = wrap_eval(lambda_term=None, F_target=best_F, fast=False)
    total = time.perf_counter() - t0_total
    final_res.setdefault("cvar_calibration", {})
    final_res["cvar_calibration"].update({
        "mode": "F",
        "status": status,
        "selected_F_target": float(best_F),
        "selected_ES95": float(final_es) if final_es is not None else None,
        "cvar_target": target, "cvar_tol": tol,
        "F_min": float(F_min), "F_max": float(F_max),
        "iterations": len(history), "history_tail": history[-5:],
        "time_total_s": round(total, 3), "time_total_hms": _fmt_hms(total),
        "evals_fast": n_fast, "evals_slow": n_slow,
        "time_fast_s": round(t_fast, 3), "time_slow_s": round(t_slow, 3),
    })
    return final_res
