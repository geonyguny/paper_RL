# project/runner/calibrate.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Iterable, Any, Callable, List
from argparse import Namespace
import time
import math

from .run import run_once

# ======================================================================
# 공용: WT 추출 / ES(Acerbi–Tasche) / ES(loss) 재계산
# ======================================================================

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
    tail_sum = float(L[j+1:].sum()) if (j + 1) < n else 0.0
    ES = ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))
    return float(ES)


def _compute_es_loss(res: Dict[str, Any], F_target: float, alpha: float) -> Tuple[Optional[float], Dict[str, float]]:
    import numpy as np
    WT = None
    for cand in (res.get("extras"), res.get("extra"), res, res.get("metrics")):
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
        "es95_source": "path"
    }
    return float(ES), out


def _fmt_hms(sec: float) -> str:
    try:
        total = int(round(float(sec)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"


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

# ======================================================================
# Evaluator 콜백 & Fast 프리셋
# ======================================================================

def _eval_point(args: Namespace,
                lambda_term: Optional[float] = None,
                F_target: Optional[float] = None,
                fast: bool = True) -> Tuple[Dict[str, Any], Optional[float]]:
    """
    단일 점(method/hyper) 평가 + ES(loss) 경로레벨 재계산
    반환: (결과 JSON, ES95(loss))
    """
    overrides: Dict[str, Any] = dict(es_mode="loss")
    if lambda_term is not None:
        overrides["lambda_term"] = float(lambda_term)
    if F_target is not None:
        overrides["F_target"] = float(F_target)

    # Fast 프리셋(캘리브레이션 탐색 단계) — seeds 단일화 + 경로 축소
    if fast and str(getattr(args, "calib_fast", "on")).lower() == "on":
        seeds1 = _first_seed_list(getattr(args, "seeds", [0]))
        overrides.update(dict(
            # HJB 빠른 격자/샘플 — 필요 시 solver에서 오버라이드 가능
            hjb_W_grid=81, hjb_Nshock=128, hjb_eta_n=41,
            n_paths=200, seeds=seeds1
        ))

    local = _copy_args(args, **overrides)
    res = run_once(local)

    # path-level ES(loss) 재계산
    F = float(getattr(local, "F_target", getattr(args, "F_target", 0.0)) or 0.0)
    alpha = float(getattr(local, "alpha", getattr(args, "alpha", 0.95)) or 0.95)
    es_paths, patch = _compute_es_loss(res, F_target=F, alpha=alpha)

    if es_paths is not None:
        m = res.get("metrics") or {}
        m.update(patch)
        res["metrics"] = m
        # extras.eval_WT 존재 보장(없으면 빈 리스트)
        ex = res.get("extras") or {}
        if "eval_WT" not in ex:
            ex["eval_WT"] = ex.get("eval_WT", [])
        res["extras"] = ex
        return res, float(es_paths)

    # WT없어 ES 계산 불가 → 기존 값 사용(가능 시)
    es0 = (res.get("metrics") or {}).get("ES95")
    return res, float(es0) if es0 is not None else None


def make_evaluator(args: Namespace) -> Callable[[Optional[float], Optional[float], bool], Tuple[Dict[str, Any], Optional[float]]]:
    """외부에서 재사용 가능한 evaluator 생성자."""
    def _inner(lambda_term: Optional[float] = None, F_target: Optional[float] = None, fast: bool = True):
        return _eval_point(args, lambda_term=lambda_term, F_target=F_target, fast=fast)
    return _inner

# ======================================================================
# Dataclass & 순수 캘리브레이션 루틴
# ======================================================================

@dataclass
class CalibResult:
    mode: str                      # "F" | "lambda"
    stopped: str                   # "tol_met" | "iter_max" | "plateau" | "unattainable_low" | "unattainable_high" | "incomplete"
    target: float
    tol: float
    selected_value: float          # F_selected or lambda_selected 값
    selected_ES95: Optional[float]
    history: List[Dict[str, Optional[float]]]
    time_total_s: float
    evals_fast: int
    evals_slow: int
    time_fast_s: float
    time_slow_s: float
    meta: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        # 출력 키 통일: F_selected / lambda_selected
        mode = d.pop("mode")
        sel = d.pop("selected_value")
        payload = {
            "mode": mode,
            "stopped": self.stopped,
            "cvar_target": self.target,
            "cvar_tol": self.tol,
            "history": self.history,
            "selected_ES95": self.selected_ES95,
            "time_total_s": round(self.time_total_s, 3),
            "time_total_hms": _fmt_hms(self.time_total_s),
            "evals_fast": self.evals_fast,
            "evals_slow": self.evals_slow,
            "time_fast_s": round(self.time_fast_s, 3),
            "time_slow_s": round(self.time_slow_s, 3),
        }
        if mode == "F":
            payload["F_selected"] = float(sel)
            payload.update({k: self.meta.get(k) for k in ["F_min", "F_max"] if k in self.meta})
        else:
            payload["lambda_selected"] = float(sel)
            payload.update({k: self.meta.get(k) for k in ["lambda_min", "lambda_max"] if k in self.meta})
        return payload


def calibrate_F(
    evaluator: Callable[[Optional[float], Optional[float], bool], Tuple[Dict[str, Any], Optional[float]]],
    F_min: float,
    F_max: float,
    target: float,
    tol: float,
    max_iter: int,
) -> Tuple[Dict[str, Any], CalibResult]:
    """
    F_target 이분 탐색: ES(loss) ≈ target
    규칙(es_mode=loss): ES95 > target ⇒ 과위험 ⇒ F ↑ (소비↓)
    """
    t0 = time.perf_counter()
    n_fast = n_slow = 0
    t_fast = t_slow = 0.0

    def _wrap(F_val: float, fast: bool):
        nonlocal n_fast, n_slow, t_fast, t_slow
        t1 = time.perf_counter()
        res, es = evaluator(None, F_val, fast)
        dt = time.perf_counter() - t1
        if fast:
            n_fast += 1; t_fast += dt
        else:
            n_slow += 1; t_slow += dt
        return res, es

    # 브래킷 평가
    res_lo, es_lo = _wrap(F_min, True)
    res_hi, es_hi = _wrap(F_max, True)

    # 히스토리 누적
    history: List[Dict[str, Optional[float]]] = [
        {"F_target": float(F_min), "ES95": float(es_lo) if es_lo is not None else None},
        {"F_target": float(F_max), "ES95": float(es_hi) if es_hi is not None else None},
    ]

    # 이분 탐색
    status = "iter_max"
    best_F = F_min
    best_res, best_es = res_lo, es_lo
    prev_es: Optional[float] = None

    for _ in range(max_iter):
        F_mid = 0.5 * (F_min + F_max)
        res_mid, es_mid = _wrap(F_mid, True)
        history.append({"F_target": float(F_mid), "ES95": float(es_mid) if es_mid is not None else None})

        if prev_es is not None and es_mid is not None and abs(es_mid - prev_es) < 1e-4:
            best_F, best_res, best_es = F_mid, res_mid, es_mid
            status = "plateau"
            break
        prev_es = es_mid

        if es_mid is None:
            # 정보 부족 → 하한 이동
            F_min = F_mid
            best_F, best_res, best_es = F_mid, res_mid, es_mid
            status = "incomplete"
            continue

        # 수렴 판정
        if abs(es_mid - target) <= tol:
            best_F, best_res, best_es = F_mid, res_mid, es_mid
            status = "tol_met"
            break

        # 단조 증가 가정(ES↑ when F↑). 과위험(ES>target)이면 F↑
        if es_mid > target:
            F_min = F_mid
        else:
            F_max = F_mid

        best_F, best_res, best_es = F_mid, res_mid, es_mid

    # 최종 슬로우 평가
    final_res, final_es = _wrap(best_F, False)

    cr = CalibResult(
        mode="F",
        stopped=status,
        target=target,
        tol=tol,
        selected_value=float(best_F),
        selected_ES95=float(final_es) if final_es is not None else None,
        history=history[-5:],  # tail만 보관
        time_total_s=time.perf_counter() - t0,
        evals_fast=n_fast,
        evals_slow=n_slow,
        time_fast_s=t_fast,
        time_slow_s=t_slow,
        meta={"F_min": float(F_min), "F_max": float(F_max)},
    )
    return final_res, cr


def calibrate_lambda_core(
    evaluator: Callable[[Optional[float], Optional[float], bool], Tuple[Dict[str, Any], Optional[float]]],
    lam_min: float,
    lam_max: float,
    target: float,
    tol: float,
    max_iter: int,
) -> Tuple[Dict[str, Any], CalibResult]:
    """
    λ 이분 탐색: ES(loss) ≈ target
    (경험적 가정) λ↑ ⇒ 위험(ES)↓ 경향 → ES > target이면 λ↑
    """
    t0 = time.perf_counter()
    n_fast = n_slow = 0
    t_fast = t_slow = 0.0

    def _wrap(lmbd: float, fast: bool):
        nonlocal n_fast, n_slow, t_fast, t_slow
        t1 = time.perf_counter()
        res, es = evaluator(lmbd, None, fast)
        dt = time.perf_counter() - t1
        if fast:
            n_fast += 1; t_fast += dt
        else:
            n_slow += 1; t_slow += dt
        return res, es

    res_lo, es_lo = _wrap(lam_min, True)
    res_hi, es_hi = _wrap(lam_max, True)

    # 브래킷 성립 여부(둘 다 아래/둘 다 위일 때 안내)
    unattainable: Optional[str] = None
    if (es_lo is not None) and (es_hi is not None):
        if (es_lo < target) and (es_hi < target):
            unattainable = "unattainable_low"
        elif (es_lo > target) and (es_hi > target):
            unattainable = "unattainable_high"

    history: List[Dict[str, Optional[float]]] = [
        {"lambda": float(lam_min), "ES95": float(es_lo) if es_lo is not None else None},
        {"lambda": float(lam_max), "ES95": float(es_hi) if es_hi is not None else None},
    ]

    if unattainable:
        # 더 가까운 경계 선택
        cand = lam_min if (es_lo is not None and es_hi is not None and abs(es_lo - target) <= abs(es_hi - target)) else lam_max
        final_res, final_es = _wrap(cand, False)
        cr = CalibResult(
            mode="lambda",
            stopped=unattainable,
            target=target,
            tol=tol,
            selected_value=float(cand),
            selected_ES95=float(final_es) if final_es is not None else None,
            history=[],  # tail 없음
            time_total_s=time.perf_counter() - t0,
            evals_fast=n_fast,
            evals_slow=n_slow,
            time_fast_s=t_fast,
            time_slow_s=t_slow,
            meta={"lambda_min": float(lam_min), "lambda_max": float(lam_max)},
        )
        return final_res, cr

    status = "iter_max"
    best_l, best_res, best_es = lam_min, res_lo, es_lo
    prev_es: Optional[float] = None

    for _ in range(max_iter):
        mid = 0.5 * (lam_min + lam_max)
        res_mid, es_mid = _wrap(mid, True)
        history.append({"lambda": float(mid), "ES95": float(es_mid) if es_mid is not None else None})

        if prev_es is not None and es_mid is not None and abs(es_mid - prev_es) < 1e-4:
            best_l, best_res, best_es = mid, res_mid, es_mid
            status = "plateau"
            break
        prev_es = es_mid

        if es_mid is None:
            lam_min = mid
            best_l, best_res, best_es = mid, res_mid, es_mid
            status = "incomplete"
            continue

        if abs(es_mid - target) <= tol:
            best_l, best_res, best_es = mid, res_mid, es_mid
            status = "tol_met"
            break

        # (가정) ES(loss) vs λ: 단조 감소 → ES>target이면 λ↑
        if es_mid > target:
            lam_min = mid
        else:
            lam_max = mid

        best_l, best_res, best_es = mid, res_mid, es_mid

    final_res, final_es = _wrap(best_l, False)
    cr = CalibResult(
        mode="lambda",
        stopped=status,
        target=target,
        tol=tol,
        selected_value=float(best_l),
        selected_ES95=float(final_es) if final_es is not None else None,
        history=history[-5:],
        time_total_s=time.perf_counter() - t0,
        evals_fast=n_fast,
        evals_slow=n_slow,
        time_fast_s=t_fast,
        time_slow_s=t_slow,
        meta={"lambda_min": float(lam_min), "lambda_max": float(lam_max)},
    )
    return final_res, cr

# ======================================================================
# 엔트리포인트(기존 호환): calibrate_lambda(args)
# - calib_param=lambda | F
# - 결과 JSON에 cvar_calibration 섹션 병합
# ======================================================================

def calibrate_lambda(args: Namespace) -> Dict[str, Any]:
    """
    기존 CLI 호환 엔트리: calib_param에 따라 λ 또는 F를 이분 탐색.
    - 모든 평가에서 es_mode=loss 강제
    - path-level ES(Acerbi–Tasche)로 일관 재계산
    - 결과 JSON: cvar_calibration = {
        mode, stopped, cvar_target, cvar_tol,
        F_selected|lambda_selected, selected_ES95,
        history(꼬리 5개), time_total_s/hms, evals/time 분해, 경계값(meta)
      }
    """
    mode = str(getattr(args, "calib_param", "lambda")).lower()
    target = float(getattr(args, "cvar_target", 0.45))
    tol = float(getattr(args, "cvar_tol", 0.01))
    max_iter = int(getattr(args, "calib_max_iter", 8))

    evaluator = make_evaluator(args)

    if mode == "f":
        F0 = float(getattr(args, "F_target", 0.60) or 0.60)

        # None이면 F0±0.5로 자동 브래킷 설정
        F_min_arg = getattr(args, "F_min", None)
        F_max_arg = getattr(args, "F_max", None)
        F_min = float(F_min_arg) if F_min_arg is not None else (F0 - 0.5)
        F_max = float(F_max_arg) if F_max_arg is not None else (F0 + 0.5)

        # NaN/Inf 방지
        if not math.isfinite(F_min):
            F_min = F0 - 0.5
        if not math.isfinite(F_max):
            F_max = F0 + 0.5

        # 잘못된 브래킷 교정(F_min >= F_max인 경우 F0 중심 재설정)
        if F_min >= F_max:
            half = max(0.1, abs(F0) * 0.5)
            F_min, F_max = F0 - half, F0 + half

        final_res, cr = calibrate_F(
            evaluator=evaluator,
            F_min=F_min, F_max=F_max,
            target=target, tol=tol, max_iter=max_iter,
        )
        final_res.setdefault("cvar_calibration", {})
        final_res["cvar_calibration"].update(cr.to_json())
        return final_res

    # default: lambda
    lam_min = float(getattr(args, "lambda_min", 0.25))
    lam_max = float(getattr(args, "lambda_max", 2.0))
    final_res, cr = calibrate_lambda_core(
        evaluator=evaluator,
        lam_min=lam_min, lam_max=lam_max,
        target=target, tol=tol, max_iter=max_iter,
    )
    final_res.setdefault("cvar_calibration", {})
    final_res["cvar_calibration"].update(cr.to_json())
    return final_res
