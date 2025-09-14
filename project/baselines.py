# project/.../baselines.py
from __future__ import annotations

import numpy as _np
from typing import Protocol, runtime_checkable, Tuple, Optional, Dict, Any

__all__ = ["rule_4pct", "rule_cpb", "rule_vpw"]


# ---- 최소 요구 인터페이스 정의(환경/설정) ----
@runtime_checkable
class EnvLike(Protocol):
    T: int                 # 총 기간(월)
    t: int                 # 현재 시점(0..T-1)
    W: float | _np.ndarray # 현재 부 (scalar 기대)
    def step(self, *, q: float, w: float) -> Tuple[object, object, bool]: ...


@runtime_checkable
class CfgLike(Protocol):
    steps_per_year: int
    w_fixed: Optional[float]
    w_max: float
    def monthly(self) -> Dict[str, Any]: ...
    # (선택) floor가 있으면 적용
    q_floor: float | int = 0.0  # 존재하지 않으면 getattr로 기본값 사용


# ---- 유틸 ----
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pick_w(cfg: CfgLike) -> float:
    w = cfg.w_fixed if getattr(cfg, "w_fixed", None) is not None else cfg.w_max
    return _clamp01(float(w))


def _apply_q_floor(cfg: CfgLike, q_m: float) -> float:
    q_floor = float(getattr(cfg, "q_floor", 0.0))
    return max(q_floor, _clamp01(q_m))


def _to_scalar_f(x) -> float:
    """env.W가 float/np scalar/0-d array여도 안전하게 float 변환."""
    a = _np.asarray(x)
    return float(a.item() if a.shape == () else a.reshape(-1)[0])


# ---- 4% Rule (연 4% -> 월 변환 고정 인출율) ----
def rule_4pct(env: EnvLike, cfg: CfgLike) -> _np.ndarray:
    """
    q_m = 1 - (1 - 0.04)^(1/steps_per_year)
    w: cfg.w_fixed 있으면 사용, 없으면 w_max.
    """
    q_m = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
    q_m = _apply_q_floor(cfg, q_m)
    w = _pick_w(cfg)

    W_hist: list[float] = []
    for _ in range(env.T):
        _, _, done = env.step(q=q_m, w=w)
        W_hist.append(_to_scalar_f(env.W))
        if done:
            break
    return _np.asarray(W_hist, dtype=float)


# ---- CPB (Constant Percentage of Balance) ----
def rule_cpb(env: EnvLike, cfg: CfgLike) -> _np.ndarray:
    """
    매월 잔고 대비 고정 비율 인출.
    cfg.monthly()['p_m']가 없으면 4%/steps_per_year로 대체.
    """
    m = cfg.monthly()
    p_m = float(m.get("p_m", 0.04 / cfg.steps_per_year))
    q_m = _apply_q_floor(cfg, p_m)
    w = _pick_w(cfg)

    W_hist: list[float] = []
    for _ in range(env.T):
        _, _, done = env.step(q=q_m, w=w)
        W_hist.append(_to_scalar_f(env.W))
        if done:
            break
    return _np.asarray(W_hist, dtype=float)


# ---- VPW (Variable Percentage Withdrawal) ----
def rule_vpw(env: EnvLike, cfg: CfgLike) -> _np.ndarray:
    """
    남은 기간 Nm과 예상 월 성장률 g에 대해 연금계수 a로 q_m=1/a.
    g≈0이면 a≈Nm. 음수/극단 g 안정화 포함.
    """
    m = cfg.monthly()
    g = float(m.get("g_m", 0.0))

    W_hist: list[float] = []
    for _ in range(env.T):
        Nm = max(int(env.T - getattr(env, "t", 0)), 1)
        g_eff = max(g, -0.99)  # (1+g)^(-Nm) 안전장치
        if abs(g_eff) < 1e-12:
            a = float(Nm)
        else:
            a = (1.0 - (1.0 + g_eff) ** (-Nm)) / g_eff
            a = max(a, 1.0)  # 과도한 인출 방지

        q_m = _apply_q_floor(cfg, 1.0 / a)
        w = _pick_w(cfg)

        _, _, done = env.step(q=q_m, w=w)
        W_hist.append(_to_scalar_f(env.W))
        if done:
            break

    return _np.asarray(W_hist, dtype=float)
