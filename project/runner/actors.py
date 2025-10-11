# project/runner/actors.py
from __future__ import annotations
from typing import Any, Tuple, Callable
import numpy as _np

from ..config import SimConfig
# ⚠️ runner 경로의 Env로 통일
from ..env.retirement_env import RetirementEnv  # type: ignore
from ..hjb import HJBSolver
from .helpers import arrhash, monthly_from_cfg, get_life_table_from_env
from .logging_filters import mute_logs, mute_kgr_year_logs_if

# K-GR rule
from ..policy.kgr_rule import (
    KGRLiteConfig, kgr_lite_init, kgr_lite_update_yearly, kgr_lite_policy_step,
)

# --------------------------
# Adapters & Safety
# --------------------------
def _as_nd_state(raw: Any, *, T_hint: int | None = None, W_hint: float | None = None) -> _np.ndarray:
    """
    정책 입력 상태를 항상 ndarray([t_norm, W]) 로 정규화.
    - dict: {"t": step_idx, "W": wealth} 가정
    - ndarray: 그대로 사용(길이<2면 보강)
    - 기타: [0.0, W_hint] 폴백
    """
    if isinstance(raw, dict):
        T = int(T_hint or 1)
        t_idx = float(raw.get("t", 0.0))
        t_norm = t_idx / float(max(1, T - 1))
        W_now = float(raw.get("W", W_hint if W_hint is not None else 0.0))
        return _np.asarray([t_norm, W_now], dtype=float)

    if isinstance(raw, _np.ndarray):
        arr = _np.asarray(raw, dtype=float).ravel()
        if arr.size >= 2:
            return arr[:2]
        W_now = float(W_hint if W_hint is not None else 0.0)
        return _np.asarray([float(arr[0]) if arr.size else 0.0, W_now], dtype=float)

    return _np.asarray([0.0, float(W_hint if W_hint is not None else 0.0)], dtype=float)


def _clip_action(q: float, w: float, cfg: SimConfig) -> Tuple[float, float]:
    q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)
    q = float(_np.clip(q, q_floor, 1.0))
    w = float(_np.clip(w, 0.0, w_max))
    return q, w


# ---------- RULE ----------
def rule_actor_4pct(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def actor(_obs):
        q_m = 1.0 - (1.0 - 0.04) ** (1.0 / int(getattr(cfg, "steps_per_year", 12) or 12))
        w = float(getattr(cfg, "w_fixed", None) if getattr(cfg, "w_fixed", None) is not None else getattr(cfg, "w_max", 1.0))
        return _clip_action(q_m, w, cfg)
    return actor


def rule_actor_cpb(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def actor(_obs):
        _g_m, p_m = monthly_from_cfg(cfg)
        q_m = float(p_m)
        w = float(getattr(cfg, "w_fixed", None) if getattr(cfg, "w_fixed", None) is not None else getattr(cfg, "w_max", 1.0))
        return _clip_action(q_m, w, cfg)
    return actor


def rule_actor_vpw(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def _get_g_m(_cfg):
        try:
            if hasattr(_cfg, "monthly") and callable(_cfg.monthly):
                m = _cfg.monthly()
                if isinstance(m, dict) and "g_m" in m:
                    gm = float(m["g_m"])
                    if _np.isfinite(gm):
                        return gm
        except Exception:
            pass
        g_ann = float(getattr(_cfg, "g_real_annual", 0.0) or 0.0)
        spm = int(getattr(_cfg, "steps_per_year", 12) or 12)
        return (1.0 + g_ann) ** (1.0 / spm) - 1.0

    def actor(_obs):
        t = int(getattr(env, "t", 0))
        T = int(getattr(env, "T", 1) or 1)
        Nm = max(T - t, 1)
        g_m = _get_g_m(cfg)
        if abs(g_m) < 1e-12:
            q_m = 1.0 / Nm
        else:
            a = (1.0 - (1.0 + g_m) ** (-Nm)) / g_m
            q_m = 1.0 / max(a, 1e-12)
        w = float(getattr(cfg, "w_fixed", None) if getattr(cfg, "w_fixed", None) is not None else getattr(cfg, "w_max", 1.0))
        return _clip_action(q_m, w, cfg)
    return actor


def rule_actor_kgr(cfg: SimConfig, env: RetirementEnv, *, quiet: bool) -> Callable[[Any], Tuple[float, float]]:
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    q_floor = float(getattr(cfg, "q_floor", 0.02) or 0.02)
    fee_annual = float(getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", 0.004)) or 0.004)
    w_fixed = float(getattr(cfg, "w_fixed", None) if getattr(cfg, "w_fixed", None) is not None else getattr(cfg, "w_max", 1.0))

    life_table_df = get_life_table_from_env(env)
    r_f_real_annual = float(getattr(env, "r_f_real_annual", 0.02) or 0.02)
    W0 = float(getattr(env, "W0", 1.0))
    age0 = float(getattr(cfg, "age0", 65) or 65)

    kgr_cfg = KGRLiteConfig(
        FR_high=1.30, FR_low=0.85, delta_up=0.07, delta_dn=-0.07,
        kappa_safety=0.002, w_fixed=w_fixed, q_floor=q_floor,
        phi_adval_annual=fee_annual, steps_per_year=steps_per_year,
    )
    kgr_state = None
    _kgr_once_logged = False

    if life_table_df is not None:
        with mute_logs(patterns=("[kgr:year]", "[kgr:init]"), enabled=(not quiet)):
            kgr_state = kgr_lite_init(
                W0=W0, age0=age0, life_table=life_table_df,
                r_f_real_annual=r_f_real_annual, cfg=kgr_cfg,
            )

    def actor(obs):
        nonlocal kgr_state, _kgr_once_logged
        if isinstance(obs, dict):
            o = dict(obs)
        else:
            o = {
                "W_t": float(getattr(env, "W", W0)),
                "age_years": float(getattr(env, "age_years", age0)),
                "cpi_yoy": float(getattr(env, "cpi_yoy", 0.0)),
                "is_new_year": bool(getattr(env, "is_new_year", False)),
            }
        o.setdefault("life_table", life_table_df)
        o.setdefault("r_f_real_annual", r_f_real_annual)

        if kgr_state is None:
            with mute_logs(patterns=("[kgr:year]", "[kgr:init]"), enabled=(not quiet)):
                kgr_state = kgr_lite_init(
                    W0=float(o.get("W_t", W0)),
                    age0=float(o.get("age_years", age0)),
                    life_table=o.get("life_table", None),
                    r_f_real_annual=float(o.get("r_f_real_annual", r_f_real_annual)),
                    cfg=kgr_cfg,
                )

        if bool(o.get("is_new_year", False)):
            no_lt = (o.get("life_table", None) is None)
            if not _kgr_once_logged and (not quiet):
                print("[kgr:info] life_table 없음 → CPI-only 조정(연 1회) (1회만)" if no_lt
                      else "[kgr:info] life_table 기반 FR 가드레일 적용 (1회만)")
                _kgr_once_logged = True

            with mute_kgr_year_logs_if(no_life_table=no_lt if (not quiet) else False):
                kgr_lite_update_yearly(
                    W_t=float(o.get("W_t", W0)),
                    age_years=float(o.get("age_years", age0)),
                    CPI_yoy=float(o.get("cpi_yoy", 0.0)),
                    life_table=o.get("life_table", None),
                    r_f_real_annual=float(o.get("r_f_real_annual", r_f_real_annual)),
                    state=kgr_state, cfg=kgr_cfg,
                )

        with mute_kgr_year_logs_if(no_life_table=(o.get("life_table", None) is None) if (not quiet) else False):
            out = kgr_lite_policy_step(o, kgr_state, kgr_cfg)

        q_annual = float(out.get("q_t", kgr_cfg.q_floor))
        q_m = 1.0 - (1.0 - q_annual) ** (1.0 / steps_per_year)
        q_floor_cfg = float(getattr(kgr_cfg, "q_floor", 0.02) or 0.02)
        if bool(getattr(kgr_cfg, "q_floor_is_annual", True)):
            q_floor_m = 1.0 - (1.0 - q_floor_cfg) ** (1.0 / steps_per_year)
        else:
            q_floor_m = q_floor_cfg
        q_m = float(_np.clip(q_m, q_floor_m, 1.0))

        try:
            w_fixed_local = float(getattr(kgr_cfg, "w_fixed", 0.6))
        except (TypeError, ValueError):
            w_fixed_local = 0.6
        try:
            w_max = float(getattr(cfg, "w_max", 1.0))
        except (TypeError, ValueError):
            w_max = 1.0
        w = max(0.0, min(w_fixed_local, w_max))
        return _clip_action(q_m, w, cfg)
    return actor


def build_rule_actor(cfg: SimConfig, args, env: RetirementEnv):
    if cfg.baseline == "4pct":
        return rule_actor_4pct(cfg, env)
    if cfg.baseline == "cpb":
        return rule_actor_cpb(cfg, env)
    if cfg.baseline == "vpw":
        return rule_actor_vpw(cfg, env)
    if cfg.baseline == "kgr":
        return rule_actor_kgr(cfg, env, quiet=(getattr(args, "quiet", "on") == "on"))
    raise SystemExit("--baseline required for method=rule (4pct|cpb|vpw|kgr)")


# ---------- HJB ----------
def build_hjb_actor(cfg: SimConfig, args, env: RetirementEnv):
    """
    HJB 정책을 상태 어댑터와 함께 감싸, 입력이 dict/ndarray 무엇이든 안전 처리.
    t 인덱스는 env.t 대신 obs[0]=t_norm 으로 계산.
    + 정책 테이블 NaN/Inf 정화 및 범위 클리핑(방호 강화).
    """
    sol = HJBSolver(cfg).solve(seed=cfg.seeds[0])
    Pi_w = sol.get("Pi_w", None)
    Pi_q = sol.get("Pi_q", None)
    print("policy_hash_q=", arrhash(Pi_q))
    print("policy_hash_w=", arrhash(Pi_w))

    # W 그리드 확보
    if "W_grid" in sol and sol["W_grid"] is not None:
        Wg = _np.asarray(sol["W_grid"], dtype=float)
    else:
        Wg = _np.linspace(float(getattr(cfg, "hjb_W_min", 0.0)),
                          float(getattr(cfg, "hjb_W_max", 1.5)),
                          int(getattr(cfg, "hjb_W_grid", 65) or 65))

    # 정책이 비어있으면 상수 정책으로 폴백
    if Pi_w is None or getattr(Pi_w, "size", 0) == 0 or Pi_q is None or getattr(Pi_q, "size", 0) == 0:
        const_w = float(min(max(float(getattr(cfg, "hjb_w_grid", [0.0, 0.6])[-1]), 0.0),
                            float(getattr(cfg, "w_max", 1.0))))
        q4 = 1.0 - (1.0 - 0.04) ** (1.0 / int(getattr(cfg, "steps_per_year", 12) or 12))
        const_q = float(q4)

        def actor(_obs):
            return _clip_action(const_q, const_w, cfg)
        return actor

    # ---- 정책 테이블 사전 정화 (NaN/Inf → 안전값 + 범위 클립) ----
    def _clean_pi(arr, lo, hi, fill):
        a = _np.asarray(arr, dtype=float)
        a = _np.nan_to_num(a, nan=fill, posinf=hi, neginf=lo)
        return _np.clip(a, lo, hi)

    q_floor_m = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    # 기본치: 월 4%룰, w 기본 0.6 (경계 내)
    q_fill = max(q_floor_m, 0.04 / int(getattr(cfg, "steps_per_year", 12) or 12))
    w_fill = min(0.6, float(getattr(cfg, "w_max", 1.0)))

    Pi_w = _clean_pi(Pi_w, 0.0, float(getattr(cfg, "w_max", 1.0)), w_fill)
    Pi_q = _clean_pi(Pi_q, q_floor_m, 1.0, q_fill)

    # grid 정합성 (최소 2노드)
    Wg = _np.asarray(Wg, dtype=float).ravel()
    if Wg.size < 2 or not _np.isfinite(Wg).all():
        Wg = _np.linspace(float(getattr(cfg, "hjb_W_min", 0.0)),
                          float(getattr(cfg, "hjb_W_max", 1.5)), 2)

    T_pol = int(Pi_w.shape[0])

    def actor(obs):
        # 상태를 [t_norm, W]로 정규화
        s = _as_nd_state(obs, T_hint=T_pol, W_hint=float(getattr(env, "W", 1.0)))
        t_norm = float(_np.clip(s[0], 0.0, 1.0))
        W_now = float(s[1])

        # 시점 인덱스: t_norm ∈ [0,1] → [0, T_pol-1]
        t_idx = int(_np.clip(round(t_norm * (T_pol - 1)), 0, T_pol - 1))

        # W 인덱스: 왼쪽 구간 선택 (searchsorted - 1), 경계 안전
        i = int(_np.clip(_np.searchsorted(Wg, W_now) - 1, 0, max(Wg.size - 2, 0)))

        q = float(Pi_q[t_idx, i])
        w = float(Pi_w[t_idx, i])

        # 최종 방호 (비정상값 기본치 대체 후 클립)
        if not _np.isfinite(q):
            q = q_fill
        if not _np.isfinite(w):
            w = w_fill
        return _clip_action(q, w, cfg)

    return actor


# ---------- RL ----------
def build_rl_actor(cfg: SimConfig, args):
    try:
        from ..rl import A2CTrainer  # noqa
        pol = A2CTrainer(cfg).train(seed=cfg.seeds[0])

        def actor(obs):
            s = _as_nd_state(obs, T_hint=int(getattr(cfg, "horizon_steps", 0) or 0), W_hint=None)
            out = pol.act(s)
            if isinstance(out, dict):
                q, w = float(out.get("q", 0.0)), float(out.get("w", 0.0))
            else:
                q, w = float(out[0]), float(out[1])
            return _clip_action(q, w, cfg)

        return actor
    except Exception:
        raise SystemExit("RL route requires --method rl in main() (trainer moved to project.trainer.rl_a2c).")


# ---------- entry ----------
def build_actor(cfg: SimConfig, args):
    # 공유 probe env (일부 규칙형 정책이 참조)
    env = RetirementEnv(cfg)
    if args.method == "rule":
        return build_rule_actor(cfg, args, env)
    if args.method == "hjb":
        return build_hjb_actor(cfg, args, env)
    if args.method == "rl":
        return build_rl_actor(cfg, args)
    raise SystemExit("Unknown method")
