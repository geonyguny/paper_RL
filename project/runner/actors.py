# project/runner/actors.py
from __future__ import annotations
from typing import Any, Tuple, Callable
import numpy as _np

from ..config import SimConfig
from ..env import RetirementEnv
from ..hjb import HJBSolver
from .helpers import arrhash, monthly_from_cfg, get_life_table_from_env
from .logging_filters import mute_logs, mute_kgr_year_logs_if

# K-GR rule
from ..policy.kgr_rule import (
    KGRLiteConfig, kgr_lite_init, kgr_lite_update_yearly, kgr_lite_policy_step,
)

# ---------- RULE ----------
def rule_actor_4pct(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def actor(_obs):
        q_m = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
        w = cfg.w_fixed if cfg.w_fixed is not None else cfg.w_max
        return q_m, w
    return actor

def rule_actor_cpb(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def actor(_obs):
        _g_m, p_m = monthly_from_cfg(cfg)
        q_m = p_m
        w = cfg.w_fixed if cfg.w_fixed is not None else cfg.w_max
        return q_m, w
    return actor

def rule_actor_vpw(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def _get_g_m(_cfg):
        try:
            if hasattr(_cfg, "monthly") and callable(_cfg.monthly):
                m = _cfg.monthly()
                if isinstance(m, dict) and "g_m" in m:
                    gm = float(m["g_m"])
                    if _np.isfinite(gm): return gm
        except Exception:
            pass
        g_ann = float(getattr(_cfg, "g_real_annual", 0.0) or 0.0)
        spm = int(getattr(_cfg, "steps_per_year", 12) or 12)
        return (1.0 + g_ann)**(1.0 / spm) - 1.0

    def actor(_obs):
        t = int(getattr(env, "t", 0)); T = int(getattr(env, "T", 1))
        Nm = max(T - t, 1)
        g_m = _get_g_m(cfg)
        if abs(g_m) < 1e-12:
            q_m = 1.0 / Nm
        else:
            a = (1.0 - (1.0 + g_m) ** (-Nm)) / g_m
            q_m = 1.0 / max(a, 1e-12)
        q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
        q_m = float(_np.clip(q_m, q_floor, 1.0))
        w = float(cfg.w_fixed if getattr(cfg, "w_fixed", None) is not None else cfg.w_max)
        return q_m, w
    return actor

def rule_actor_kgr(cfg: SimConfig, env: RetirementEnv, *, quiet: bool) -> Callable[[Any], Tuple[float, float]]:
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    q_floor = float(getattr(cfg, "q_floor", 0.02) or 0.02)
    fee_annual = float(getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", 0.004)) or 0.004)
    w_fixed = float(getattr(cfg, "w_fixed", None) if getattr(cfg, "w_fixed", None) is not None else cfg.w_max)

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

        try: w_fixed_local = float(getattr(kgr_cfg, "w_fixed", 0.6))
        except (TypeError, ValueError): w_fixed_local = 0.6
        try: w_max = float(getattr(cfg, "w_max", 1.0))
        except (TypeError, ValueError): w_max = 1.0
        w = max(0.0, min(w_fixed_local, w_max))
        return q_m, w
    return actor

def build_rule_actor(cfg: SimConfig, args, env: RetirementEnv):
    if cfg.baseline == "4pct": return rule_actor_4pct(cfg, env)
    if cfg.baseline == "cpb":  return rule_actor_cpb(cfg, env)
    if cfg.baseline == "vpw":  return rule_actor_vpw(cfg, env)
    if cfg.baseline == "kgr":  return rule_actor_kgr(cfg, env, quiet=(getattr(args, "quiet", "on") == "on"))
    raise SystemExit("--baseline required for method=rule (4pct|cpb|vpw|kgr)")

# ---------- HJB ----------
def build_hjb_actor(cfg: SimConfig, args, env: RetirementEnv):
    sol = HJBSolver(cfg).solve(seed=cfg.seeds[0])
    Pi_w = sol.get('Pi_w', None); Pi_q = sol.get('Pi_q', None)
    print("policy_hash_q=", arrhash(Pi_q)); print("policy_hash_w=", arrhash(Pi_w))

    if 'W_grid' in sol and sol['W_grid'] is not None:
        Wg = _np.asarray(sol['W_grid'], dtype=float)
    else:
        Wg = _np.linspace(cfg.hjb_W_min, cfg.hjb_W_max, cfg.hjb_W_grid)

    if Pi_w is None or getattr(Pi_w, 'size', 0) == 0 or Pi_q is None or getattr(Pi_q, 'size', 0) == 0:
        const_w = float(min(max(cfg.hjb_w_grid[-1], 0.0), cfg.w_max))
        q4 = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
        const_q = float(q4)
        def actor(_obs): return const_q, const_w
        return actor

    T_pol = int(Pi_w.shape[0])
    def actor(obs):
        t_idx = int(_np.clip(getattr(env, "t", 0), 0, T_pol - 1))
        W = float(obs[1])
        i = int(_np.clip(_np.searchsorted(Wg, W) - 1, 0, Wg.size - 2))
        w = float(Pi_w[t_idx, i]); q = float(Pi_q[t_idx, i])
        return q, w
    return actor

# ---------- RL ----------
def build_rl_actor(cfg: SimConfig, args):
    try:
        from ..rl import A2CTrainer  # noqa
        pol = A2CTrainer(cfg).train(seed=cfg.seeds[0])
        def actor(obs): return pol.act(obs)
        return actor
    except Exception:
        raise SystemExit("RL route requires --method rl in main() (trainer moved to project.trainer.rl_a2c).")

# ---------- entry ----------
def build_actor(cfg: SimConfig, args):
    env = RetirementEnv(cfg)  # shared probe for closures
    if args.method == "rule": return build_rule_actor(cfg, args, env)
    if args.method == "hjb":  return build_hjb_actor(cfg, args, env)
    if args.method == "rl":   return build_rl_actor(cfg, args)
    raise SystemExit("Unknown method")
