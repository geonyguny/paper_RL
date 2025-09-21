# project/run_experiment.py
from __future__ import annotations

import argparse, json, os, hashlib, sys, csv, datetime, io, contextlib
from typing import Optional, Callable, Tuple, Any, Dict

import numpy as _np
import pandas as pd

from .config import (
    SimConfig, ASSET_PRESETS,
    CVAR_TARGET_DEFAULT, CVAR_TOL_DEFAULT, LAMBDA_MIN_DEFAULT, LAMBDA_MAX_DEFAULT
)
from .env import RetirementEnv
from .hjb import HJBSolver
from .eval import evaluate

# K-GR rule
from .policy.kgr_rule import (
    KGRLiteConfig, kgr_lite_init, kgr_lite_update_yearly, kgr_lite_policy_step,
)

# [ANN] annuity overlay (MVP)
from .annuity.overlay import AnnuityConfig, init_annuity

# ---------- Fast help short-circuit (CI friendliness) ----------
if any(x in sys.argv for x in ("-h", "--help")):
    argparse.ArgumentParser(add_help=True, description="Run experiments").print_help()
    raise SystemExit(0)

# ---------- Optional autosave ----------
try:
    from .eval import save_metrics_autocsv  # noqa
    _HAS_AUTOSAVE = True
except Exception:
    _HAS_AUTOSAVE = False

# ---------- stdout UTF-8 ----------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ============================================================================
#  LOG FILTERS (inline; 쉽게 utils로 이동 가능)
# ============================================================================
class _DevNull:
    def __init__(self): self.encoding = "utf-8"
    def write(self, s): return len(s)
    def flush(self): pass
    def isatty(self): return False

@contextlib.contextmanager
def _silence_stdio(also_stderr=True):
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _DevNull()
        if also_stderr:
            sys.stderr = _DevNull()
        yield
    finally:
        sys.stdout = saved_out
        if also_stderr:
            sys.stderr = saved_err

class _LineFilterWriter:
    def __init__(self, underlying, patterns):
        self._u = underlying
        self._buf = io.StringIO()
        self._pat = tuple(patterns)
    def write(self, s):
        self._buf.write(s)
        text = self._buf.getvalue()
        if "\n" not in text:
            return len(s)
        lines = text.splitlines(keepends=True)
        if not text.endswith("\n"):
            self._buf = io.StringIO(); self._buf.write(lines[-1]); lines = lines[:-1]
        else:
            self._buf = io.StringIO()
        kept = []
        for ln in lines:
            if any(p in ln for p in self._pat):
                continue
            kept.append(ln)
        if kept:
            self._u.write("".join(kept))
        return len(s)
    def flush(self):
        tail = self._buf.getvalue()
        if tail and not any(p in tail for p in self._pat):
            self._u.write(tail)
        self._u.flush()

@contextlib.contextmanager
def _mute_logs(patterns=("[kgr:year]",), enabled=True, streams=("stdout", "stderr")):
    if not enabled:
        yield; return
    saved = {}
    try:
        if "stdout" in streams:
            saved["stdout"] = sys.stdout
            sys.stdout = _LineFilterWriter(sys.stdout, patterns)
        if "stderr" in streams:
            saved["stderr"] = sys.stderr
            sys.stderr = _LineFilterWriter(sys.stderr, patterns)
        yield
    finally:
        if "stdout" in saved:
            try: sys.stdout.flush()
            except Exception: pass
            sys.stdout = saved["stdout"]
        if "stderr" in saved:
            try: sys.stderr.flush()
            except Exception: pass
            sys.stderr = saved["stderr"]

def _mute_kgr_year_logs_if(*, no_life_table: bool):
    return _mute_logs(patterns=("[kgr:year]",), enabled=bool(no_life_table))


# ============================================================================
#  SMALL HELPERS (hash, grids, life-table, monthly conversions)
# ============================================================================
def _arrhash(a) -> str:
    if a is None: return "none"
    x = _np.asarray(a, dtype=_np.float32).tobytes()
    return hashlib.md5(x).hexdigest()

def _auto_eta_grid(cfg: SimConfig, requested_n: int | None = None) -> None:
    cur = getattr(cfg, "hjb_eta_grid", ())
    if float(getattr(cfg, "lambda_term", 0.0) or 0.0) <= 0.0:
        if not cur or len(cur) <= 1:
            cfg.hjb_eta_grid = (0.0,)
        return
    n = int(requested_n or getattr(cfg, "hjb_eta_n", 41) or 41)
    F = float(getattr(cfg, "F_target", 0.0) or 0.0)
    F = F if F > 0.0 else 1.0
    try:
        cfg.hjb_eta_grid = tuple(_np.linspace(0.0, F, n))
    except Exception:
        step = F / max(n - 1, 1)
        cfg.hjb_eta_grid = tuple(0.0 + i * step for i in range(n))

def _get_life_table_from_env(env) -> Optional[pd.DataFrame]:
    lt = getattr(env, "life_table", None)
    if isinstance(lt, pd.DataFrame) and not lt.empty:
        return lt
    lt2 = getattr(env, "mort_table_df", None)
    if isinstance(lt2, pd.DataFrame) and not lt2.empty:
        return lt2
    return None

def _monthly_from_cfg(cfg: SimConfig) -> Tuple[float, float]:
    if hasattr(cfg, "monthly"):
        try:
            m = cfg.monthly()
            return float(m.get("g_m", 0.0)), float(m.get("p_m", 0.0))
        except Exception:
            pass
    steps = int(getattr(cfg, "steps_per_year", 12))
    g_ann = float(getattr(cfg, "g_real_annual", 0.0))
    p_ann = float(getattr(cfg, "p_annual", 0.0))
    g_m = (1.0 + g_ann) ** (1.0 / max(steps, 1)) - 1.0
    p_m = (1.0 + p_ann) ** (1.0 / max(steps, 1)) - 1.0
    return g_m, p_m


# ============================================================================
#  CONFIG BUILD
# ============================================================================
def make_cfg(args) -> SimConfig:
    cfg = SimConfig()
    if args.asset in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[args.asset].items():  # type: ignore
            setattr(cfg, k, v)
    cfg.asset = args.asset

    # bulk-set
    for k, v in dict(
        w_max=args.w_max, fee_annual=args.fee_annual, phi_adval=args.fee_annual,
        horizon_years=args.horizon_years, lambda_term=args.lambda_term,
        alpha=args.alpha, baseline=args.baseline, p_annual=args.p_annual,
        g_real_annual=args.g_real_annual, w_fixed=args.w_fixed,
        floor_on=args.floor_on, f_min_real=args.f_min_real, F_target=args.F_target,
        hjb_W_grid=args.hjb_W_grid, hjb_Nshock=args.hjb_Nshock,
        # hedge
        hedge=args.hedge, hedge_on=(args.hedge == "on"), hedge_mode=args.hedge_mode,
        hedge_cost=args.hedge_cost, hedge_sigma_k=args.hedge_sigma_k, hedge_tx=args.hedge_tx,
        # market
        market_mode=args.market_mode, market_csv=args.market_csv,
        bootstrap_block=args.bootstrap_block, use_real_rf=args.use_real_rf,
        # mortality
        mortality=args.mortality, mortality_on=(args.mortality == "on"),
        mort_table=args.mort_table, age0=args.age0, sex=args.sex,
        bequest_kappa=args.bequest_kappa, bequest_gamma=args.bequest_gamma,
        # RL shaping
        rl_q_cap=args.rl_q_cap, teacher_eps0=args.teacher_eps0, teacher_decay=args.teacher_decay,
        lw_scale=args.lw_scale, survive_bonus=args.survive_bonus,
        crra_gamma=args.crra_gamma, u_scale=args.u_scale,
        # stage-wise
        cvar_stage_on=(args.cvar_stage == "on"),
        alpha_stage=args.alpha_stage, lambda_stage=args.lambda_stage,
        cstar_mode=args.cstar_mode, cstar_m=args.cstar_m,
        # XAI
        xai_on=(args.xai_on == "on"),
        # raw
        q_floor=args.q_floor, beta=args.beta,
        # [ANN] pre-wire flags (values finalized by overlay setup)
        ann_on=args.ann_on, ann_alpha=args.ann_alpha, ann_L=args.ann_L,
        ann_d=args.ann_d, ann_index=args.ann_index,
    ).items():
        if v is not None:
            setattr(cfg, k, v)

    cfg.seeds = tuple(args.seeds)
    cfg.n_paths_eval = int(args.n_paths)
    cfg.outputs = args.outputs
    cfg.method = args.method
    cfg.es_mode = args.es_mode

    # q_floor: annual → monthly
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    if getattr(args, "q_floor", None) is not None:
        q_floor_ann = float(args.q_floor)
        q_floor_m = 1.0 - (1.0 - max(min(q_floor_ann, 0.999999), 0.0)) ** (1.0 / spm)
        setattr(cfg, "q_floor", float(q_floor_m))
        setattr(cfg, "q_floor_annual", float(q_floor_ann))
        print(f"[cfg] q_floor_annual={q_floor_ann:.6f} → q_floor_monthly={q_floor_m:.6f} (steps_per_year={spm})")

    _auto_eta_grid(cfg, requested_n=args.hjb_eta_n)
    if hasattr(cfg, "hjb_w_grid") and (getattr(cfg, "hjb_w_grid", None) in (None, ())):
        n_w = 8
        cfg.hjb_w_grid = tuple(_np.round(_np.linspace(0.0, cfg.w_max, n_w), 2))

    setattr(cfg, "dev_split_w_grid", False)
    setattr(cfg, "hjb_Nshock", max(int(getattr(cfg, "hjb_Nshock", 32) or 32), 256))
    return cfg


# ============================================================================
#  ACTORS (rule/hjb/rl)
# ============================================================================
def _build_rule_actor_4pct(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def actor(_obs):
        q_m = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
        w = cfg.w_fixed if cfg.w_fixed is not None else cfg.w_max
        return q_m, w
    return actor

def _build_rule_actor_cpb(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def actor(_obs):
        _g_m, p_m = _monthly_from_cfg(cfg)
        q_m = p_m
        w = cfg.w_fixed if cfg.w_fixed is not None else cfg.w_max
        return q_m, w
    return actor

def _build_rule_actor_vpw(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def _get_g_m_from_cfg(_cfg):
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
        T = int(getattr(env, "T", 1))
        Nm = max(T - t, 1)
        g_m = _get_g_m_from_cfg(cfg)
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

def _build_rule_actor_kgr(cfg: SimConfig, env: RetirementEnv, quiet: bool) -> Callable[[Any], Tuple[float, float]]:
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    q_floor = float(getattr(cfg, "q_floor", 0.02) or 0.02)
    fee_annual = float(getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", 0.004)) or 0.004)
    w_fixed = float(getattr(cfg, "w_fixed", None) if getattr(cfg, "w_fixed", None) is not None else cfg.w_max)

    life_table_df = _get_life_table_from_env(env)
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
        with _mute_logs(patterns=("[kgr:year]", "[kgr:init]"), enabled=(not quiet)):
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
            with _mute_logs(patterns=("[kgr:year]", "[kgr:init]"), enabled=(not quiet)):
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

            with _mute_kgr_year_logs_if(no_life_table=no_lt if (not quiet) else False):
                kgr_lite_update_yearly(
                    W_t=float(o.get("W_t", W0)),
                    age_years=float(o.get("age_years", age0)),
                    CPI_yoy=float(o.get("cpi_yoy", 0.0)),
                    life_table=o.get("life_table", None),
                    r_f_real_annual=float(o.get("r_f_real_annual", r_f_real_annual)),
                    state=kgr_state, cfg=kgr_cfg,
                )

        with _mute_kgr_year_logs_if(no_life_table=(o.get("life_table", None) is None) if (not quiet) else False):
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

def _build_rule_actor(cfg: SimConfig, args, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    if cfg.baseline == "4pct":
        return _build_rule_actor_4pct(cfg, env)
    if cfg.baseline == "cpb":
        return _build_rule_actor_cpb(cfg, env)
    if cfg.baseline == "vpw":
        return _build_rule_actor_vpw(cfg, env)
    if cfg.baseline == "kgr":
        return _build_rule_actor_kgr(cfg, env, quiet=(getattr(args, "quiet", "on") == "on"))
    raise SystemExit("--baseline required for method=rule (4pct|cpb|vpw|kgr)")

def _build_hjb_actor(cfg: SimConfig, args, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    sol = HJBSolver(cfg).solve(seed=cfg.seeds[0])
    Pi_w = sol.get('Pi_w', None)
    Pi_q = sol.get('Pi_q', None)
    print("policy_hash_q=", _arrhash(Pi_q))
    print("policy_hash_w=", _arrhash(Pi_w))

    if 'eta' in sol:
        try: print("eta_selected=", float(sol['eta']))
        except Exception: pass
        try:
            if Pi_w is not None and Pi_q is not None and getattr(Pi_w, "size", 0) > 0 and getattr(Pi_q, "size", 0) > 0:
                _w = _np.asarray(Pi_w); _q = _np.asarray(Pi_q)
                print("w_stats[min,mean,max]=", [float(_w.min()), float(_w.mean()), float(_w.max())])
                print("q_stats[min,mean,max]=", [float(_q.min()), float(_q.mean()), float(_q.max())])
                w_t0 = _w[0] if _w.ndim >= 1 else _w
                q_t0 = _q[0] if _q.ndim >= 1 else _q
                print("unique_w_t0 (rounded):", _np.unique(_np.round(w_t0, 3))[:10])
                print("unique_q_t0 (rounded):", _np.unique(_np.round(q_t0, 3))[:10])
        except Exception as _e:
            print(f"[dbg] policy stats skipped: {_e}")

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

def _build_rl_actor(cfg: SimConfig, args) -> Callable[[Any], Tuple[float, float]]:
    try:
        from .rl import A2CTrainer  # noqa
        pol = A2CTrainer(cfg).train(seed=cfg.seeds[0])
        def actor(obs): return pol.act(obs)
        return actor
    except Exception:
        raise SystemExit("RL route requires --method rl in main() (trainer moved to project.trainer.rl_a2c).")

def build_actor(cfg: SimConfig, args):
    env = RetirementEnv(cfg)  # shared probe env for actor closures
    if args.method == "rule":
        return _build_rule_actor(cfg, args, env)
    if args.method == "hjb":
        return _build_hjb_actor(cfg, args, env)
    if args.method == "rl":
        return _build_rl_actor(cfg, args)
    raise SystemExit("Unknown method")


# ============================================================================
#  IO / AUTOSAVE
# ============================================================================
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def now_iso() -> str: return datetime.datetime.now().isoformat(timespec="seconds")

def slim_args(args) -> dict:
    keys = [
        "asset", "method", "baseline", "w_max", "fee_annual", "horizon_years",
        "alpha", "lambda_term", "F_target", "p_annual", "g_real_annual",
        "w_fixed", "floor_on", "f_min_real", "es_mode", "outputs",
        "hjb_W_grid", "hjb_Nshock", "hjb_eta_n",
        "hedge", "hedge_mode", "hedge_cost", "hedge_sigma_k", "hedge_tx",
        "market_mode", "market_csv", "bootstrap_block", "use_real_rf",
        "mortality", "mort_table", "age0", "sex", "bequest_kappa", "bequest_gamma",
        "cvar_stage", "alpha_stage", "lambda_stage", "cstar_mode", "cstar_m",
        "rl_q_cap", "teacher_eps0", "teacher_decay", "lw_scale", "survive_bonus",
        "crra_gamma", "u_scale", "xai_on",
        "seeds", "n_paths",
        "rl_epochs", "rl_steps_per_epoch", "rl_n_paths_eval", "gae_lambda",
        "entropy_coef", "value_coef", "lr", "max_grad_norm",
        "q_floor", "beta", "quiet",
        # [ANN]
        "ann_on", "ann_alpha", "ann_L", "ann_d", "ann_index",
    ]
    return {k: getattr(args, k, None) for k in keys}

def append_metrics_csv(path: str, payload: dict):
    row = {
        'ts': now_iso(),
        'asset': payload.get('asset'),
        'method': payload.get('method'),
        'lambda': payload.get('lambda_term'),
        'F_target': payload.get('F_target'),
        'alpha': payload.get('alpha'),
        'ES95': (payload.get('metrics') or {}).get('ES95'),
        'EW': (payload.get('metrics') or {}).get('EW'),
        'Ruin': (payload.get('metrics') or {}).get('Ruin'),
        'mean_WT': (payload.get('metrics') or {}).get('mean_WT'),
        'hedge_on': (payload.get('args') or {}).get('hedge') == 'on',
        'hedge_mode': (payload.get('args') or {}).get('hedge_mode'),
        'fee_annual': payload.get('fee_annual'),
        'w_max': payload.get('w_max'),
        'horizon_years': payload.get('horizon_years'),
        'seeds': (payload.get('args') or {}).get('seeds'),
        'n_paths': (payload.get('args') or {}).get('n_paths'),
        'mortality_on': (payload.get('args') or {}).get('mortality') == 'on',
        'market_mode': (payload.get('args') or {}).get('market_mode'),
        # [ANN]
        'ann_on': (payload.get('args') or {}).get('ann_on'),
        'ann_alpha': (payload.get('args') or {}).get('ann_alpha'),
        'ann_L': (payload.get('args') or {}).get('ann_L'),
        'ann_d': (payload.get('args') or {}).get('ann_d'),
        'ann_index': (payload.get('args') or {}).get('ann_index'),
        'y_ann': (payload.get('metrics') or {}).get('y_ann'),
        'a_factor': (payload.get('metrics') or {}).get('a_factor'),
        'P': (payload.get('metrics') or {}).get('P'),
    }
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)


# ============================================================================
#  ANNUITY OVERLAY WIRING
# ============================================================================
def _setup_annuity_overlay(cfg: SimConfig, args):
    """
    - 한 번 probe env를 띄워 life_table / r_f_real_annual 확보
    - init_annuity로 (W0_after, y_ann, a_factor, P) 계산
    - cfg에 반영 (evaluate가 새 env를 만들 때 적용)
    """
    if getattr(args, "ann_on", "off") != "on" or float(getattr(args, "ann_alpha", 0.0)) <= 0.0:
        setattr(cfg, "ann_on", "off"); setattr(cfg, "y_ann", 0.0)
        setattr(cfg, "ann_P", 0.0); setattr(cfg, "ann_a_factor", 0.0)
        return None

    probe = RetirementEnv(cfg)
    life_df = _get_life_table_from_env(probe)
    age0 = int(getattr(cfg, "age0", 65) or 65)
    if hasattr(probe, "r_f_real_annual") and probe.r_f_real_annual is not None:
        r_annual = float(probe.r_f_real_annual)
    else:
        r_m = float(_np.mean(getattr(probe, "path_safe", _np.array([0.0]))))
        r_annual = (1.0 + r_m) ** int(getattr(cfg, "steps_per_year", 12) or 12) - 1.0

    if life_df is None:
        # MVP: life table 없으면 annuity overlay 비활성 (안전동작)
        setattr(cfg, "ann_on", "off"); setattr(cfg, "y_ann", 0.0)
        setattr(cfg, "ann_P", 0.0); setattr(cfg, "ann_a_factor", 0.0)
        return None

    ann_cfg = AnnuityConfig(
        on=True,
        alpha=float(args.ann_alpha), L=float(args.ann_L),
        d=int(args.ann_d), index=str(args.ann_index),
    )

    W0 = float(getattr(probe, "W0", 1.0))
    W0_after, st = init_annuity(W0, ann_cfg, age0, life_df, r_annual)

    setattr(cfg, "ann_on", "on")
    setattr(cfg, "W0", float(W0_after))
    setattr(cfg, "y_ann", float(st.y_ann))
    setattr(cfg, "ann_P", float(st.P))
    setattr(cfg, "ann_a_factor", float(st.a_factor))
    return st


# ============================================================================
#  RUNNERS
# ============================================================================
def run_once(args):
    quiet_ctx = _silence_stdio(also_stderr=True) if getattr(args, "quiet", "on") == "on" else contextlib.nullcontext()
    with quiet_ctx:
        cfg = make_cfg(args)
        ensure_dir(args.outputs)

        # [ANN] overlay 적용 (cfg.W0, cfg.y_ann 반영)
        ann_state = None
        if getattr(args, "ann_on", "off") == "on" and float(getattr(args, "ann_alpha", 0.0)) > 0.0:
            ann_state = _setup_annuity_overlay(cfg, args)

        actor = build_actor(cfg, args)
        m = evaluate(cfg, actor, es_mode=args.es_mode)

    if ann_state is not None:
        m = dict(m) if isinstance(m, dict) else {"_": m}
        m.update({
            "y_ann": float(getattr(cfg, "y_ann", 0.0)),
            "a_factor": float(getattr(cfg, "ann_a_factor", 0.0)),
            "P": float(getattr(cfg, "ann_P", 0.0)),
        })

    out = dict(
        asset=cfg.asset, method=args.method, baseline=args.baseline, metrics=m,
        w_max=cfg.w_max, fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term, alpha=cfg.alpha, F_target=cfg.F_target, es_mode=args.es_mode,
        n_paths=cfg.n_paths_eval * len(cfg.seeds),
        args=slim_args(args),
    )
    if args.autosave == "on":
        try:
            if _HAS_AUTOSAVE:
                csv_path = save_metrics_autocsv(m, cfg, outputs=cfg.outputs)
                print(f"[autosave] metrics -> {csv_path}")
            else:
                ensure_dir(os.path.join(cfg.outputs, "_logs"))
                csv_path = os.path.join(cfg.outputs, "_logs", "metrics.csv")
                append_metrics_csv(csv_path, out)
                print(f"[autosave:fallback] metrics -> {csv_path}")
        except Exception as e:
            print(f"[autosave] skipped: {e}")
    return out

def run_rl(args):
    cfg = make_cfg(args)
    ensure_dir(args.outputs)

    if getattr(args, "ann_on", "off") == "on" and float(getattr(args, "ann_alpha", 0.0)) > 0.0:
        _setup_annuity_overlay(cfg, args)

    try:
        from .trainer.rl_a2c import train_rl  # new trainer route
    except Exception as e:
        raise SystemExit(f"RL trainer import failed: {e}")

    fields = train_rl(
        cfg,
        seed_list=args.seeds,
        outputs=args.outputs,
        n_paths_eval=args.rl_n_paths_eval,
        rl_epochs=args.rl_epochs,
        steps_per_epoch=args.rl_steps_per_epoch,
        lr=args.lr,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    out = dict(
        asset=cfg.asset, method="rl", baseline="",
        metrics={"EW": fields.get("EW"), "ES95": fields.get("ES95"),
                 "Ruin": fields.get("Ruin"), "mean_WT": fields.get("mean_WT")},
        w_max=cfg.w_max,
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term, alpha=cfg.alpha, F_target=cfg.F_target, es_mode="loss",
        n_paths=args.rl_n_paths_eval * len(args.seeds),
        args=slim_args(args),
    )
    return out


# ============================================================================
#  CVaR λ calibration
# ============================================================================
def copy_args(args, **overrides):
    from argparse import Namespace
    d = vars(args).copy(); d.update(overrides)
    return Namespace(**d)

def calibrate_lambda(args):
    lo, hi = float(args.lambda_min), float(args.lambda_max)
    target = float(args.cvar_target); tol = float(args.cvar_tol)
    max_iter = int(getattr(args, "calib_max_iter", 8))
    use_fast = (getattr(args, "calib_fast", "on") == "on")

    history = []; cache: Dict[Tuple[float,bool], Tuple[dict, Optional[float]]] = {}

    def eval_at(lmbd: float, fast: bool = True):
        key = (lmbd, fast)
        if key in cache: return cache[key]
        overrides = dict(lambda_term=float(lmbd), es_mode="loss")
        if fast and use_fast:
            overrides.update(dict(hjb_W_grid=81, hjb_Nshock=128, hjb_eta_n=41,
                                  n_paths=150, seeds=[args.seeds[0]]))
        local = copy_args(args, **overrides)
        res = run_once(local)
        es = (res.get('metrics') or {}).get('ES95')
        cache[key] = (res, es)
        return cache[key]

    res_lo, es_lo = eval_at(lo, fast=True)
    res_hi, es_hi = eval_at(hi, fast=True)

    if (es_lo is not None) and (es_hi is not None) and (es_lo <= target) and (es_hi <= target):
        final_res, final_es = eval_at(lo, fast=False)
        final_res['cvar_calibration'] = {
            'selected_lambda': float(lo),
            'selected_ES95': float(final_es) if final_es is not None else None,
            'cvar_target': target, 'cvar_tol': tol,
            'lambda_min': float(args.lambda_min), 'lambda_max': float(args.lambda_max),
            'iterations': 1, 'status': 'already_below_target',
            'history_tail': [{'lambda': float(lo), 'ES95': float(es_lo)}],
        }
        return final_res

    expand = 0
    while (es_lo is not None) and (es_hi is not None) and (es_lo > target) and (es_hi > target) and expand < 3:
        hi *= 2.0
        res_hi, es_hi = eval_at(hi, fast=True)
        expand += 1

    best = (lo, res_lo, es_lo); prev_es = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        res_mid, es_mid = eval_at(mid, fast=True)
        history.append({'lambda': float(mid), 'ES95': float(es_mid) if es_mid is not None else None})
        if prev_es is not None and es_mid is not None and abs(es_mid - prev_es) < 1e-4:
            best = (mid, res_mid, es_mid); status = 'plateau'; break
        prev_es = es_mid
        if es_mid is None:
            lo = mid; best = (mid, res_mid, es_mid); status = 'incomplete'; continue
        if abs(es_mid - target) <= tol:
            best = (mid, res_mid, es_mid); status = 'ok'; break
        if es_mid > target: lo = mid
        else: hi = mid
        best = (mid, res_mid, es_mid); status = 'ok'

    chosen_lambda, _, _ = best
    final_res, final_es = eval_at(chosen_lambda, fast=False)
    final_res['cvar_calibration'] = {
        'selected_lambda': float(chosen_lambda),
        'selected_ES95': float(final_es) if final_es is not None else None,
        'cvar_target': target, 'cvar_tol': tol,
        'lambda_min': float(args.lambda_min), 'lambda_max': float(args.lambda_max),
        'iterations': len(history),
        'status': status if 'status' in locals() else 'ok',
        'history_tail': history[-5:],
    }
    return final_res


# ============================================================================
#  CLI
# ============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
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
    p.add_argument("--cstar_m", type=float, default=0.04/12)

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

    return p

def main():
    p = _build_arg_parser()
    args = p.parse_args()

    if args.method == "rl":
        out = run_rl(args)
    elif args.method == "hjb" and (args.cvar_target is not None):
        out = calibrate_lambda(args)
    else:
        out = run_once(args)

    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
