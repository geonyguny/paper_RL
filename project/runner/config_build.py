# project/runner/config_build.py
from __future__ import annotations
from typing import Any
import numpy as _np
from ..config import SimConfig, ASSET_PRESETS
from .helpers import auto_eta_grid

def make_cfg(args) -> SimConfig:
    cfg = SimConfig()
    if args.asset in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[args.asset].items():  # type: ignore
            setattr(cfg, k, v)
    cfg.asset = args.asset

    bulk = dict(
        w_max=args.w_max, fee_annual=args.fee_annual, phi_adval=args.fee_annual,
        horizon_years=args.horizon_years, lambda_term=args.lambda_term,
        alpha=args.alpha, baseline=args.baseline, p_annual=args.p_annual,
        g_real_annual=args.g_real_annual, w_fixed=args.w_fixed,
        floor_on=args.floor_on, f_min_real=args.f_min_real, F_target=args.F_target,
        hjb_W_grid=args.hjb_W_grid, hjb_Nshock=args.hjb_Nshock,
        hedge=args.hedge, hedge_on=(args.hedge == "on"), hedge_mode=args.hedge_mode,
        hedge_cost=args.hedge_cost, hedge_sigma_k=args.hedge_sigma_k, hedge_tx=args.hedge_tx,
        market_mode=args.market_mode, market_csv=args.market_csv,
        bootstrap_block=args.bootstrap_block, use_real_rf=args.use_real_rf,
        mortality=args.mortality, mortality_on=(args.mortality == "on"),
        mort_table=args.mort_table, age0=args.age0, sex=args.sex,
        bequest_kappa=args.bequest_kappa, bequest_gamma=args.bequest_gamma,
        rl_q_cap=args.rl_q_cap, teacher_eps0=args.teacher_eps0, teacher_decay=args.teacher_decay,
        lw_scale=args.lw_scale, survive_bonus=args.survive_bonus,
        crra_gamma=args.crra_gamma, u_scale=args.u_scale,
        cvar_stage_on=(args.cvar_stage == "on"),
        alpha_stage=args.alpha_stage, lambda_stage=args.lambda_stage,
        cstar_mode=args.cstar_mode, cstar_m=args.cstar_m,
        xai_on=(args.xai_on == "on"),
        q_floor=args.q_floor, beta=args.beta,
        ann_on=args.ann_on, ann_alpha=args.ann_alpha, ann_L=args.ann_L,
        ann_d=args.ann_d, ann_index=args.ann_index,
    )
    for k, v in bulk.items():
        if v is not None:
            setattr(cfg, k, v)

    cfg.seeds = tuple(args.seeds)
    cfg.n_paths_eval = int(args.n_paths)
    cfg.outputs = args.outputs
    cfg.method = args.method
    cfg.es_mode = args.es_mode

    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    if getattr(args, "q_floor", None) is not None:
        q_floor_ann = float(args.q_floor)
        q_floor_m = 1.0 - (1.0 - max(min(q_floor_ann, 0.999999), 0.0)) ** (1.0 / spm)
        setattr(cfg, "q_floor", float(q_floor_m))
        setattr(cfg, "q_floor_annual", float(q_floor_ann))
        print(f"[cfg] q_floor_annual={q_floor_ann:.6f} → q_floor_monthly={q_floor_m:.6f} (steps_per_year={spm})")

    auto_eta_grid(cfg, requested_n=args.hjb_eta_n)
    if hasattr(cfg, "hjb_w_grid") and (getattr(cfg, "hjb_w_grid", None) in (None, ())):
        n_w = 8
        cfg.hjb_w_grid = tuple(_np.round(_np.linspace(0.0, cfg.w_max, n_w), 2))

    setattr(cfg, "dev_split_w_grid", False)
    setattr(cfg, "hjb_Nshock", max(int(getattr(cfg, "hjb_Nshock", 32) or 32), 256))
    return cfg
