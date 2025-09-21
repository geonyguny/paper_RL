# project/runner/annuity_wiring.py
from __future__ import annotations
from typing import Optional
import numpy as _np

from ..env import RetirementEnv
from ..annuity.overlay import AnnuityConfig, init_annuity
from .helpers import get_life_table_from_env
from ..config import SimConfig

def setup_annuity_overlay(cfg: SimConfig, args):
    if getattr(args, "ann_on", "off") != "on" or float(getattr(args, "ann_alpha", 0.0)) <= 0.0:
        setattr(cfg, "ann_on", "off"); setattr(cfg, "y_ann", 0.0)
        setattr(cfg, "ann_P", 0.0); setattr(cfg, "ann_a_factor", 0.0)
        return None

    probe = RetirementEnv(cfg)
    life_df = get_life_table_from_env(probe)
    if life_df is None:
        # MVP: 생명표 없으면 annuity off (안전 동작)
        setattr(cfg, "ann_on", "off"); setattr(cfg, "y_ann", 0.0)
        setattr(cfg, "ann_P", 0.0); setattr(cfg, "ann_a_factor", 0.0)
        return None

    age0 = int(getattr(cfg, "age0", 65) or 65)
    if getattr(probe, "r_f_real_annual", None) is not None:
        r_annual = float(probe.r_f_real_annual)
    else:
        r_m = float(_np.mean(getattr(probe, "path_safe", _np.array([0.0]))))
        r_annual = (1.0 + r_m) ** int(getattr(cfg, "steps_per_year", 12) or 12) - 1.0

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
