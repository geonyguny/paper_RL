# project/annuity/overlay.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple

@dataclass
class AnnuityConfig:
    on: bool
    alpha: float   # purchase fraction of W0
    L: float       # load
    d: int         # deferral, MVP=0
    index: str     # 'real'|'nominal'

@dataclass
class AnnuityState:
    purchased: bool
    P: float        # premium paid at t=0
    y_ann: float    # monthly payout
    a_factor: float # annuity-immediate factor (monthly)
    t_star: int     # MVP:0

def _monthly_survival_from_life_table(age0:int, lt:pd.DataFrame, S:int=12, max_age:int=110) -> np.ndarray:
    # lt: columns include 'age' and either 'qx' or 'px'
    lt = lt.copy()
    if 'px' not in lt.columns:
        # make px from qx if needed
        qx = lt['qx'].to_numpy(dtype=float)
        px = 1.0 - np.clip(qx, 0.0, 1.0)
    else:
        px = lt['px'].to_numpy(dtype=float)
    ages = lt['age'].to_numpy(dtype=int)
    # force of mortality per year (UDD): mu_x = -ln(px)
    mu = -np.log(np.clip(px, 1e-12, 1.0))
    # build monthly survival from age0 to max_age under UDD
    S_m = []
    age_max = min(max_age, int(ages.max()))
    # align index of age0
    start_idx = int(np.searchsorted(ages, age0))
    alive = 1.0
    for a_idx in range(start_idx, len(ages)):
        mu_y = float(mu[a_idx])
        for _m in range(S):
            # monthly px under UDD: exp(-mu_y/S)
            p_m = float(np.exp(-mu_y / S))
            S_m.append(alive * p_m)
            alive *= p_m
        if ages[a_idx] >= age_max:
            break
    return np.array(S_m, dtype=float)

def compute_ax_real(age0_years:int, life_table:pd.DataFrame, r_f_real_annual:float, S:int=12) -> float:
    """Real annuity-immediate factor with monthly payments (first at month 1)."""
    i_m = (1.0 + float(r_f_real_annual))**(1.0/S) - 1.0
    v = 1.0 / (1.0 + i_m)
    surv = _monthly_survival_from_life_table(age0_years, life_table, S=S)
    # annuity-immediate: sum_{m=1..} v^m * {survive to end of month m}
    disc = v ** np.arange(1, len(surv)+1, dtype=float)
    a = float(np.sum(disc * surv))
    return max(a, 1e-9)

def init_annuity(W0:float, cfg:AnnuityConfig, age0_years:int, life_table, r_f_real_annual:float, S:int=12) -> Tuple[float, AnnuityState]:
    if (not cfg.on) or (cfg.alpha <= 0.0):
        return W0, AnnuityState(False, 0.0, 0.0, 0.0, -1)
    if not isinstance(life_table, pd.DataFrame) or life_table.empty:
        # life table 없으면 MVP에서는 annuity 미적용(안전)
        return W0, AnnuityState(False, 0.0, 0.0, 0.0, -1)
    P = (1.0 + cfg.L) * cfg.alpha * W0
    a = compute_ax_real(age0_years, life_table, r_f_real_annual, S=S)
    y = P / a
    return W0 - P, AnnuityState(True, P, y, a, 0)
