# project/runner/helpers.py
from __future__ import annotations
import hashlib
from typing import Optional, Tuple, Any
import numpy as _np
import pandas as pd

def arrhash(a) -> str:
    if a is None: return "none"
    x = _np.asarray(a, dtype=_np.float32).tobytes()
    return hashlib.md5(x).hexdigest()

def auto_eta_grid(cfg, requested_n: int | None = None) -> None:
    cur = getattr(cfg, "hjb_eta_grid", ())
    if float(getattr(cfg, "lambda_term", 0.0) or 0.0) <= 0.0:
        if not cur or len(cur) <= 1:
            cfg.hjb_eta_grid = (0.0,)
        return
    n = int(requested_n or getattr(cfg, "hjb_eta_n", 41) or 41)
    F = float(getattr(cfg, "F_target", 0.0) or 0.0); F = F if F > 0.0 else 1.0
    try: cfg.hjb_eta_grid = tuple(_np.linspace(0.0, F, n))
    except Exception:
        step = F / max(n - 1, 1)
        cfg.hjb_eta_grid = tuple(0.0 + i * step for i in range(n))

def get_life_table_from_env(env) -> Optional[pd.DataFrame]:
    lt = getattr(env, "life_table", None)
    if isinstance(lt, pd.DataFrame) and not lt.empty: return lt
    lt2 = getattr(env, "mort_table_df", None)
    if isinstance(lt2, pd.DataFrame) and not lt2.empty: return lt2
    return None

def monthly_from_cfg(cfg) -> Tuple[float, float]:
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
