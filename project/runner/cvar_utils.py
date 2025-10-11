from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

try:
    from ..utils.metrics_utils import terminal_losses, cvar_alpha  # type: ignore
except Exception:
    terminal_losses = None  # type: ignore
    cvar_alpha = None       # type: ignore

def cvar_fallback(losses: Iterable[float], alpha: float) -> float:
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
    tail_sum = float(L[j + 1 :].sum())
    ES = ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))
    return float(ES)

def maybe_extract_WT(candidate: Any) -> Optional[Iterable[float]]:
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
            wt = maybe_extract_WT(candidate[1])
            if wt is not None:
                return wt
        if candidate and all(isinstance(x, (int, float)) for x in candidate):
            return candidate  # type: ignore
    if isinstance(candidate, dict):
        for k in [
            "eval_WT", "W_T", "WT", "terminal_wealth", "terminal_wealths",
            "paths_WT", "wt_paths", "wealth_terminal", "wealth_T",
        ]:
            if k in candidate and candidate[k] is not None:
                return candidate[k]  # type: ignore
        for k in ["metrics", "extra", "extras", "eval", "payload", "data", "result"]:
            if k in candidate and isinstance(candidate[k], (dict, list, tuple)):
                wt = maybe_extract_WT(candidate[k])
                if wt is not None:
                    return wt
    for attr in ["eval_WT", "W_T", "WT", "terminal_wealth", "paths_WT", "wealth_T", "terminal_wealths"]:
        try:
            wt = getattr(candidate, attr)
            if wt is not None:
                return wt  # type: ignore
        except Exception:
            pass
    return None

def resolve_F_for_cvar(args, out: Dict[str, Any]) -> float:
    try:
        cc = out.get("cvar_calibration", {}) if isinstance(out, dict) else {}
        sf = cc.get("selected_F_target") or cc.get("F_selected")
        if isinstance(sf, (int, float)):
            return float(sf)
    except Exception:
        pass
    try:
        ft = out.get("F_target") if isinstance(out, dict) else None
        if isinstance(ft, (int, float)):
            return float(ft)
    except Exception:
        pass
    try:
        return float(getattr(args, "F_target", 0.0) or 0.0)
    except Exception:
        return 0.0

def fixup_metrics_with_cvar(args, out: Dict[str, Any]) -> Dict[str, Any]:
    metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    F_target = resolve_F_for_cvar(args, out if isinstance(out, dict) else {})
    alpha_v = float(getattr(args, "alpha", 0.95) or 0.95)

    if es_mode != "loss" or F_target <= 0.0:
        metrics["es_mode"] = es_mode
        return out

    WT = None
    for cand in (out, metrics):
        WT = maybe_extract_WT(cand)
        if WT is not None:
            break

    if WT is None:
        try:
            EW = float(metrics.get("EW", 0.0) or metrics.get("mean_WT", 0.0) or 0.0)
            ES_old = float(metrics.get("ES95", 0.0) or 0.0)
            if abs((EW + ES_old) - F_target) < 1e-9 and ES_old > 0:
                metrics["es95_note"] = "ES95 looks like (F_target - EW). No W_T to recompute; please expose path-level W_T from evaluate."
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        return out

    try:
        import numpy as np
        WT_arr = np.asarray(list(WT), dtype=float)
        if terminal_losses is not None and cvar_alpha is not None:
            L = terminal_losses(WT_arr, F_target)
            ES = cvar_alpha(L, alpha=alpha_v)
        else:
            L = np.maximum(F_target - WT_arr, 0.0)
            ES = cvar_fallback(L, alpha=alpha_v)

        EW = float(metrics.get("EW", 0.0))
        if EW == 0.0:
            EW = float(metrics.get("mean_WT", 0.0) or float(np.mean(WT_arr)))
        metrics["EW"] = EW
        metrics["mean_WT"] = EW
        metrics["ES95"] = float(ES)
        try:
            metrics["Ruin"] = float((WT_arr <= 0.0).mean())
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        metrics["es95_source"] = "path_level_cvar"
    except Exception as e:
        metrics["es_mode"] = es_mode
        metrics["es95_note"] = f"failed to recompute ES95: {type(e).__name__}"
    return out
