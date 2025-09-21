# project/runner/calibrate.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
from argparse import Namespace

from .run import run_once

def copy_args(args, **overrides):
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
