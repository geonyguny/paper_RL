# project/runner/run.py
from __future__ import annotations
import contextlib, json
from typing import Any, Dict

from ..eval import evaluate
from ..config import SimConfig
from .config_build import make_cfg
from .actors import build_actor
from .annuity_wiring import setup_annuity_overlay
from .io_utils import ensure_dir, slim_args, do_autosave
from .logging_filters import silence_stdio

def run_once(args) -> Dict[str, Any]:
    quiet_ctx = silence_stdio(also_stderr=True) if getattr(args, "quiet", "on") == "on" else contextlib.nullcontext()
    with quiet_ctx:
        cfg = make_cfg(args)
        ensure_dir(args.outputs)

        ann_state = None
        if getattr(args, "ann_on", "off") == "on" and float(getattr(args, "ann_alpha", 0.0)) > 0.0:
            ann_state = setup_annuity_overlay(cfg, args)

        actor = build_actor(cfg, args)
        m = evaluate(cfg, actor, es_mode=args.es_mode)

    if ann_state is not None and isinstance(m, dict):
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

    if getattr(args, "autosave", "off") == "on":
        do_autosave(m, cfg, args, out)

    return out

def run_rl(args):
    cfg = make_cfg(args)
    ensure_dir(args.outputs)

    if getattr(args, "ann_on", "off") == "on" and float(getattr(args, "ann_alpha", 0.0)) > 0.0:
        setup_annuity_overlay(cfg, args)

    try:
        from ..trainer.rl_a2c import train_rl
    except Exception as e:
        raise SystemExit(f"RL trainer import failed: {e}")

    fields = train_rl(
        cfg,
        seed_list=args.seeds,
        outputs=args.outputs,
        n_paths_eval=args.rl_n_paths_eval,
        rl_epochs=args.rl_epochs,
        steps_per_epoch=args.rl_steps_per_epoch,
        lr=args.lr, gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef, value_coef=args.value_coef,
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
