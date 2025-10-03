# project/runner/run.py
from __future__ import annotations
import contextlib, os
from typing import Any, Dict

from ..eval import evaluate
from ..config import SimConfig
from .config_build import make_cfg
from .actors import build_actor
from .annuity_wiring import setup_annuity_overlay
from .io_utils import ensure_dir, slim_args, do_autosave
from .logging_filters import silence_stdio

# market CSV loader
from ..data.loader import load_market_csv


def _wire_market_data(cfg: SimConfig, args) -> None:
    """
    market_mode=bootstrap 인 경우 CSV 로더를 통해
    원시 시계열(ret_asset, rf_real/nom, dates, cpi)을 cfg에 주입.
    Env/actor/eval은 cfg에서 이를 사용.
    """
    # eval/plot에서 참조할 플래그도 cfg에 고정 주입
    setattr(cfg, "bands", getattr(args, "bands", "on"))
    setattr(cfg, "data_window", getattr(args, "data_window", None))
    setattr(cfg, "use_real_rf", getattr(args, "use_real_rf", "on"))

    if getattr(cfg, "market_mode", "iid") != "bootstrap":
        return

    market_csv = getattr(args, "market_csv", None)
    if not market_csv:
        raise SystemExit(
            "market_mode=bootstrap 이지만 --market_csv 가 지정되지 않았습니다. "
            "또는 --data_profile(dev|full)을 사용하세요."
        )

    abs_csv = os.path.abspath(market_csv)
    if not os.path.exists(abs_csv):
        cwd = os.getcwd()
        raise SystemExit(
            "market_csv 파일을 찾을 수 없습니다.\n"
            f"  asked: {market_csv}\n"
            f"  abs:   {abs_csv}\n"
            f"  cwd:   {cwd}\n"
            "힌트: 실제 파일 경로를 지정하거나 --data_profile dev|full 을 사용하세요."
        )

    blob = load_market_csv(
        path=abs_csv,
        asset=getattr(cfg, "asset", "KR"),
        use_real_rf=getattr(args, "use_real_rf", "on"),
        data_window=getattr(args, "data_window", None),
        cache=True,
    )

    # Env에서 사용할 수 있도록 cfg에 부착
    setattr(cfg, "data_dates", blob.get("dates"))
    setattr(cfg, "data_cpi", blob.get("cpi"))
    setattr(cfg, "data_ret_series", blob.get("ret_asset"))

    # 실질/명목 RF 선택 주입
    if getattr(args, "use_real_rf", "on") == "on":
        setattr(cfg, "data_rf_series", blob.get("rf_real"))
    else:
        setattr(cfg, "data_rf_series", blob.get("rf_nom"))

    # 보조 시계열(선택적으로 활용)
    setattr(cfg, "data_ret_kr_eq", blob.get("ret_kr_eq"))
    setattr(cfg, "data_ret_us_eq_krw", blob.get("ret_us_eq_krw"))
    setattr(cfg, "data_ret_gold_krw", blob.get("ret_gold_krw"))

    # ---- 진단 로그 (quiet=off 일 때만) ----
    if str(getattr(args, "quiet", "on")).lower() != "on":
        import numpy as _np
        _ret = blob.get("ret_asset")
        _rf = blob.get("rf_real") if getattr(args, "use_real_rf", "on") == "on" else blob.get("rf_nom")
        try:
            ret_mean = float(_np.nanmean(_ret)) if _ret is not None else float("nan")
            rf_mean = float(_np.nanmean(_rf)) if _rf is not None else float("nan")
            print(
                f"[data] len={len(_ret) if _ret is not None else 0}, "
                f"ret_mean={ret_mean:.4f}, rf_mean={rf_mean:.4f}, "
                f"asset={getattr(cfg, 'asset', '?')}, window={getattr(cfg, 'data_window', None)}"
            )
        except Exception:
            # 진단로그는 실패해도 시뮬레이션에는 영향 없게 조용히 패스
            pass


def run_once(args) -> Dict[str, Any]:
    quiet_ctx = silence_stdio(also_stderr=True) if getattr(args, "quiet", "on") == "on" else contextlib.nullcontext()
    with quiet_ctx:
        cfg: SimConfig = make_cfg(args)
        ensure_dir(args.outputs)

        # tag → cfg 주입 (autosave에서 사용)
        if getattr(args, "tag", None) is not None:
            setattr(cfg, "tag", args.tag)

        # 실데이터 로더와 연결 (bootstrap일 때만)
        _wire_market_data(cfg, args)

        # 연금 오버레이(필요 시 설정)
        ann_enabled = (
            str(getattr(args, "ann_on", "off")).lower() == "on"
            and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0
        )
        if ann_enabled:
            setup_annuity_overlay(cfg, args)

        # 정책 생성 → 평가
        actor = build_actor(cfg, args)
        m = evaluate(cfg, actor, es_mode=args.es_mode)

    # --- 메트릭 병합: 연금 파생 파라미터는 항상 기록(미설정 시 0.0) ---
    if isinstance(m, dict):
        y_ann = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        a_fac = float(getattr(cfg, "ann_a_factor", 0.0) or 0.0)
        P_val = float(getattr(cfg, "ann_P", 0.0) or 0.0)
        m.update({
            "y_ann": y_ann if ann_enabled or y_ann != 0.0 else 0.0,
            "ann_a_factor": a_fac if ann_enabled or a_fac != 0.0 else 0.0,  # 편의상 유지
            "a_factor": a_fac if ann_enabled or a_fac != 0.0 else 0.0,      # CSV에 쓰이는 필드
            "P": P_val if ann_enabled or P_val != 0.0 else 0.0,
        })

    # 총 평가 경로 수
    n_paths_total = getattr(cfg, "n_paths_eval", getattr(cfg, "n_paths", 0)) * len(getattr(cfg, "seeds", []))

    out = dict(
        asset=cfg.asset,
        method=args.method,
        baseline=args.baseline,
        metrics=m,
        w_max=cfg.w_max,
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term,
        alpha=cfg.alpha,
        F_target=cfg.F_target,
        es_mode=args.es_mode,
        n_paths=n_paths_total,
        args=slim_args(args),
    )

    if getattr(args, "autosave", "off") == "on":
        do_autosave(m, cfg, args, out)

    return out


def run_rl(args) -> Dict[str, Any]:
    """
    RL 학습 러너:
    - cfg 구성 및 실데이터 연결
    - (선택) 연금 오버레이 주입
    - A2C 학습 후 요약 메트릭 반환
    - es_mode 출력은 인자(args.es_mode)를 그대로 반영
    """
    # RL도 동일하게 cfg 주입 → trainer 내부 Env 생성 시 실데이터 사용
    cfg: SimConfig = make_cfg(args)
    ensure_dir(args.outputs)

    # tag → cfg 주입 (RL도 동일)
    if getattr(args, "tag", None) is not None:
        setattr(cfg, "tag", args.tag)

    _wire_market_data(cfg, args)

    ann_enabled = (
        str(getattr(args, "ann_on", "off")).lower() == "on"
        and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0
    )
    if ann_enabled:
        setup_annuity_overlay(cfg, args)

    try:
        from ..trainer.rl_a2c import train_rl
    except Exception as e:
        raise SystemExit(f"RL trainer import failed: {e}")

    fields: Dict[str, Any] = train_rl(
        cfg,
        seed_list=getattr(args, "seeds", []),
        outputs=args.outputs,
        n_paths_eval=getattr(args, "rl_n_paths_eval", 0),
        rl_epochs=getattr(args, "rl_epochs", 0),
        steps_per_epoch=getattr(args, "rl_steps_per_epoch", 0),
        lr=getattr(args, "lr", 3e-4),
        gae_lambda=getattr(args, "gae_lambda", 0.95),
        entropy_coef=getattr(args, "entropy_coef", 0.0),
        value_coef=getattr(args, "value_coef", 0.5),
        max_grad_norm=getattr(args, "max_grad_norm", 0.5),
    )

    metrics = {
        "EW": fields.get("EW"),
        "ES95": fields.get("ES95"),
        "Ruin": fields.get("Ruin"),
        "mean_WT": fields.get("mean_WT"),
    }

    eval_mode = getattr(args, "es_mode", "wealth")
    n_paths = int(getattr(args, "rl_n_paths_eval", 0)) * len(getattr(args, "seeds", []))

    out = dict(
        asset=cfg.asset,
        method="rl",
        baseline="",
        metrics={
            "EW": fields.get("EW"),
            "ES95": fields.get("ES95"),
            "Ruin": fields.get("Ruin"),
            "mean_WT": fields.get("mean_WT"),
        },
        w_max=cfg.w_max,
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term,
        alpha=cfg.alpha,
        F_target=cfg.F_target,
        es_mode=getattr(args, "es_mode", "wealth"),
        n_paths=args.rl_n_paths_eval * len(args.seeds),
        args=slim_args(args),
    )

    # RL도 autosave 동일 적용
    if str(getattr(args, "autosave", "off")).lower() == "on":
        do_autosave(metrics, cfg, args, out)

    return out
