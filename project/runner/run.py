# project/runner/run.py
from __future__ import annotations
import contextlib, os
from typing import Any, Dict, Optional, Callable

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
            # 진단 로그 실패 시 조용히 패스
            pass


def _to_actor(policy_like: Any) -> Callable[[Dict[str, Any]], tuple[float, float]]:
    """
    다양한 형태의 policy/agent를 (q,w) 반환 actor(state)->(float,float) 로 래핑.
    """
    if policy_like is None:
        raise RuntimeError("policy_like is None")

    def _actor(state: Dict[str, Any]):
        out = None
        # 우선 순위: act → __call__ → predict → callable
        if hasattr(policy_like, "act"):
            out = policy_like.act(state)
        elif callable(policy_like):
            out = policy_like(state)
        elif hasattr(policy_like, "predict"):
            out = policy_like.predict(state)
        else:
            raise RuntimeError("No callable interface for actor: need .act/.predict/callable")

        # (q,w)로 정규화
        if isinstance(out, dict) and "q" in out and "w" in out:
            q, w = out["q"], out["w"]
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            q, w = out[0], out[1]
        else:
            raise RuntimeError("actor must return (q, w) or dict with keys 'q','w'")
        return float(q), float(w)

    return _actor


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
        m, extras = evaluate(cfg, actor, es_mode=args.es_mode)

    # --- 메트릭 병합: 연금 파생 파라미터는 항상 기록(미설정 시 0.0) ---
    if isinstance(m, dict):
        y_ann = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        a_fac = float(getattr(cfg, "ann_a_factor", 0.0) or 0.0)
        P_val = float(getattr(cfg, "ann_P", 0.0) or 0.0)
        m.update({
            "y_ann": y_ann if (y_ann != 0.0) else 0.0,
            "ann_a_factor": a_fac if (a_fac != 0.0) else 0.0,  # 편의상 유지
            "a_factor": a_fac if (a_fac != 0.0) else 0.0,      # CSV에 쓰이는 필드
            "P": P_val if (P_val != 0.0) else 0.0,
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
        extra=extras,  # 경로 등 부가정보 (CSV에는 저장 안 함)
    )

    if getattr(args, "autosave", "off") == "on":
        do_autosave(m, cfg, args, out)

    return out


def _maybe_load_actor_from_ckpt(
    ckpt_path: Optional[str],
    cfg_hint: Optional[Any],
) -> Optional[Callable[[Dict[str, Any]], tuple[float, float]]]:
    """
    trainer가 actor/policy를 직접 안 주는 경우 ckpt에서 로딩 시도.
    우선 project/trainer/policy_io.load_policy_as_actor(ckpt, cfg_hint)를 시도하고
    실패 시 과거/대안 로더들을 순차적으로 시도.
    """
    if not ckpt_path:
        return None

    # 최신 로더 우선 (cfg 힌트 전달)
    try:
        from ..trainer.policy_io import load_policy_as_actor  # type: ignore
        actor = load_policy_as_actor(ckpt_path, cfg_hint=cfg_hint)  # 새 시그니처
        if callable(actor):
            return actor
    except Exception:
        pass

    # 구버전/대체 로더 후보
    loaders = [
        ("..trainer.policy_io", "load_policy_as_actor"),
        ("..trainer.policy_io", "load_actor"),
        ("..trainer.rl_a2c", "load_policy_as_actor"),
        ("..trainer.rl_io", "load_actor"),
    ]
    import importlib
    for mod, attr in loaders:
        try:
            modobj = importlib.import_module(mod, package=__package__)
            loader = getattr(modobj, attr)
            try:
                policy_like = loader(ckpt_path, cfg_hint)  # cfg_hint 지원 로더
            except TypeError:
                policy_like = loader(ckpt_path)            # 구형 시그니처
            return _to_actor(policy_like)                  # actor/policy-like 모두 지원
        except Exception:
            continue
    return None


def run_rl(args):
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
        seed_list=args.seeds,
        outputs=args.outputs,
        n_paths_eval=args.rl_n_paths_eval,
        rl_epochs=args.rl_epochs,
        steps_per_epoch=args.rl_steps_per_epoch,
        lr=args.lr, gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef, value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )

    # trainer가 선택적으로 반환할 수 있는 확장 필드들(없으면 None)
    best_epoch   = fields.get("best_epoch")
    ckpt_path    = fields.get("ckpt_path")   # 예: outputs/<tag>/policy.pt
    train_time_s = fields.get("train_time_s")
    eval_time_s  = fields.get("eval_time_s")

    # --- actor 획득 시도: 반환 필드 → ckpt 로딩 → 실패 시 None ---
    policy_like = fields.get("actor") or fields.get("policy") or fields.get("pi") or fields.get("agent")
    actor: Optional[Callable[[Dict[str, Any]], tuple[float, float]]] = None
    try:
        if policy_like is not None:
            actor = _to_actor(policy_like)
    except Exception:
        actor = None
    if actor is None:
        actor = _maybe_load_actor_from_ckpt(ckpt_path, cfg_hint=cfg)

    # 총 평가 경로 수(= rl_n_paths_eval × seeds 수)
    n_paths_total = int(getattr(args, "rl_n_paths_eval", 0)) * len(getattr(args, "seeds", []))

    # --- 옵션: actor를 그대로 반환해 CLI가 재평가 하도록 위임 (권장 경로) ---
    if actor is not None and getattr(args, "return_actor", "on") == "on":
        # CLI는 (cfg, actor) 튜플을 받으면 evaluate를 호출해 W_T 기반으로 ES95를 정확히 산출
        return (cfg, actor)

    # --- 폴백 경로: 여기서 바로 evaluate 수행 ---
    metrics_dict: Dict[str, Any]
    extras_dict: Dict[str, Any] = {}

    if actor is not None:
        try:
            metrics_dict, extras_dict = evaluate(cfg, actor, es_mode=getattr(args, "es_mode", "wealth"))
            # 러닝 메타 부가
            metrics_dict.update({
                "best_epoch": best_epoch,
                "train_time_s": train_time_s,
                "eval_time_s": eval_time_s,
            })
            # 소스 표기(wealth/loss는 evaluate가 이미 es95_source를 넣어줌)
            metrics_dict.setdefault("es95_source", "computed_in_evaluate")
        except Exception as e:
            # 평가 실패 시 trainer 메트릭으로 폴백
            metrics_dict = {
                "EW": fields.get("EW"),
                "ES95": fields.get("ES95"),
                "Ruin": fields.get("Ruin"),
                "mean_WT": fields.get("mean_WT"),
                "best_epoch": best_epoch,
                "train_time_s": train_time_s,
                "eval_time_s": eval_time_s,
                "es95_note": f"evaluate failed in run_rl: {type(e).__name__}",
            }
    else:
        # actor를 못 얻었다면 trainer가 준 요약 메트릭만 사용
        metrics_dict = {
            "EW": fields.get("EW"),
            "ES95": fields.get("ES95"),
            "Ruin": fields.get("Ruin"),
            "mean_WT": fields.get("mean_WT"),
            "best_epoch": best_epoch,
            "train_time_s": train_time_s,
            "eval_time_s": eval_time_s,
            "es95_note": "no actor available (trainer didn't return policy and ckpt loader failed)",
        }

    out = dict(
        asset=cfg.asset,
        method="rl",
        baseline="",
        metrics=metrics_dict,
        w_max=cfg.w_max,
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term,
        alpha=cfg.alpha,
        F_target=cfg.F_target,
        es_mode=getattr(args, "es_mode", "wealth"),
        n_paths=n_paths_total,
        args=slim_args(args) | {
            # 결과 재현을 위해 args에도 몇 가지 핵심 하이퍼 명시
            "rl_q_cap": getattr(args, "rl_q_cap", None),
            "teacher_eps0": getattr(args, "teacher_eps0", None),
            "teacher_decay": getattr(args, "teacher_decay", None),
            "survive_bonus": getattr(args, "survive_bonus", None),
            "u_scale": getattr(args, "u_scale", None),
            "lw_scale": getattr(args, "lw_scale", None),
            "tag": getattr(args, "tag", None),
        },
        ckpt_path=ckpt_path,
        extra=extras_dict,   # 여기 안에 eval_WT 등 경로 정보가 포함될 수 있음
    )

    # RL도 autosave 동일 적용
    if getattr(args, "autosave", "off") == "on":
        do_autosave(out.get("metrics") or {}, cfg, args, out)

    return out
