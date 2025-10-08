# project/runner/run.py
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, Optional, Callable, Tuple

from ..eval import evaluate
from ..config import SimConfig
from .config_build import make_cfg
from .actors import build_actor
from .annuity_wiring import setup_annuity_overlay
from .io_utils import ensure_dir, slim_args, do_autosave
from .logging_filters import silence_stdio

# market CSV loader
from ..data.loader import load_market_csv


# --------------------------
# Utilities
# --------------------------
def _fmt_hms(sec: float) -> str:
    try:
        total = int(round(float(sec)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"


# --------------------------
# Helpers: parsing mix / hedge
# --------------------------
def _parse_alpha_mix(args) -> Tuple[float, float, float]:
    """
    α 파싱: --alpha_mix "a,b,c" or 개별 --alpha_kr/us/au.
    합이 1이 아니면 자동 정규화. 미지정 시 (1/3,1/3,1/3).
    """
    def _as_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    if hasattr(args, "alpha_mix") and args.alpha_mix:
        raw = str(args.alpha_mix).replace(" ", "")
        parts = raw.split(",")
        if len(parts) == 3:
            kr = _as_float(parts[0], 1 / 3)
            us = _as_float(parts[1], 1 / 3)
            au = _as_float(parts[2], 1 / 3)
        else:
            kr = us = au = 1 / 3
    else:
        kr = _as_float(getattr(args, "alpha_kr", None), None)
        us = _as_float(getattr(args, "alpha_us", None), None)
        au = _as_float(getattr(args, "alpha_au", None), None)
        if kr is None or us is None or au is None:
            kr = us = au = 1 / 3

    s = kr + us + au
    if s <= 0:
        kr = us = au = 1 / 3
        s = 1.0
    return (kr / s, us / s, au / s)


def _get_fx_hedge_params(args) -> Tuple[float, float]:
    """
    h_FX ∈ [0,1], 연 환헤지비용(기본 0.002=0.2%)
    CLI에 없으면 속성이 없을 수 있으니 조용히 기본값 사용.
    """
    h = getattr(args, "h_FX", getattr(args, "h_fx", None))
    try:
        h = float(h)
    except Exception:
        h = 0.0
    h = max(0.0, min(1.0, h))

    fx_cost_annual = getattr(args, "fx_hedge_cost", None)
    try:
        fx_cost_annual = float(fx_cost_annual)
    except Exception:
        fx_cost_annual = 0.002  # 0.2%p/년 (설계 문서 기본)
    return h, fx_cost_annual


# --------------------------
# Data wiring (with mix & FX hedge)
# --------------------------
def _wire_market_data(cfg: SimConfig, args) -> None:
    """
    market_mode=bootstrap 인 경우 CSV 로더를 통해
    원시 시계열을 cfg에 주입. (멀티자산 혼합 + 환헤지 반영)
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

    # ---- 개별 시계열 추출 ----
    ret_kr = blob.get("ret_kr_eq")            # KR 주식 수익률(월)
    ret_us_l = blob.get("ret_us_eq_krw")      # US 주식, KRW 기준(= FX 노출 포함)
    ret_au = blob.get("ret_gold_krw")         # 금(KRW)
    rf_real = blob.get("rf_real")
    rf_nom = blob.get("rf_nom")
    dates = blob.get("dates")
    cpi = blob.get("cpi")

    # (가능 시) FX 월수익률: 로더가 제공하면 사용
    ret_fx = blob.get("ret_fx") or blob.get("ret_fx_usdkrw") or None

    # ---- 환헤지 처리 (US) ----
    import numpy as np
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    h_fx, fx_cost_ann = _get_fx_hedge_params(args)
    fx_cost_m = float(fx_cost_ann) / float(steps_per_year)

    if ret_us_l is None:
        # 로더에 종합 risky 수익률만 있는 레거시 케이스
        mixed = blob.get("ret_asset")
    else:
        # US 현지자산 수익률을 근사: KRW기준 수익률에서 FX를 제거
        if ret_fx is not None:
            ret_us_hedged = (
                np.asarray(ret_us_l, dtype=float)
                - h_fx * np.asarray(ret_fx, dtype=float)
                - h_fx * fx_cost_m
            )
        else:
            # FX 시계열이 없으면 비용만 차감(보수적 보정)
            ret_us_hedged = np.asarray(ret_us_l, dtype=float) - h_fx * fx_cost_m

        # ---- 믹스 가중합 ----
        a_kr, a_us, a_au = _parse_alpha_mix(args)
        # None 안전 처리
        kr = np.asarray(ret_kr, dtype=float) if ret_kr is not None else 0.0
        us = np.asarray(ret_us_hedged, dtype=float)
        au = np.asarray(ret_au, dtype=float) if ret_au is not None else 0.0
        # 브로드캐스트 대비, 모두 같은 길이 가정. 길이 불일치 시 가능한 최소 길이로 자름.
        lens = [x.shape[0] for x in [kr, us, au] if isinstance(x, np.ndarray)]
        T = min(lens) if len(lens) >= 1 else 0
        if isinstance(kr, np.ndarray):
            kr = kr[:T]
        if isinstance(us, np.ndarray):
            us = us[:T]
        if isinstance(au, np.ndarray):
            au = au[:T]
        mixed = a_kr * kr + a_us * us + a_au * au

        # cfg에 믹스/헤지 파라미터 기록(평가/로그용)
        setattr(cfg, "alpha_mix", (a_kr, a_us, a_au))
        setattr(cfg, "h_FX", h_fx)
        setattr(cfg, "fx_hedge_cost_annual", fx_cost_ann)

    # Env에서 사용할 수 있도록 cfg에 부착
    setattr(cfg, "data_dates", dates)
    setattr(cfg, "data_cpi", cpi)
    setattr(cfg, "data_ret_series", mixed)

    # 실질/명목 RF 선택 주입
    if getattr(args, "use_real_rf", "on") == "on":
        setattr(cfg, "data_rf_series", rf_real)
    else:
        setattr(cfg, "data_rf_series", rf_nom)

    # 보조 시계열(선택적으로 활용)
    setattr(cfg, "data_ret_kr_eq", ret_kr)
    setattr(cfg, "data_ret_us_eq_krw", ret_us_l)
    setattr(cfg, "data_ret_gold_krw", ret_au)

    # ---- 진단 로그 (quiet=off 일 때만) ----
    if str(getattr(args, "quiet", "on")).lower() != "on":
        try:
            import numpy as _np
            ret_mean = float(_np.nanmean(mixed)) if mixed is not None else float("nan")
            rf_series = rf_real if getattr(args, "use_real_rf", "on") == "on" else rf_nom
            rf_mean = float(_np.nanmean(rf_series)) if rf_series is not None else float("nan")
            a = getattr(cfg, "alpha_mix", None)
            a_str = f"alpha={a}" if a is not None else "alpha=legacy"
            print(
                f"[data] len={len(mixed) if mixed is not None else 0}, "
                f"ret_mean={ret_mean:.4f}, rf_mean={rf_mean:.4f}, "
                f"h_FX={getattr(cfg,'h_FX',0.0):.2f}, {a_str}, "
                f"asset={getattr(cfg, 'asset', '?')}, window={getattr(cfg, 'data_window', None)}"
            )
        except Exception:
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


# --- evaluate 결과 정규화 유틸 ---
def _normalize_evaluate_output(ret, es_mode: str):
    """evaluate 반환을 (metrics: dict, extras: dict)로 정규화."""
    metrics, extras = {}, {}

    if isinstance(ret, dict):
        metrics = ret
    elif isinstance(ret, tuple):
        if len(ret) >= 1 and isinstance(ret[0], dict):
            metrics = ret[0]
        if len(ret) >= 2 and isinstance(ret[1], dict):
            extras = ret[1]
        # 그 외 (길이>2, dict가 아닌 두 번째 요소 등)도 extras에 참고용으로 보관
        if len(ret) > 2:
            extras["_rest"] = ret[2:]
    else:
        metrics = {"note": "unexpected evaluate return type", "type": str(type(ret))}

    if "es_mode" not in metrics:
        metrics["es_mode"] = str(es_mode).lower()
    return metrics, extras


def _call_evaluate(cfg, actor, es_mode: str):
    """return_paths 지원/비지원 모두 안전 호출."""
    try:
        ret = evaluate(cfg, actor, es_mode=str(es_mode).lower(), return_paths=True)
    except TypeError:
        ret = evaluate(cfg, actor, es_mode=str(es_mode).lower())
    return _normalize_evaluate_output(ret, es_mode)


def run_once(args) -> Dict[str, Any]:
    t_all_0 = time.perf_counter()

    quiet_ctx = (
        silence_stdio(also_stderr=True)
        if getattr(args, "quiet", "on") == "on"
        else contextlib.nullcontext()
    )
    with quiet_ctx:
        t0 = time.perf_counter()
        cfg: SimConfig = make_cfg(args)
        time_make_cfg = time.perf_counter() - t0

        ensure_dir(args.outputs)

        # tag → cfg 주입 (autosave에서 사용)
        if getattr(args, "tag", None) is not None:
            setattr(cfg, "tag", args.tag)

        t1 = time.perf_counter()
        # 실데이터 로더와 연결 (bootstrap일 때만)
        _wire_market_data(cfg, args)
        time_wire_data = time.perf_counter() - t1

        # 연금 오버레이(필요 시 설정)
        t2 = time.perf_counter()
        ann_enabled = (
            str(getattr(args, "ann_on", "off")).lower() == "on"
            and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0
        )
        if ann_enabled:
            setup_annuity_overlay(cfg, args)
        time_annuity = time.perf_counter() - t2

        # 정책 생성 → 평가
        t3 = time.perf_counter()
        actor = build_actor(cfg, args)
        time_build_actor = time.perf_counter() - t3

        t4 = time.perf_counter()
        m, extras = _call_evaluate(cfg, actor, es_mode=args.es_mode)
        time_eval = time.perf_counter() - t4

    # --- 메트릭 병합: 연금 파생 파라미터는 항상 기록(미설정 시 0.0) ---
    if isinstance(m, dict):
        y_ann = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        a_fac = float(getattr(cfg, "ann_a_factor", 0.0) or 0.0)
        P_val = float(getattr(cfg, "ann_P", 0.0) or 0.0)
        m.update(
            {
                "y_ann": y_ann if (y_ann != 0.0) else 0.0,
                "ann_a_factor": a_fac if (a_fac != 0.0) else 0.0,  # 편의상 유지
                "a_factor": a_fac if (a_fac != 0.0) else 0.0,      # CSV에 쓰이는 필드
                "P": P_val if (P_val != 0.0) else 0.0,
            }
        )

    # 총 평가 경로 수
    n_paths_total = (
        getattr(cfg, "n_paths_eval", getattr(cfg, "n_paths", 0))
        * len(getattr(cfg, "seeds", []))
    )

    # 타이밍 집계
    time_total = time.perf_counter() - t_all_0
    timing = {
        "make_cfg_s": round(time_make_cfg, 6),
        "wire_data_s": round(time_wire_data, 6),
        "annuity_setup_s": round(time_annuity, 6),
        "build_actor_s": round(time_build_actor, 6),
        "evaluate_s": round(time_eval, 6),
        "total_s": round(time_total, 6),
        "total_hms": _fmt_hms(time_total),
    }

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
        timing=timing,
        time_total_s=timing["total_s"],
        time_total_hms=timing["total_hms"],
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
        t0 = time.perf_counter()
        actor = load_policy_as_actor(ckpt_path, cfg_hint=cfg_hint)  # 새 시그니처
        _ = time.perf_counter() - t0  # 로딩 시간은 run_rl의 timing에서 별도 측정
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
                policy_like = loader(ckpt_path)  # 구형 시그니처
            return _to_actor(policy_like)  # actor/policy-like 모두 지원
        except Exception:
            continue
    return None


def run_rl(args):
    t_all_0 = time.perf_counter()

    # RL도 동일하게 cfg 주입 → trainer 내부 Env 생성 시 실데이터 사용
    t0 = time.perf_counter()
    cfg: SimConfig = make_cfg(args)
    time_make_cfg = time.perf_counter() - t0

    ensure_dir(args.outputs)

    # tag → cfg 주입 (RL도 동일)
    if getattr(args, "tag", None) is not None:
        setattr(cfg, "tag", args.tag)

    t1 = time.perf_counter()
    _wire_market_data(cfg, args)
    time_wire_data = time.perf_counter() - t1

    ann_enabled = (
        str(getattr(args, "ann_on", "off")).lower() == "on"
        and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0
    )
    t2 = time.perf_counter()
    if ann_enabled:
        setup_annuity_overlay(cfg, args)
    time_annuity = time.perf_counter() - t2

    try:
        from ..trainer.rl_a2c import train_rl
    except Exception as e:
        raise SystemExit(f"RL trainer import failed: {e}")

    # 학습 호출 자체의 벽시계 시간 래핑(트레이너가 별도로 제공하는 train_time_s와는 별개)
    t3 = time.perf_counter()
    fields: Dict[str, Any] = train_rl(
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
    time_train_call = time.perf_counter() - t3

    # trainer가 선택적으로 반환할 수 있는 확장 필드들(없으면 None)
    best_epoch = fields.get("best_epoch")
    ckpt_path = fields.get("ckpt_path")  # 예: outputs/<tag>/policy.pt
    train_time_s = fields.get("train_time_s")
    eval_time_s = fields.get("eval_time_s")

    # --- actor 획득 시도: 반환 필드 → ckpt 로딩 → 실패 시 None ---
    policy_like = fields.get("actor") or fields.get("policy") or fields.get("pi") or fields.get("agent")
    t4 = time.perf_counter()
    actor: Optional[Callable[[Dict[str, Any]], tuple[float, float]]] = None
    try:
        if policy_like is not None:
            actor = _to_actor(policy_like)
    except Exception:
        actor = None
    if actor is None:
        actor = _maybe_load_actor_from_ckpt(ckpt_path, cfg_hint=cfg)
    time_actor_load = time.perf_counter() - t4

    # 총 평가 경로 수(= rl_n_paths_eval × seeds 수)
    n_paths_total = int(getattr(args, "rl_n_paths_eval", 0)) * len(getattr(args, "seeds", []))

    # --- 옵션: actor를 그대로 반환해 CLI가 재평가 하도록 위임 (권장 경로) ---
    if actor is not None and str(getattr(args, "return_actor", "off")).lower() == "on":
        # CLI 경로에서 총 시간은 CLI가 별도 계측
        return (cfg, actor)

    # --- 폴백 경로: 여기서 바로 evaluate 수행 ---
    metrics_dict: Dict[str, Any]
    extras_dict: Dict[str, Any] = {}

    t5 = time.perf_counter()
    if actor is not None:
        try:
            metrics_dict, extras_dict = _call_evaluate(cfg, actor, es_mode=getattr(args, "es_mode", "wealth"))
            # 러닝 메타 부가
            metrics_dict.update(
                {
                    "best_epoch": best_epoch,
                    "train_time_s": train_time_s,
                    "eval_time_s": eval_time_s,
                }
            )
            # 소스 표기(wealth/loss는 evaluate가 이미 es95_source를 넣어줄 수 있음)
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
    time_eval = time.perf_counter() - t5

    # 타이밍 집계
    time_total = time.perf_counter() - t_all_0
    timing = {
        "make_cfg_s": round(time_make_cfg, 6),
        "wire_data_s": round(time_wire_data, 6),
        "annuity_setup_s": round(time_annuity, 6),
        "train_call_s": round(time_train_call, 6),  # train_rl 호출 벽시계 시간
        "actor_load_s": round(time_actor_load, 6),
        "evaluate_s": round(time_eval, 6),
        "total_s": round(time_total, 6),
        "total_hms": _fmt_hms(time_total),
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
        args=slim_args(args)
        | {
            # 결과 재현을 위해 args에도 몇 가지 핵심 하이퍼 명시
            "rl_q_cap": getattr(args, "rl_q_cap", None),
            "teacher_eps0": getattr(args, "teacher_eps0", None),
            "teacher_decay": getattr(args, "teacher_decay", None),
            "survive_bonus": getattr(args, "survive_bonus", None),
            "u_scale": getattr(args, "u_scale", None),
            "lw_scale": getattr(args, "lw_scale", None),
            "tag": getattr(args, "tag", None),
            "alpha_mix": getattr(cfg, "alpha_mix", None),
            "h_FX": getattr(cfg, "h_FX", None),
        },
        ckpt_path=ckpt_path,
        extra=extras_dict,   # 여기 안에 eval_WT 등 경로 정보가 포함될 수 있음
        timing=timing,
        time_total_s=timing["total_s"],
        time_total_hms=timing["total_hms"],
    )

    # RL도 autosave 동일 적용
    if getattr(args, "autosave", "off") == "on":
        do_autosave(out.get("metrics") or {}, cfg, args, out)

    return out
