# project/runner/run.py
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, Optional, Callable, Tuple, List, Union

import numpy as _np

from ..eval import evaluate
from ..config import SimConfig
from .config_build import make_cfg
from .actors import build_actor
from .annuity_wiring import setup_annuity_overlay
from .io_utils import ensure_dir, slim_args, do_autosave
from .logging_filters import silence_stdio

# market CSV loader
from ..data.loader import load_market_csv

# ⚠ HOTFIX 0-2: 로컬 롤아웃에서도 평가와 동일한 Env 사용(상태 스케일/필드 통일)
# from ..env import RetirementEnv
from ..env.retirement_env import RetirementEnv  # type: ignore


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
        fx_cost_annual = 0.002  # 0.2%p/년
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
    ret_kr = blob.get("ret_kr_eq")
    ret_us_l = blob.get("ret_us_eq_krw")
    ret_au = blob.get("ret_gold_krw")
    rf_real = blob.get("rf_real")
    rf_nom = blob.get("rf_nom")
    dates = blob.get("dates")
    cpi = blob.get("cpi")

    # (가능 시) FX 월수익률
    ret_fx = blob.get("ret_fx") or blob.get("ret_fx_usdkrw") or None

    # ---- 환헤지 처리 (US) ----
    import numpy as np
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    h_fx, fx_cost_ann = _get_fx_hedge_params(args)
    fx_cost_m = float(fx_cost_ann) / float(steps_per_year)

    if ret_us_l is None:
        mixed = blob.get("ret_asset")
    else:
        if ret_fx is not None:
            ret_us_hedged = (
                np.asarray(ret_us_l, dtype=float)
                - h_fx * np.asarray(ret_fx, dtype=float)
                - h_fx * fx_cost_m
            )
        else:
            ret_us_hedged = np.asarray(ret_us_l, dtype=float) - h_fx * fx_cost_m

        a_kr, a_us, a_au = _parse_alpha_mix(args)
        kr = np.asarray(ret_kr, dtype=float) if ret_kr is not None else 0.0
        us = np.asarray(ret_us_hedged, dtype=float)
        au = np.asarray(ret_au, dtype=float) if ret_au is not None else 0.0

        lens = [x.shape[0] for x in [kr, us, au] if isinstance(x, np.ndarray)]
        T = min(lens) if len(lens) >= 1 else 0
        if isinstance(kr, np.ndarray):
            kr = kr[:T]
        if isinstance(us, np.ndarray):
            us = us[:T]
        if isinstance(au, np.ndarray):
            au = au[:T]
        mixed = a_kr * kr + a_us * us + a_au * au

        setattr(cfg, "alpha_mix", (a_kr, a_us, a_au))
        setattr(cfg, "h_FX", h_fx)
        setattr(cfg, "fx_hedge_cost_annual", fx_cost_ann)

    setattr(cfg, "data_dates", dates)
    setattr(cfg, "data_cpi", cpi)
    setattr(cfg, "data_ret_series", mixed)

    if getattr(args, "use_real_rf", "on") == "on":
        setattr(cfg, "data_rf_series", rf_real)
    else:
        setattr(cfg, "data_rf_series", rf_nom)

    setattr(cfg, "data_ret_kr_eq", ret_kr)
    setattr(cfg, "data_ret_us_eq_krw", ret_us_l)
    setattr(cfg, "data_ret_gold_krw", ret_au)

    if str(getattr(args, "quiet", "on")).lower() != "on":
        try:
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


def _to_actor(policy_like: Any) -> Callable[[Any], tuple[float, float]]:
    """
    다양한 형태의 policy/agent를 (q,w) 반환 actor(state)->(float,float) 로 래핑.
    state는 dict/ndarray 모두 허용 (정책이 원하는 형태를 그대로 전달).
    """
    if policy_like is None:
        raise RuntimeError("policy_like is None")

    def _actor(state: Any):
        out = None
        if hasattr(policy_like, "act"):
            out = policy_like.act(state)
        elif callable(policy_like):
            out = policy_like(state)
        elif hasattr(policy_like, "predict"):
            out = policy_like.predict(state)
        else:
            raise RuntimeError("No callable interface for actor: need .act/.predict/callable")

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


# ==========================
# 안전한 경로 재평가(로컬 롤아웃)
# ==========================
def _as_ndarray_state(raw_obs: Union[Dict[str, Any], _np.ndarray, None], env: Any) -> _np.ndarray:
    """
    ⚠ HOTFIX 0-1: 정책에 전달할 상태를 항상 ndarray([t_norm, W]) 로 강제.
    - dict: {"t": step_index, "W": wealth} 가정 → [t_norm, W]
    - ndarray: 그대로 float 배열로 강제 캐스팅
    - 기타: [0.0, env.W] 폴백
    """
    if isinstance(raw_obs, dict):
        T = int(getattr(env, "T", 1) or 1)
        t_idx = float(raw_obs.get("t", 0.0))
        t_norm = t_idx / float(max(1, T - 1))
        W_now = float(raw_obs.get("W", getattr(env, "W", 0.0)))
        return _np.asarray([t_norm, W_now], dtype=float)
    if isinstance(raw_obs, _np.ndarray):
        arr = _np.asarray(raw_obs, dtype=float).ravel()
        if arr.size >= 2:
            return arr[:2]
        # 관측이 1차원만 온다면 W는 env.W 로 보강
        W_now = float(getattr(env, "W", 0.0))
        return _np.asarray([float(arr[0]) if arr.size >= 1 else 0.0, W_now], dtype=float)
    # 기타 폴백
    return _np.asarray([0.0, float(getattr(env, "W", 0.0))], dtype=float)


def _rollout_terminal_wealths(cfg: SimConfig,
                              actor: Callable[[Any], tuple[float, float]],
                              n_paths: int) -> List[float]:
    """
    evaluate가 경로를 안 주거나, 스칼라 복제 의심 시 방어적으로 직접 경로 샘플.
    - 동일 Env 사용(상태 스케일/필드 동일)
    - seed를 넘기지 않고 reset()만 반복 → 내부 path_counter/RNG로 경로 다양화
    - 상태는 항상 ndarray([t_norm, W])로 정책에 전달
    """
    env = RetirementEnv(cfg)
    WTs: List[float] = []
    n_paths = int(max(1, n_paths))

    for _ in range(n_paths):
        env.reset()  # seed 인자 미전달 → 경로 다양화는 Env 내부에 위임
        done = False
        while not done:
            raw = env._obs() if hasattr(env, "_obs") else None
            state_nd = _as_ndarray_state(raw, env)
            q, w = actor(state_nd)
            _, _, done, _, _ = env.step(q, w)
        WTs.append(float(getattr(env, "W", 0.0)))
    return WTs


def _looks_degenerate_wt(xs) -> bool:
    """경로 배열이 없거나, 전부 동일/거의 동일하면 True."""
    try:
        arr = _np.asarray(xs, dtype=float).ravel()
        if arr.size <= 1:
            return True
        return bool(_np.allclose(arr, arr[0], atol=0, rtol=0))
    except Exception:
        return True


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
        _wire_market_data(cfg, args)
        time_wire_data = time.perf_counter() - t1

        t2 = time.perf_counter()
        ann_enabled = (
            str(getattr(args, "ann_on", "off")).lower() == "on"
            and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0
        )
        if ann_enabled:
            setup_annuity_overlay(cfg, args)
        time_annuity = time.perf_counter() - t2

        t3 = time.perf_counter()
        actor = build_actor(cfg, args)
        time_build_actor = time.perf_counter() - t3

        t4 = time.perf_counter()
        m, extras = _call_evaluate(cfg, actor, es_mode=args.es_mode)
        time_eval = time.perf_counter() - t4

    # ---- 방어적 후처리: eval_WT 보정 ----
    extras = extras or {}
    need_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))

    wt_from_eval = extras.get("eval_WT", None)
    if (not isinstance(wt_from_eval, (list, tuple))) or _looks_degenerate_wt(wt_from_eval):
        if need_paths:
            n_paths = getattr(args, "n_paths", None)
            if n_paths is None or int(n_paths) <= 0:
                n_paths = int(getattr(cfg, "n_paths_eval", 0)) or 100
            try:
                WTs = _rollout_terminal_wealths(cfg, _to_actor(actor), int(n_paths))
                extras["eval_WT"] = [float(x) for x in WTs]
                m.setdefault("mean_WT", float(_np.mean(WTs)))
                m.setdefault("EW", float(_np.mean(WTs)))
            except Exception as _e:
                extras.setdefault("eval_WT_note", f"local rollout failed: {type(_e).__name__}")

    # --- 메트릭 병합: 연금 파생 파라미터는 항상 기록(미설정 시 0.0) ---
    if isinstance(m, dict):
        y_ann = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        a_fac = float(getattr(cfg, "ann_a_factor", 0.0) or 0.0)
        P_val = float(getattr(cfg, "ann_P", 0.0) or 0.0)
        m.update(
            {
                "y_ann": y_ann if (y_ann != 0.0) else 0.0,
                "ann_a_factor": a_fac if (a_fac != 0.0) else 0.0,
                "a_factor": a_fac if (a_fac != 0.0) else 0.0,
                "P": P_val if (P_val != 0.0) else 0.0,
            }
        )

    # 총 평가 경로 수(보고용)
    n_paths_total = len(extras.get("eval_WT", [])) or (
        getattr(cfg, "n_paths_eval", getattr(cfg, "n_paths", 0))
        * len(getattr(cfg, "seeds", []))
    )

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
        asset=getattr(cfg, "asset", None),
        method=args.method,
        baseline=getattr(args, "baseline", ""),
        metrics=m,
        w_max=getattr(cfg, "w_max", None),
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=getattr(cfg, "lambda_term", None),
        alpha=getattr(cfg, "alpha", None),
        F_target=getattr(cfg, "F_target", None),
        es_mode=args.es_mode,
        n_paths=int(n_paths_total),
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
) -> Optional[Callable[[Any], tuple[float, float]]]:
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
        _ = time.perf_counter() - t0
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

    best_epoch = fields.get("best_epoch")
    ckpt_path = fields.get("ckpt_path")
    train_time_s = fields.get("train_time_s")
    eval_time_s = fields.get("eval_time_s")

    policy_like = fields.get("actor") or fields.get("policy") or fields.get("pi") or fields.get("agent")
    t4 = time.perf_counter()
    actor: Optional[Callable[[Any], tuple[float, float]]] = None
    try:
        if policy_like is not None:
            actor = _to_actor(policy_like)
    except Exception:
        actor = None
    if actor is None:
        actor = _maybe_load_actor_from_ckpt(ckpt_path, cfg_hint=cfg)
    time_actor_load = time.perf_counter() - t4

    n_paths_total = int(getattr(args, "rl_n_paths_eval", 0)) * len(getattr(args, "seeds", []))

    if actor is not None and str(getattr(args, "return_actor", "off")).lower() == "on":
        return (cfg, actor)

    metrics_dict: Dict[str, Any]
    extras_dict: Dict[str, Any] = {}

    t5 = time.perf_counter()
    if actor is not None:
        try:
            metrics_dict, extras_dict = _call_evaluate(cfg, actor, es_mode=getattr(args, "es_mode", "wealth"))
            metrics_dict.update(
                {
                    "best_epoch": best_epoch,
                    "train_time_s": train_time_s,
                    "eval_time_s": eval_time_s,
                }
            )
            metrics_dict.setdefault("es95_source", "computed_in_evaluate")
        except Exception as e:
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

    time_total = time.perf_counter() - t_all_0
    timing = {
        "make_cfg_s": round(time_make_cfg, 6),
        "wire_data_s": round(time_wire_data, 6),
        "annuity_setup_s": round(time_annuity, 6),
        "train_call_s": round(time_train_call, 6),
        "actor_load_s": round(time_actor_load, 6),
        "evaluate_s": round(time_eval, 6),
        "total_s": round(time_total, 6),
        "total_hms": _fmt_hms(time_total),
    }

    out = dict(
        asset=getattr(cfg, "asset", None),
        method="rl",
        baseline="",
        metrics=metrics_dict,
        w_max=getattr(cfg, "w_max", None),
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=getattr(cfg, "lambda_term", None),
        alpha=getattr(cfg, "alpha", None),
        F_target=getattr(cfg, "F_target", None),
        es_mode=getattr(args, "es_mode", "wealth"),
        n_paths=n_paths_total,
        args=slim_args(args)
        | {
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

    if getattr(args, "autosave", "off") == "on":
        do_autosave(out.get("metrics") or {}, cfg, args, out)

    return out
