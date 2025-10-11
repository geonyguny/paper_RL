# project/runner/config_build.py
from __future__ import annotations
from typing import Any, Iterable
import numpy as _np
from ..config import SimConfig, ASSET_PRESETS
from .helpers import auto_eta_grid

def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)

def _as_tuple(x, *, empty_ok: bool = True):
    if x is None:
        return tuple() if empty_ok else (0,)
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)

def _normalize_hjb_w_grid(cfg: SimConfig, raw) -> None:
    """
    raw가:
      - int: 해당 개수로 0..w_max 균등격자
      - Iterable[float]: 그대로 정규화하여 tuple로
      - None/빈값: 기본 8개 격자
    """
    w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)
    if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
        n_w = 8
        grid = _np.linspace(0.0, w_max, n_w)
    elif isinstance(raw, int):
        n_w = max(2, int(raw))
        grid = _np.linspace(0.0, w_max, n_w)
    elif isinstance(raw, Iterable):
        arr = _np.asarray(list(raw), dtype=float)
        if arr.size < 2:
            arr = _np.linspace(0.0, w_max, 8)
        # 0..w_max 범위로 클리핑
        arr = _np.clip(arr, 0.0, w_max)
        # 중복 제거 + 정렬
        grid = _np.unique(_np.round(arr, 6))
        if grid.size < 2:
            grid = _np.linspace(0.0, w_max, 8)
    else:
        # 미지정 형태 → 기본 8개
        grid = _np.linspace(0.0, w_max, 8)
    cfg.hjb_w_grid = tuple(_np.round(grid, 4))

def _set_fee_fields(cfg: SimConfig, args) -> None:
    """
    phi_adval(선취) vs fee_annual(연 운용보수) 동시 지원.
    - 둘 다 주어지면 phi_adval 우선.
    - 하나만 주어지면 그 값으로 둘 다 세팅(하위 호환).
    """
    fee_annual = _get(args, "fee_annual", None)
    phi_adval  = _get(args, "phi_adval", None)

    if phi_adval is not None:
        cfg.phi_adval = float(phi_adval)
        cfg.fee_annual = float(fee_annual if fee_annual is not None else phi_adval)
    elif fee_annual is not None:
        cfg.fee_annual = float(fee_annual)
        cfg.phi_adval = float(fee_annual)
    else:
        # 디폴트 0.004 유지
        cfg.fee_annual = float(getattr(cfg, "fee_annual", 0.004) or 0.004)
        cfg.phi_adval = float(getattr(cfg, "phi_adval", cfg.fee_annual) or cfg.fee_annual)

def _set_q_floor_monthly(cfg: SimConfig, args) -> None:
    """
    입력 q_floor가 '연환산 소비비율'이면 월로 변환하여 cfg.q_floor에 저장.
    - args.q_floor가 None이면 기존 cfg 값 유지(또는 0.0).
    """
    qf_ann = _get(args, "q_floor", None)
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    if qf_ann is None:
        # 이미 cfg에 값이 있으면 그대로 사용(없으면 0)
        qf_m = float(getattr(cfg, "q_floor", 0.0) or 0.0)
        cfg.q_floor = float(qf_m)
        cfg.q_floor_annual = float(_get(cfg, "q_floor_annual", 0.0) or 0.0)
        return

    qf_ann = float(qf_ann)
    # 수치 안정화(0~0.999999)
    qf_ann = max(0.0, min(0.999999, qf_ann))
    qf_m = 1.0 - (1.0 - qf_ann) ** (1.0 / spm)
    cfg.q_floor = float(qf_m)
    cfg.q_floor_annual = float(qf_ann)
    print(f"[cfg] q_floor_annual={qf_ann:.6f} → q_floor_monthly={qf_m:.6f} (steps_per_year={spm})")

def _choose_n_paths_eval(args, default_val: int = 100) -> int:
    n = int(_get(args, "n_paths", 0) or 0)
    if n <= 0:
        n = default_val
    return int(n)

def make_cfg(args) -> SimConfig:
    cfg = SimConfig()

    # 0) 자산 프리셋 반영
    if _get(args, "asset", None) in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[args.asset].items():  # type: ignore
            setattr(cfg, k, v)
    cfg.asset = _get(args, "asset", getattr(cfg, "asset", None))

    # 1) steps_per_year & horizon 확정(최소값 보장)
    steps_per_year = int(_get(args, "steps_per_year", getattr(cfg, "steps_per_year", 12)) or 12)
    steps_per_year = max(1, steps_per_year)
    cfg.steps_per_year = steps_per_year
    cfg.horizon_years = int(_get(args, "horizon_years", getattr(cfg, "horizon_years", 15)) or 15)

    # 2) 주 요인자 일괄 주입
    bulk = dict(
        w_max=_get(args, "w_max", getattr(cfg, "w_max", None)),
        horizon_years=cfg.horizon_years,
        lambda_term=_get(args, "lambda_term", getattr(cfg, "lambda_term", None)),
        alpha=_get(args, "alpha", getattr(cfg, "alpha", None)),
        baseline=_get(args, "baseline", getattr(cfg, "baseline", None)),
        p_annual=_get(args, "p_annual", getattr(cfg, "p_annual", None)),
        g_real_annual=_get(args, "g_real_annual", getattr(cfg, "g_real_annual", None)),
        w_fixed=_get(args, "w_fixed", getattr(cfg, "w_fixed", None)),
        floor_on=_get(args, "floor_on", getattr(cfg, "floor_on", None)),
        f_min_real=_get(args, "f_min_real", getattr(cfg, "f_min_real", None)),
        F_target=_get(args, "F_target", getattr(cfg, "F_target", None)),
        hjb_W_grid=_get(args, "hjb_W_grid", getattr(cfg, "hjb_W_grid", None)),
        hjb_Nshock=_get(args, "hjb_Nshock", getattr(cfg, "hjb_Nshock", None)),
        hedge=_get(args, "hedge", getattr(cfg, "hedge", "off")),
        hedge_on=(str(_get(args, "hedge", "off")).lower() == "on"),
        hedge_mode=_get(args, "hedge_mode", getattr(cfg, "hedge_mode", None)),
        hedge_cost=_get(args, "hedge_cost", getattr(cfg, "hedge_cost", None)),
        hedge_sigma_k=_get(args, "hedge_sigma_k", getattr(cfg, "hedge_sigma_k", None)),
        hedge_tx=_get(args, "hedge_tx", getattr(cfg, "hedge_tx", None)),
        market_mode=_get(args, "market_mode", getattr(cfg, "market_mode", "iid")),
        market_csv=_get(args, "market_csv", getattr(cfg, "market_csv", "")),
        bootstrap_block=_get(args, "bootstrap_block", getattr(cfg, "bootstrap_block", 24)),
        use_real_rf=_get(args, "use_real_rf", getattr(cfg, "use_real_rf", "on")),
        mortality=_get(args, "mortality", getattr(cfg, "mortality", "off")),
        mortality_on=(str(_get(args, "mortality", "off")).lower() == "on"),
        mort_table=_get(args, "mort_table", getattr(cfg, "mort_table", None)),
        age0=_get(args, "age0", getattr(cfg, "age0", 65)),
        sex=_get(args, "sex", getattr(cfg, "sex", "M")),
        bequest_kappa=_get(args, "bequest_kappa", getattr(cfg, "bequest_kappa", None)),
        bequest_gamma=_get(args, "bequest_gamma", getattr(cfg, "bequest_gamma", None)),
        rl_q_cap=_get(args, "rl_q_cap", getattr(cfg, "rl_q_cap", None)),
        teacher_eps0=_get(args, "teacher_eps0", getattr(cfg, "teacher_eps0", None)),
        teacher_decay=_get(args, "teacher_decay", getattr(cfg, "teacher_decay", None)),
        lw_scale=_get(args, "lw_scale", getattr(cfg, "lw_scale", None)),
        survive_bonus=_get(args, "survive_bonus", getattr(cfg, "survive_bonus", None)),
        crra_gamma=_get(args, "crra_gamma", getattr(cfg, "crra_gamma", None)),
        u_scale=_get(args, "u_scale", getattr(cfg, "u_scale", None)),
        cvar_stage_on=(str(_get(args, "cvar_stage", "off")).lower() == "on"),
        alpha_stage=_get(args, "alpha_stage", getattr(cfg, "alpha_stage", None)),
        lambda_stage=_get(args, "lambda_stage", getattr(cfg, "lambda_stage", None)),
        cstar_mode=_get(args, "cstar_mode", getattr(cfg, "cstar_mode", None)),
        cstar_m=_get(args, "cstar_m", getattr(cfg, "cstar_m", None)),
        xai_on=(str(_get(args, "xai_on", "off")).lower() == "on"),
        beta=_get(args, "beta", getattr(cfg, "beta", None)),
        ann_on=_get(args, "ann_on", getattr(cfg, "ann_on", "off")),
        ann_alpha=_get(args, "ann_alpha", getattr(cfg, "ann_alpha", 0.0)),
        ann_L=_get(args, "ann_L", getattr(cfg, "ann_L", 0.0)),
        ann_d=_get(args, "ann_d", getattr(cfg, "ann_d", 0)),
        ann_index=_get(args, "ann_index", getattr(cfg, "ann_index", "real")),
        tag=_get(args, "tag", getattr(cfg, "tag", None)),
        steps_per_year=steps_per_year,  # 확정 반영
    )
    for k, v in bulk.items():
        if v is not None:
            setattr(cfg, k, v)

    # 3) 수수료 필드 정합성(우선순위: phi_adval > fee_annual)
    _set_fee_fields(cfg, args)

    # 4) seeds & n_paths_eval 확정
    seeds = list(_as_tuple(_get(args, "seeds", (0,))))
    if len(seeds) == 0:
        seeds = [0]
    cfg.seeds = tuple(int(s) for s in seeds)

    cfg.n_paths_eval = _choose_n_paths_eval(args, default_val=100)
    # RL용 평가 수 경로도 기본 채움(없으면 n_paths_eval 재사용)
    if getattr(args, "rl_n_paths_eval", None) is not None:
        cfg.rl_n_paths_eval = int(args.rl_n_paths_eval)
    else:
        cfg.rl_n_paths_eval = int(cfg.n_paths_eval)

    # 5) 출력/메타
    cfg.outputs = _get(args, "outputs", getattr(cfg, "outputs", "./outputs"))
    cfg.method = _get(args, "method", getattr(cfg, "method", "hjb"))
    cfg.es_mode = _get(args, "es_mode", getattr(cfg, "es_mode", "wealth"))

    # 6) q_floor(연→월) 안전 변환
    _set_q_floor_monthly(cfg, args)

    # 7) HJB 격자/충격 수 정규화
    _normalize_hjb_w_grid(cfg, _get(args, "hjb_w_grid", getattr(cfg, "hjb_w_grid", None)))
    # 최소 충격 수 하한
    cfg.hjb_Nshock = max(int(getattr(cfg, "hjb_Nshock", 256) or 256), 256)

    # 8) ETA grid 자동 구성
    auto_eta_grid(cfg, requested_n=_get(args, "hjb_eta_n", None))

    # 디버그 출력(선택)
    # print(f"[cfg] seeds={cfg.seeds}, n_paths_eval={cfg.n_paths_eval}, steps_per_year={cfg.steps_per_year}")
    return cfg
