# project/policy/kgr_rule.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

# ---- Config / State ----
@dataclass
class KGRLiteConfig:
    # 고정 파라미터(Lite)
    FR_high: float = 1.30
    FR_low: float = 0.85
    delta_up: float = 0.07
    delta_dn: float = -0.07
    kappa_safety: float = 0.002  # = 0.20%p
    w_fixed: float = 0.60

    # 공통 제약(환경과 일치)
    q_floor: float = 0.02                    # 기본: 연 단위 바닥
    q_floor_is_annual: bool = True           # True면 월로 변환해 사용(헬퍼에서)
    phi_adval_annual: float = 0.004          # fee (연)
    steps_per_year: int = 12

    # Fallback 용: life_table이 없을 때 사용할 계획기간(연)
    horizon_years: int = 35


@dataclass
class KGRLiteState:
    age_years: float          # 현재 나이(년)
    B: float                  # 약속소비(연 기준), 명목
    q0: float                 # 초기 인출률(연)
    last_rail_year_idx: int   # 마지막 가드레일 적용 연(0-based)
    # 디버깅/로깅 보조
    FR_last: Optional[float] = None
    PVB_last: Optional[float] = None
    a_real_last: Optional[float] = None


# ---- Helpers (rates & floors) ----
def annual_to_monthly_rate(r_annual: float, steps_per_year: int = 12) -> float:
    """연간 비율 r을 동일효과의 월간 비율로 변환."""
    steps = max(int(steps_per_year or 12), 1)
    return 1.0 - (1.0 - float(r_annual)) ** (1.0 / steps)


def monthly_floor_from_cfg(cfg: KGRLiteConfig) -> float:
    """cfg.q_floor 해석에 따라 월 바닥율을 반환 (actor에서 클리핑용)."""
    if getattr(cfg, "q_floor_is_annual", True):
        return annual_to_monthly_rate(cfg.q_floor, getattr(cfg, "steps_per_year", 12))
    return float(cfg.q_floor)


def kgr_lite_q_monthly(q_annual: float, cfg: KGRLiteConfig, steps_per_year: Optional[int] = None) -> float:
    """
    [핵심 헬퍼] 연 q를 월 q로 변환한 뒤, cfg에 맞는 '월 바닥'으로 클리핑해 반환.
    actor 쪽에서: q_m = kgr_lite_q_monthly(out['q_t'], kgr_cfg)
    """
    spm = int(steps_per_year or getattr(cfg, "steps_per_year", 12) or 12)
    q_m = 1.0 - (1.0 - float(q_annual)) ** (1.0 / spm)
    q_floor_m = monthly_floor_from_cfg(cfg)
    return float(np.clip(q_m, float(q_floor_m), 1.0))


# ---- Helpers (annuity factors) ----
def _annuity_factor_real(
    life_table: pd.DataFrame,
    age: float,
    r_f_real_annual: float,
    max_age: int = 110,
    use_yearly: bool = True,
    steps_per_year: int = 12,
) -> float:
    """
    실질 연금계수 a_x^{real}.
    - life_table: 'age'와 ('px' 또는 'qx') 필요
    - use_yearly=True: 연 단위 합산 (Lite 권장)
    """
    age_int = int(np.floor(age))
    ages = np.arange(age_int, max_age + 1, dtype=int)

    if 'px' in life_table.columns:
        px = life_table.set_index('age').loc[ages, 'px'].values
    elif 'Px' in life_table.columns:
        px = life_table.set_index('age').loc[ages, 'Px'].values
    elif 'qx' in life_table.columns:
        qx = life_table.set_index('age').loc[ages, 'qx'].values
        px = 1.0 - qx
    else:
        raise ValueError("life_table에 'age'와 ('px' 또는 'qx') 컬럼이 필요합니다.")

    # 연 단위 생존확률 누적:  _k p_x
    kpx = np.ones_like(px, dtype=float)
    for k in range(1, len(px)):
        kpx[k] = kpx[k - 1] * float(px[k - 1])

    if use_yearly:
        disc = (1.0 + r_f_real_annual) ** np.arange(0, len(kpx))
        a_real = np.sum(kpx / disc)
    else:
        # (참고) 월 단위 내부 환산 버전(근사)
        r_m = r_f_real_annual / steps_per_year
        disc = (1.0 + r_m) ** np.arange(0, len(kpx) * steps_per_year)
        kpx_m = np.repeat(kpx, steps_per_year)
        a_real = np.sum(kpx_m / disc)

    return float(a_real)


def _annuity_factor_horizon_only(
    r_f_real_annual: float,
    years: int,
) -> float:
    """
    Mortality table이 없을 때 쓰는 Fallback: horizon 기반 실질 연금계수.
    a = sum_{k=0}^{years} 1/(1+r)^k
    """
    years = max(int(years), 0)
    disc = (1.0 + r_f_real_annual) ** np.arange(0, years + 1)
    return float(np.sum(1.0 / disc))


# ---- API ----
def kgr_lite_init(
    W0: float,
    age0: float,
    life_table: Optional[pd.DataFrame],
    r_f_real_annual: float,
    cfg: Optional[KGRLiteConfig] = None,
) -> KGRLiteState:
    """
    초기 인출률 q0와 B0(연 약속소비)를 계산하고 상태를 반환.
    - life_table이 없거나 포맷 이슈가 있으면 horizon-only annuity fallback 사용
    - a_x0_real은 '연 지급' 기준 연금계수이므로 q0_core = 1 / a_x0_real
    """
    cfg = cfg or KGRLiteConfig()

    # 1) 연금계수 계산 (life table 우선, 실패 시 horizon-only fallback)
    use_fallback = False
    if isinstance(life_table, pd.DataFrame):
        try:
            a_x0_real = _annuity_factor_real(
                life_table=life_table,
                age=age0,
                r_f_real_annual=r_f_real_annual,
                use_yearly=True,
                steps_per_year=cfg.steps_per_year,
            )
            print(f"[kgr:init] annuity=life_table based, a_real={a_x0_real:.6f}")
        except Exception as e:
            use_fallback = True
            print(f"[kgr:init] life_table error -> horizon-only fallback "
                  f"(years={cfg.horizon_years}) | err={e}")
    else:
        use_fallback = True
        print(f"[kgr:init] life_table missing -> horizon-only annuity (years={cfg.horizon_years})")

    if use_fallback:
        a_x0_real = _annuity_factor_horizon_only(
            r_f_real_annual=r_f_real_annual,
            years=cfg.horizon_years,
        )
        print(f"[kgr:init] a_real(fallback)={a_x0_real:.6f}")

    # NaN/Inf 방지
    if not np.isfinite(a_x0_real) or a_x0_real <= 0.0:
        a_x0_real = _annuity_factor_horizon_only(r_f_real_annual, cfg.horizon_years)
        print(f"[kgr:init] a_real invalid -> reset via fallback, a_real={a_x0_real:.6f}")

    # 2) 초기 인출률/약속소비 (연 단위)
    q0_core = 1.0 / max(a_x0_real, 1e-9)  # <- 단위 확정: 연 지급 계수의 역수
    q0_raw = q0_core - float(cfg.phi_adval_annual) - float(cfg.kappa_safety)
    q0 = max(float(cfg.q_floor), float(q0_raw))  # 여기서 q_floor는 '연' 해석(월 클리핑은 actor/헬퍼에서)
    B0 = float(q0) * float(W0)

    # 3) 로깅 (sanity)
    print(f"[kgr:init] q0_core={q0_core:.6f} (1/a_real), "
          f"fee={cfg.phi_adval_annual:.4f}, κ={cfg.kappa_safety:.4f} "
          f"-> q0={q0:.6f}, B0={B0:.6f}")
    if q0 > 0.20:
        print(f"[kgr:init][warn] q0 unusually high ({q0:.3f}). "
              f"Check life_table / r_f_real_annual / horizon_years.")

    return KGRLiteState(
        age_years=float(age0),
        B=float(B0),
        q0=float(q0),
        last_rail_year_idx=-1,
        FR_last=None,
        PVB_last=None,
        a_real_last=float(a_x0_real),
    )


def kgr_lite_update_yearly(
    W_t: float,
    age_years: float,
    CPI_yoy: float,
    life_table: Optional[pd.DataFrame],
    r_f_real_annual: float,
    state: KGRLiteState,
    cfg: Optional[KGRLiteConfig] = None,
) -> KGRLiteState:
    """
    연 1회 가드레일 조정(B_t 업데이트).
    - life_table이 없으면 CPI-only bump로 안전 동작
    """
    cfg = cfg or KGRLiteConfig()

    if not isinstance(life_table, pd.DataFrame):
        # Mortality/annuity 정보를 모르면 FR 계산 스킵
        state.B = float(state.B * (1.0 + float(CPI_yoy)))
        state.age_years = float(age_years)
        state.FR_last = None
        state.PVB_last = None
        state.a_real_last = None
        print(f"[kgr:year] CPI-only bump, B={state.B:.6f} (FR skipped; no life_table)")
        return state

    # PV(B) = B_{t-1} * a_{x+t}^{real}
    a_real = _annuity_factor_real(
        life_table, age_years, r_f_real_annual,
        use_yearly=True, steps_per_year=cfg.steps_per_year
    )
    PVB = float(state.B) * float(a_real)
    FR = (float(W_t) / PVB) if PVB > 0 else np.inf

    # 밴드에 따른 B 조정 (명목 인상 + 밴드 조정)
    if FR >= cfg.FR_high:
        B_new = state.B * (1.0 + CPI_yoy) * (1.0 + cfg.delta_up)
        tag = "▲up"
    elif FR <= cfg.FR_low:
        B_new = state.B * (1.0 + CPI_yoy) * (1.0 + cfg.delta_dn)
        tag = "▼down"
    else:
        B_new = state.B * (1.0 + CPI_yoy)
        tag = "—hold"

    state.B = float(B_new)
    state.age_years = float(age_years)
    state.FR_last = float(FR)
    state.PVB_last = float(PVB)
    state.a_real_last = float(a_real)
    print(f"[kgr:year] {tag} FR={FR:.4f}, a_real={a_real:.6f}, PVB={PVB:.6f}, B={state.B:.6f}")
    return state


def kgr_lite_policy_step(
    obs: Dict[str, float],
    state: KGRLiteState,
    cfg: Optional[KGRLiteConfig] = None,
) -> Dict[str, float]:
    """
    규칙 액터(한 스텝) 출력(연 기준 q_t).
    기대 입력 obs 예시:
      - 'W_t': 현재 자산
      - 'age_years': 현재 나이(년)
      - 'cpi_yoy': 전년동월 대비 CPI(연율)
      - 'is_new_year': 연초 여부(bool→0/1)
      - 'y_t', 'y_ann_t' (선택)
      - (옵션) 'life_table'(pd.DataFrame), 'r_f_real_annual'(float)
    출력:
      - 'q_t'(annual): 인출률 (연 기준; 월은 actor/헬퍼에서 연→월 변환·클리핑)
      - 'w_t': 위험자산 비중
      - 'B_t': 약속소비(연)
    """
    cfg = cfg or KGRLiteConfig()
    W_t = float(obs.get('W_t', 0.0))
    y_t = float(obs.get('y_t', 0.0))
    y_ann_t = float(obs.get('y_ann_t', 0.0))
    cpi_yoy = float(obs.get('cpi_yoy', 0.0))
    is_new_year = bool(obs.get('is_new_year', 0))
    age_years = float(obs.get('age_years', state.age_years))

    # 연 1회 가드레일 업데이트
    if is_new_year:
        lt = obs.get('life_table', None)
        rf = obs.get('r_f_real_annual', None)
        if isinstance(lt, pd.DataFrame) and isinstance(rf, (int, float)):
            kgr_lite_update_yearly(
                W_t=W_t,
                age_years=age_years,
                CPI_yoy=cpi_yoy,
                life_table=lt,
                r_f_real_annual=float(rf),
                state=state,
                cfg=cfg,
            )
        else:
            # life_table/real rf가 없으면 CPI 인상만
            state.B = float(state.B * (1.0 + cpi_yoy))
            state.age_years = float(age_years)
            print(f"[kgr:year] CPI-only bump in policy_step, B={state.B:.6f} (no life_table in obs)")

    # 소비 타겟(연) → 현재 자산 대비 인출률 근사 (연 단위 floor 해석)
    # c_t(annual) = max(q_floor_annual * W_t, y_t + y_ann_t + B_t)
    target_c_annual = max(float(cfg.q_floor) * W_t, y_t + y_ann_t + state.B)
    q_t_annual = 0.0 if W_t <= 0 else (target_c_annual / max(W_t, 1e-12))

    return {
        "q_t": float(q_t_annual),         # annual — 월 변환/클리핑은 actor에서 kgr_lite_q_monthly 사용
        "w_t": float(cfg.w_fixed),
        "B_t": float(state.B),
    }
