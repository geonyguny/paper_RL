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
    q_floor: float = 0.02
    phi_adval_annual: float = 0.004  # fee (연)
    steps_per_year: int = 12

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

# ---- Helpers ----
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
    - life_table: index(or col 'age')에 정수나이, 생존확률 px 또는 사망률 qx 보유
      허용 컬럼: 'px', 'Px', '_kpx', 'qx' (우선순위 px→qx)
    - use_yearly=True: 연 단위 간단 합산 (Lite 권장)
    """
    # age는 소수점 가능(예: 65.75); 기준 연령은 내림
    age_int = int(np.floor(age))
    ages = np.arange(age_int, max_age + 1, dtype=int)

    # 생존확률 pmf 추출
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
    kpx = np.ones_like(px)
    for k in range(1, len(px)):
        kpx[k] = kpx[k - 1] * px[k - 1]

    if use_yearly:
        disc = (1.0 + r_f_real_annual) ** np.arange(0, len(kpx))
        a_real = np.sum(kpx / disc)
    else:
        # (참고) 월 단위 내부 환산 버전
        r_m = r_f_real_annual / steps_per_year
        disc = (1.0 + r_m) ** np.arange(0, len(kpx) * steps_per_year)
        # 연간 생존확률을 월로 단순 확장하는 엄밀치 않은 근사(라이트판 비권장)
        kpx_m = np.repeat(kpx, steps_per_year)
        a_real = np.sum(kpx_m / disc)

    return float(a_real)

# ---- API ----
def kgr_lite_init(
    W0: float,
    age0: float,
    life_table: pd.DataFrame,
    r_f_real_annual: float,
    cfg: Optional[KGRLiteConfig] = None,
) -> KGRLiteState:
    """
    초기 인출률 q0와 B0(연 약속소비)를 계산하고 상태를 반환.
    """
    cfg = cfg or KGRLiteConfig()
    a_x0_real = _annuity_factor_real(life_table, age0, r_f_real_annual, use_yearly=True, steps_per_year=cfg.steps_per_year)

    q0_core = 12.0 / a_x0_real  # 연 환산 인출률 기반
    q0 = max(cfg.q_floor, q0_core - cfg.phi_adval_annual - cfg.kappa_safety)
    B0 = q0 * W0  # 연 기준 약속소비

    return KGRLiteState(
        age_years=age0,
        B=B0,
        q0=q0,
        last_rail_year_idx=-1,
        FR_last=None,
        PVB_last=None,
        a_real_last=a_x0_real,
    )

def kgr_lite_update_yearly(
    W_t: float,
    age_years: float,
    CPI_yoy: float,
    life_table: pd.DataFrame,
    r_f_real_annual: float,
    state: KGRLiteState,
    cfg: Optional[KGRLiteConfig] = None,
) -> KGRLiteState:
    """
    연 1회 가드레일 조정(B_t 업데이트).
    """
    cfg = cfg or KGRLiteConfig()

    # PV(B) = B_{t-1} * a_{x+t}^{real}
    a_real = _annuity_factor_real(life_table, age_years, r_f_real_annual, use_yearly=True, steps_per_year=cfg.steps_per_year)
    PVB = state.B * a_real
    FR = (W_t / PVB) if PVB > 0 else np.inf

    # 밴드에 따른 B 조정 (명목 인상 + 밴드 조정)
    if FR >= cfg.FR_high:
        B_new = state.B * (1.0 + CPI_yoy) * (1.0 + cfg.delta_up)
    elif FR <= cfg.FR_low:
        B_new = state.B * (1.0 + CPI_yoy) * (1.0 + cfg.delta_dn)
    else:
        B_new = state.B * (1.0 + CPI_yoy)

    state.B = float(B_new)
    state.age_years = age_years
    state.FR_last = float(FR)
    state.PVB_last = float(PVB)
    state.a_real_last = float(a_real)
    return state

def kgr_lite_policy_step(
    obs: Dict[str, float],
    state: KGRLiteState,
    cfg: Optional[KGRLiteConfig] = None,
) -> Dict[str, float]:
    """
    규칙 액터(한 스텝) 출력.
    기대 입력 obs 예시:
      - 't': 현재 스텝(0-based)
      - 'W_t': 현재 자산
      - 'age_years': 현재 나이(년)
      - 'cpi_yoy': 전년동월 대비 CPI(연율)
      - 'is_new_year': 연초 여부(bool→0/1)
      - 'y_t', 'y_ann_t' (선택)
    출력:
      - 'q_t': 인출률 (월은 env에서 연→월로 처리됨)
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
        # life_table, rf는 env에서 전달하는 것이 이상적이나,
        # Lite에서는 obs에 없으면 업데이트 생략(=CPI 인상만 반영)하도록 처리 가능.
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
            state.age_years = age_years

    # 소비 타겟(연) → 현재 자산 대비 인출률 근사
    # c_t = max(q_floor*W_t, y_t + y_ann_t + B_t)
    target_c = max(cfg.q_floor * W_t, y_t + y_ann_t + state.B)
    q_t = 0.0 if W_t <= 0 else target_c / W_t  # 연 기준 비율 추정
    # env에서 clipping → consumption → returns → fee 순서를 지킴

    return {
        "q_t": float(q_t),
        "w_t": float(cfg.w_fixed),
        "B_t": float(state.B),
    }
