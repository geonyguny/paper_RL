# project/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, Dict, Sequence, Union

# === CVaR calibration defaults (used by CLI defaults) ===
CVAR_TARGET_DEFAULT: Optional[float] = None   # e.g., 0.45
CVAR_TOL_DEFAULT: float = 0.01
LAMBDA_MIN_DEFAULT: float = 0.0
LAMBDA_MAX_DEFAULT: float = 5.0
LAMBDA_MAX_ITER: int = 14

# === Hedge defaults (MVP-level toggle) ===
HEDGE_ON_DEFAULT: bool = False
HEDGE_MODE_DEFAULT: Literal["mu", "sigma", "downside"] = "sigma"
HEDGE_COST_DEFAULT: float = 0.005                   # annual premium / haircut
HEDGE_SIGMA_K_DEFAULT: float = 0.20                 # fraction reduction on σ

# 타입 헬퍼
FloatGrid = Union[Sequence[float], Tuple[float, ...]]


@dataclass
class SimConfig:
    # --- Core asset/market settings ---
    asset: Literal["KR", "US", "Gold"] = "KR"
    mu_annual: float = 0.06
    sigma_annual: float = 0.20
    rf_annual: float = 0.02

    # --- Horizon / time resolution ---
    horizon_years: int = 35
    steps_per_year: int = 12

    # --- Controls / constraints ---
    w_max: float = 0.70
    # 과거(phi_adval)와 최신(fee_annual) 동시 지원
    fee_annual: float = 0.004       # ad-valorem fee (annual)
    phi_adval: Optional[float] = None  # 구버전 alias; 주어지면 fee_annual 대신 사용
    allow_short: bool = False
    allow_leverage: bool = False

    # --- Floor (level) ---
    floor_on: bool = False
    f_min_real: float = 0.0  # level floor per period in wealth units

    # --- Objective (CVaR@alpha with RU-dual at terminal) ---
    alpha: float = 0.95
    lambda_term: float = 0.0
    F_target: float = 0.0  # loss 정의용 목표 최종자산 (wealth 모드일 땐 참고값)

    # --- Baseline (rule-based) options ---
    baseline: Optional[Literal["4pct", "cpb", "vpw", "kgr"]] = None
    p_annual: float = 0.04          # constant-%-of-balance (annual)
    g_real_annual: float = 0.02     # VPW growth assumption (real)
    w_fixed: Optional[float] = None # fixed risky weight for rules

    # --- RL placeholders ---
    rl_lr: float = 3e-4
    rl_gamma: float = 0.996
    rl_epochs: int = 60
    rl_steps_per_epoch: int = 2048
    rl_hidden: int = 64
    rl_q_cap: float = 0.0

    # --- HJB grids (wealth / action / eta) ---
    hjb_W_min: float = 0.0
    hjb_W_max: float = 5.0
    hjb_W_grid: int = 121                         # 논문 기본 해상도(201도 가능)
    # w-grid: 자동 생성(0~w_max, 균등분할). 직접 지정 시 리스트/튜플 모두 허용.
    hjb_w_grid: Optional[FloatGrid] = None
    hjb_w_grid_n: int = 8                         # 0~w_max를 n등분 (기본 8점)
    # η-grid 자동 생성을 위해 기본은 비움 → __post_init__에서 0..F_target 구간 linspace
    hjb_eta_grid: Tuple[float, ...] = field(default_factory=tuple)
    hjb_eta_n: int = 81                           # η 격자 점수(기본 81; dev 41~61 권장)

    # MC samples for expectation inside HJB
    hjb_Nshock: int = 1024

    # --- Eval / bookkeeping ---
    # seeds는 리스트/튜플 모두 허용 → 내부적으로 튜플로 고정
    seeds: Union[Tuple[int, ...], Sequence[int]] = (0, 1, 2, 3, 4)
    n_paths_eval: int = 500
    tag: str = "dev"

    # --- Hedge options (MVP toggle) ---
    hedge_on: bool = HEDGE_ON_DEFAULT
    hedge_mode: Literal["mu", "sigma", "downside"] = HEDGE_MODE_DEFAULT
    hedge_cost: float = HEDGE_COST_DEFAULT        # = hedge_premium_annual
    hedge_sigma_k: float = HEDGE_SIGMA_K_DEFAULT  # σ 감소 비율(또는 downside 강도)
    hedge_tx: float = 0.0                         # 거래비용(월 환산은 env에서)

    # --- DEV-ONLY knobs (turn off for paper runs) ---
    w_min_dev: float = 0.0            # 개발용 최소 risky 비중(본실험은 0.0 권장)
    dev_cvar_stage: bool = False      # 개발용 stage-wise tail penalty
    dev_cvar_kappa: float = 10.0      # stage penalty intensity(DEV에서만 영향)
    dev_w2_penalty: float = 0.02      # 개발용 w^2 penalty
    dev_split_w_grid: bool = False    # λ에 따라 w-grid 분리(개발 가시화)

    # --- Market (iid / bootstrap CSV) ---
    market_mode: Literal["iid", "bootstrap"] = "iid"
    market_csv: Optional[str] = None
    bootstrap_block: int = 24
    use_real_rf: Literal["on", "off"] = "on"

    # --- Mortality / bequest (옵션) ---
    mortality: Literal["on", "off"] = "off"
    mort_table: Optional[str] = None
    age0: int = 65
    sex: Literal["M", "F"] = "M"
    bequest_kappa: float = 0.0
    bequest_gamma: float = 1.0

    # --- Annuity overlay (옵션: env에서 사용) ---
    ann_on: Literal["on", "off"] = "off"
    ann_alpha: float = 0.0
    ann_L: float = 0.0
    ann_d: int = 0
    ann_index: Literal["real", "nominal"] = "real"
    y_ann: float = 0.0  # 외부 고정지급(있다면)

    # --- XAI / IO / misc ---
    xai_on: Literal["on", "off"] = "on"
    quiet: Literal["on", "off"] = "on"
    bands: Literal["on", "off"] = "on"
    outputs: str = "./outputs"
    data_profile: Optional[Literal["dev", "full"]] = None
    data_window: Optional[str] = None

    # --- Allocation & FX hedge (멀티에셋일 때 사용 가능) ---
    alpha_mix: Optional[str] = None
    alpha_kr: Optional[float] = None
    alpha_us: Optional[float] = None
    alpha_au: Optional[float] = None
    h_FX: Optional[float] = None
    fx_hedge_cost: Optional[float] = None

    # --- Lite overrides (optional) ---
    q_floor: Optional[float] = None
    beta: Optional[float] = None

    # --- Stage-wise CVaR (optional) ---
    cvar_stage: Literal["on", "off"] = "off"
    alpha_stage: float = 0.95
    lambda_stage: float = 0.0
    cstar_mode: Literal["fixed", "annuity", "vpw"] = "annuity"
    cstar_m: float = 0.04 / 12

    def __post_init__(self):
        # seeds: 튜플로 고정
        if not isinstance(self.seeds, tuple):
            try:
                self.seeds = tuple(int(s) for s in self.seeds)  # type: ignore
            except Exception:
                self.seeds = (0,)

        # η-grid 자동 세팅: 미지정이면 [0, F_target] 구간을 hjb_eta_n 점으로 생성
        if not self.hjb_eta_grid or len(self.hjb_eta_grid) <= 3:
            try:
                import numpy as _np
                F = float(self.F_target or 1.0)
                n = int(self.hjb_eta_n or 41)
                self.hjb_eta_grid = tuple(float(x) for x in _np.linspace(0.0, F, n))
            except Exception:
                n = max(int(self.hjb_eta_n or 41), 1)
                F = float(self.F_target or 1.0)
                step = F / (n - 1 if n > 1 else 1)
                self.hjb_eta_grid = tuple(0.0 + i * step for i in range(n))

        # w-grid 자동 세팅: 미지정(None/빈)일 때 0~w_max 균등분할(hjb_w_grid_n 점)
        if not self.hjb_w_grid:
            n = max(int(self.hjb_w_grid_n or 8), 2)
            try:
                import numpy as _np
                self.hjb_w_grid = tuple(float(x) for x in _np.linspace(0.0, float(self.w_max), n))
            except Exception:
                step = float(self.w_max) / (n - 1)
                self.hjb_w_grid = tuple(0.0 + i * step for i in range(n))
        else:
            # 사용자가 리스트/튜플로 준 그리드를 정규화(0~w_max 범위, 오름차순, 중복제거)
            try:
                vals = [float(x) for x in self.hjb_w_grid]  # type: ignore[arg-type]
                vals = [min(max(0.0, x), float(self.w_max)) for x in vals]
                vals = sorted(set(vals))
                if len(vals) < 2:
                    # 안전장치: 최소 2점은 필요
                    vals = [0.0, float(self.w_max)]
                self.hjb_w_grid = tuple(vals)
            except Exception:
                # 실패 시 균등분할로 대체
                n = max(int(self.hjb_w_grid_n or 8), 2)
                step = float(self.w_max) / (n - 1)
                self.hjb_w_grid = tuple(0.0 + i * step for i in range(n))

    def monthly(self) -> Dict[str, float]:
        """월간 단위로 변환(수수료는 ad-valorem 기준)."""
        mu_m = (1.0 + self.mu_annual) ** (1.0 / self.steps_per_year) - 1.0
        sigma_m = self.sigma_annual / (self.steps_per_year ** 0.5)
        rf_m = (1.0 + self.rf_annual) ** (1.0 / self.steps_per_year) - 1.0

        # fee: 최신 필드(fee_annual)를 우선, 없으면 phi_adval 사용
        fee_annual = float(self.fee_annual if self.phi_adval is None else self.phi_adval)
        # ad-valorem을 “월간 *비율*”로 변환 (복리기준 역산)
        phi_m = 1.0 - (1.0 - fee_annual) ** (1.0 / self.steps_per_year)

        p_m = 1.0 - (1.0 - self.p_annual) ** (1.0 / self.steps_per_year)
        g_m = (1.0 + self.g_real_annual) ** (1.0 / self.steps_per_year) - 1.0
        beta_m = self.rl_gamma  # 월간 할인인자(간편 재사용)

        return dict(mu_m=mu_m, sigma_m=sigma_m, rf_m=rf_m,
                    phi_m=phi_m, p_m=p_m, g_m=g_m, beta_m=beta_m)


ASSET_PRESETS = {
    "KR":   dict(mu_annual=0.06,  sigma_annual=0.20),
    "US":   dict(mu_annual=0.065, sigma_annual=0.16),
    "Gold": dict(mu_annual=0.03,  sigma_annual=0.15),
}
