from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict

# === CVaR calibration defaults (used by run_experiment CLI defaults) ===
CVAR_TARGET_DEFAULT: Optional[float] = None   # e.g., 0.45
CVAR_TOL_DEFAULT: float = 0.01
LAMBDA_MIN_DEFAULT: float = 0.0
LAMBDA_MAX_DEFAULT: float = 5.0
LAMBDA_MAX_ITER: int = 14

# === Hedge defaults (MVP-level toggle) ===
HEDGE_ON_DEFAULT: bool = False
HEDGE_MODE_DEFAULT: Literal["mu", "sigma"] = "mu"   # μ haircut or σ reduction
HEDGE_COST_DEFAULT: float = 0.005                   # annual haircut on risky mu
HEDGE_SIGMA_K_DEFAULT: float = 0.20                 # fraction reduction on σ

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
    phi_adval: float = 0.004  # ad-valorem fee (annual)
    allow_short: bool = False
    allow_leverage: bool = False

    # --- Floor (level) ---
    floor_on: bool = False
    f_min_real: float = 0.0  # level floor per period in wealth units

    # --- Objective (CVaR@alpha with RU-dual at terminal) ---
    alpha: float = 0.95
    lambda_term: float = 0.0
    F_target: float = 0.0  # target terminal wealth for loss definition

    # --- Baseline (rule-based) options ---
    baseline: Optional[Literal["4pct", "cpb", "vpw"]] = None
    p_annual: float = 0.04          # constant-%-of-balance (annual)
    g_real_annual: float = 0.02     # VPW growth assumption (real)
    w_fixed: Optional[float] = None # fixed risky weight for rules

    # --- RL placeholders ---
    rl_lr: float = 3e-4
    rl_gamma: float = 0.996
    rl_epochs: int = 60
    rl_steps_per_epoch: int = 2048
    rl_hidden: int = 64

    # --- HJB grids (wealth / action / eta) ---
    hjb_W_min: float = 0.0
    hjb_W_max: float = 5.0
    hjb_W_grid: int = 121                         # 논문 기본 해상도(201도 가능)
    # w-grid: 자동 생성(0~w_max, 균등분할). 직접 지정 시 None이 아닌 튜플로 주입.
    hjb_w_grid: Optional[Tuple[float, ...]] = None
    hjb_w_grid_n: int = 8                         # 0~w_max를 n등분 (기본 8점)
    # η-grid 자동 생성을 위해 기본은 비움 → __post_init__에서 0..F_target 구간 linspace
    hjb_eta_grid: Tuple[float, ...] = ()
    hjb_eta_n: int = 81                           # η 격자 점수(기본 81; dev 41~61 권장)

    # MC samples for expectation inside HJB
    hjb_Nshock: int = 1024

    # --- Eval / bookkeeping ---
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_paths_eval: int = 500
    tag: str = "dev"

    # --- Hedge options (MVP toggle) ---
    hedge_on: bool = HEDGE_ON_DEFAULT
    hedge_mode: Literal["mu", "sigma"] = HEDGE_MODE_DEFAULT
    hedge_cost: float = HEDGE_COST_DEFAULT
    hedge_sigma_k: float = HEDGE_SIGMA_K_DEFAULT

    # --- DEV-ONLY knobs (turn off for paper runs) ---
    w_min_dev: float = 0.0            # 개발용 최소 risky 비중(본실험은 0.0 권장)
    dev_cvar_stage: bool = False      # 개발용 stage-wise tail penalty
    dev_cvar_kappa: float = 10.0      # stage penalty intensity(DEV에서만 영향)
    dev_w2_penalty: float = 0.02      # 개발용 w^2 penalty
    dev_split_w_grid: bool = False    # λ에 따라 w-grid 분리(개발 가시화)

    def __post_init__(self):
        # η-grid 자동 세팅: 미지정이면 [0, F_target] 구간을 hjb_eta_n 점으로 생성
        if not self.hjb_eta_grid or len(self.hjb_eta_grid) <= 3:
            try:
                import numpy as _np
                F = float(self.F_target or 1.0)
                n = int(self.hjb_eta_n or 41)
                self.hjb_eta_grid = tuple(float(x) for x in _np.linspace(0.0, F, n))
            except Exception:
                # numpy 사용 불가 시 안전 fallback
                n = max(int(self.hjb_eta_n or 41), 1)
                F = float(self.F_target or 1.0)
                step = F / (n - 1 if n > 1 else 1)
                self.hjb_eta_grid = tuple(0.0 + i * step for i in range(n))

        # w-grid 자동 세팅: 미지정(None)일 때 0~w_max 균등분할(hjb_w_grid_n 점)
        if self.hjb_w_grid is None or len(self.hjb_w_grid) == 0:
            n = max(int(self.hjb_w_grid_n or 8), 2)
            try:
                import numpy as _np
                self.hjb_w_grid = tuple(float(x) for x in _np.linspace(0.0, float(self.w_max), n))
            except Exception:
                step = float(self.w_max) / (n - 1)
                self.hjb_w_grid = tuple(0.0 + i * step for i in range(n))

    def monthly(self) -> Dict[str, float]:
        """Monthly conversions keeping units consistent."""
        mu_m = (1.0 + self.mu_annual) ** (1.0 / self.steps_per_year) - 1.0
        sigma_m = self.sigma_annual / (self.steps_per_year ** 0.5)
        rf_m = (1.0 + self.rf_annual) ** (1.0 / self.steps_per_year) - 1.0
        phi_m = 1.0 - (1.0 - self.phi_adval) ** (1.0 / self.steps_per_year)
        p_m = 1.0 - (1.0 - self.p_annual) ** (1.0 / self.steps_per_year)
        g_m = (1.0 + self.g_real_annual) ** (1.0 / self.steps_per_year) - 1.0
        beta_m = self.rl_gamma  # reuse rl_gamma as monthly discount
        return dict(mu_m=mu_m, sigma_m=sigma_m, rf_m=rf_m,
                    phi_m=phi_m, p_m=p_m, g_m=g_m, beta_m=beta_m)

ASSET_PRESETS = {
    "KR":   dict(mu_annual=0.06,  sigma_annual=0.20),
    "US":   dict(mu_annual=0.065, sigma_annual=0.16),
    "Gold": dict(mu_annual=0.03,  sigma_annual=0.15),
}
