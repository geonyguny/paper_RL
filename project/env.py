import numpy as _np
from typing import Optional, Tuple, Dict
from .config import SimConfig
from .market import IIDNormalMarket, BootstrapMarket
from .mortality import MortalitySampler

class RetirementEnv:
    """
    Retirement decumulation environment.

    Update order:
      1) clipping (constraints)
      2) consumption: c_t = q_t * W_t
      3) returns: W_net * [1 + w_t*R_risk + (1-w_t)*R_safe]
      4) fee: - phi_m * W_t   (ad-valorem on beginning-of-step wealth)
      5) mortality: Bernoulli(p_death(age_t)) -> if death, done & record

    Notes:
      - Floor(level) 적용 시 q_min = min(1, f_min_real / W_t)
      - w ∈ [0, w_max]
      - Market: IIDNormal or Bootstrap (real returns)
      - Hedge(MVP):
          * mode="mu": per-period μ haircut (R_t ← R_t - cost_m)
          * mode="sigma": per-period σ reduction (R_t ← μ_m + (1-k)·(R_t - μ_m))
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.m = cfg.monthly()  # {'mu_m','sigma_m','rf_m','phi_m','p_m','g_m','beta_m'}
        self.T = cfg.horizon_years * cfg.steps_per_year

        # Market
        mode = str(getattr(cfg, "market_mode", "iid"))
        if mode == "bootstrap":
            self.market = BootstrapMarket(cfg)
        else:
            self.market = IIDNormalMarket(cfg)

        # Hedge params (monthly)
        self._hedge_on: bool = bool(getattr(cfg, "hedge_on", False))
        self._hedge_mode: str = str(getattr(cfg, "hedge_mode", "mu"))
        hedge_cost_annual = float(getattr(cfg, "hedge_cost", 0.0) or 0.0)
        steps = max(int(cfg.steps_per_year), 1)
        self._hedge_cost_m: float = (1.0 + hedge_cost_annual) ** (1.0 / steps) - 1.0
        k = float(getattr(cfg, "hedge_sigma_k", 0.0) or 0.0)
        self._hedge_sigma_k: float = float(min(max(k, 0.0), 1.0))
        self._mu_m: float = float(self.m["mu_m"])

        # Mortality
        self.mortality_on = bool(getattr(cfg, "mortality_on", False))
        self.age0 = int(getattr(cfg, "age0", 65))
        self.sex  = str(getattr(cfg, "sex", "M"))
        self.mort = None
        if self.mortality_on:
            self.mort = MortalitySampler(getattr(cfg, "mort_table", None))

        # Bequest (optional)
        self.bequest_kappa = float(getattr(cfg, "bequest_kappa", 0.0) or 0.0)
        self.bequest_gamma = float(getattr(cfg, "bequest_gamma", 1.0) or 1.0)

        self.reset()

    # internal: hedge transform
    def _apply_hedge(self, r_path: _np.ndarray) -> _np.ndarray:
        if not self._hedge_on:
            return r_path
        if self._hedge_mode == "mu":
            return r_path - self._hedge_cost_m
        else:
            return self._mu_m + (1.0 - self._hedge_sigma_k) * (r_path - self._mu_m)

    def reset(self, seed: Optional[int] = None, W0: float = 1.0) -> Dict:
        if seed is not None:
            self.market.seed(seed)
        self.t = 0
        self.W = float(W0)
        self.W0 = float(W0)
        self.peakW = float(W0)
        # age in years (float), increase by 1/12 per step
        self.age = float(self.age0)

        # Pre-sample returns
        if isinstance(self.market, BootstrapMarket):
            risky, safe = self.market.sample_paths(self.T, block=getattr(self.cfg, "bootstrap_block", 24))
            self.path_risky = self._apply_hedge(_np.asarray(risky, dtype=float))
            self.path_safe  = _np.asarray(safe, dtype=float)
        else:
            base_path = self.market.sample_risky(self.T)
            self.path_risky = self._apply_hedge(_np.asarray(base_path, dtype=float))
            self.path_safe  = _np.full(self.T, float(self.m["rf_m"]), dtype=float)

        return self._state()

    def _state(self) -> Dict:
        # 관측: [t, W, age_norm, 0] 형태로 제공하는 쪽에서 변환 (_GymShim)
        return dict(t=self.t, W=self.W, W0=self.W0, peakW=self.peakW, age=self.age)

    def step(self, q: float, w: float) -> Tuple[Dict, float, bool, bool, Dict]:
        info: Dict = {}

        # 1) clipping
        w = float(_np.clip(float(w), 0.0, self.cfg.w_max))
        q = float(_np.clip(float(q), 0.0, 1.0))
        if getattr(self.cfg, "floor_on", False) and self.cfg.f_min_real > 0.0 and self.W > 0.0:
            q_min = min(1.0, self.cfg.f_min_real / self.W)
            q = max(q, q_min)

        # 2) consumption
        c_t = q * self.W
        W_net = self.W - c_t

        # 3) returns (risky + safe)
        R_risk = float(self.path_risky[self.t])
        R_safe = float(self.path_safe[self.t])
        gross = 1.0 + (w * R_risk) + ((1.0 - w) * R_safe)
        W_next = W_net * gross

        # 4) fee on beginning-of-step wealth
        W_next = W_next - float(self.m["phi_m"]) * self.W

        # clip / bookkeeping
        if hasattr(self.cfg, "hjb_W_max"):
            W_next = float(_np.clip(W_next, 0.0, self.cfg.hjb_W_max))
        else:
            W_next = float(max(W_next, 0.0))

        self.t += 1
        early_ruin = bool(W_next <= 0.0 and self.t < self.T)

        self.W = W_next
        self.peakW = max(self.peakW, self.W)

        # 5) mortality
        died = False
        bequest_val = 0.0
        if self.mortality_on and (self.W > 0.0):
            p = float(self.mort.p_death_month(self.age))
            # Bernoulli draw using numpy
            if _np.random.rand() < p:
                died = True
                # simple bequest utility (if used in reward shaping)
                kappa = self.bequest_kappa
                gamma = self.bequest_gamma
                if kappa > 0.0:
                    if abs(gamma - 1.0) < 1e-9:
                        bequest_val = kappa * _np.log(max(self.W, 1e-12))
                    else:
                        bequest_val = kappa * ((max(self.W, 0.0) ** (1.0 - gamma) - 1.0) / (1.0 - gamma))
        self.age += 1.0 / float(self.cfg.steps_per_year)

        info["W_T"] = float(self.W)
        if died:
            info["death"] = True
            info["bequest"] = float(bequest_val)

        done = bool(self.t >= self.T or self.W <= 0.0 or died)
        return self._state(), c_t, done, early_ruin, info
