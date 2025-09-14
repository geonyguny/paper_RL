# -*- coding: utf-8 -*-
# project/env/retirement_env.py
from __future__ import annotations
from typing import Tuple, Any, Optional
import numpy as np
import os, math


# ---------- helpers ----------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _crra_u(c: float, gamma: float) -> float:
    c = max(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return math.log(c)
    return (c**(1.0 - gamma) - 1.0) / (1.0 - gamma)

def _load_market_arrays(csv_path: str, use_real_rf: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSV columns: date, risky_nom, tbill_nom, cpi
    - cpi가 '지수(level)'이면 월간률로 변환 (r_t = CPI_t / CPI_{t-1} - 1)
    - cpi가 이미 '월간 인플레율'이면 그대로 사용
    - use_real_rf == 'on' 이면 실질 월간수익률로 변환하여 반환
    """
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')

    risky_nom = np.asarray(data['risky_nom'], dtype=float)
    tbill_nom = np.asarray(data['tbill_nom'], dtype=float)
    cpi_col   = np.asarray(data['cpi'],       dtype=float)

    def _to_monthly_rate(x: np.ndarray) -> np.ndarray:
        """CPI가 지수면 월간률로, 이미 월간률이면 그대로."""
        x = np.asarray(x, dtype=float)

        # 휴리스틱: 값의 스케일이 크면(>5) 지수로 간주.
        # 혹은 절대 중앙값이 0.2보다 크면(= 20% 수준) 지수로 간주.
        is_index_like = (np.nanmax(x) > 5.0) or (np.nanmedian(np.abs(x)) > 0.2)

        if is_index_like and x.size >= 2:
            r = np.empty_like(x, dtype=float)
            r[1:] = x[1:] / x[:-1] - 1.0
            # 첫 값은 두 번째 값으로 보간(또는 0.0)
            r[0] = r[1] if x.size > 1 and np.isfinite(x[1]) else 0.0
            return r
        else:
            # 이미 월간률로 판단
            return x.astype(float)

    cpi_rate = _to_monthly_rate(cpi_col)

    if str(use_real_rf).lower() == 'on':
        risky = (1.0 + risky_nom) / (1.0 + cpi_rate) - 1.0
        safe  = (1.0 + tbill_nom) / (1.0 + cpi_rate) - 1.0
    else:
        risky, safe = risky_nom, tbill_nom

    return risky.astype(float), safe.astype(float)

# ---------- Environment ----------
class RetirementEnv:
    """
    Minimal retirement decumulation env:
      - state: (t_norm, W_t)
      - action: (q, w) in [0,1]^2
      - order: clip -> consume -> returns(+hedge) -> fee -> reward
    Required cfg attrs (with defaults if missing):
      horizon_years(15), steps_per_year(12), W0(1.0),
      w_max(1.0), q_floor(0.0), fee_annual(0.004),
      survive_bonus(0.0), u_scale(0.05), crra_gamma(3.0),
      market_mode('bootstrap'|'iid'), market_csv, bootstrap_block(24),
      use_real_rf('on'|'off'),
      hedge('on'|'off'), hedge_mode('sigma'|'downside'), hedge_sigma_k(0~1), hedge_cost(annual)
    """
    def __init__(self, cfg: Any):
        # --- time / wealth / prefs ---
        self.steps_per_year = int(getattr(cfg, 'steps_per_year', 12))
        self.T = int(getattr(cfg, 'horizon_years', 15) * self.steps_per_year)
        self.W0 = float(getattr(cfg, 'W0', 1.0))
        self.w_max = float(getattr(cfg, 'w_max', 1.0))
        self.q_floor = float(getattr(cfg, 'q_floor', 0.0))
        self.fee_annual = float(getattr(cfg, 'fee_annual', 0.004))
        self.fee_m = self.fee_annual / max(self.steps_per_year, 1)
        self.survive_bonus = float(getattr(cfg, 'survive_bonus', 0.0))
        self.u_scale = float(getattr(cfg, 'u_scale', 0.05))
        self.gamma = float(getattr(cfg, 'crra_gamma', 3.0))

        # --- market sources ---
        self.market_mode = str(getattr(cfg, 'market_mode', 'bootstrap')).lower()
        self.market_csv = str(getattr(cfg, 'market_csv', ''))
        self.bootstrap_block = int(getattr(cfg, 'bootstrap_block', 24))
        self.use_real_rf = str(getattr(cfg, 'use_real_rf', 'on'))

        # --- hedge params ---
        self.hedge = str(getattr(cfg, "hedge", "off")).lower()              # 'on'|'off'
        self.hedge_mode = str(getattr(cfg, "hedge_mode", "sigma")).lower()  # 'sigma'|'downside'
        self.hedge_sigma_k = float(getattr(cfg, "hedge_sigma_k", 0.50))     # 0~1
        self.hedge_cost = float(getattr(cfg, "hedge_cost", 0.005))          # annual
        self.hedge_cost_m = self.hedge_cost / max(self.steps_per_year, 1)   # monthly

        # --- seeding / path counter ---
        self.seed_base = int(getattr(cfg, "seed", None) or (getattr(cfg, "seeds", [0])[0] if hasattr(cfg, "seeds") else 0))
        self._path_counter = 0  # increments each reset for iid reproducibility

        # --- preload market arrays if bootstrap ---
        if self.market_mode == 'bootstrap' and os.path.exists(self.market_csv):
            self._risky, self._safe = _load_market_arrays(self.market_csv, self.use_real_rf)
        else:
            # iid fallback distribution (realistic-ish monthly)
            rng = np.random.default_rng(7)
            self._risky = rng.normal(0.06/12, 0.18/np.sqrt(12), size=6000)
            self._safe  = np.full(6000, 0.02/12)

        self.reset()

    # ----- market path builders -----
    def _bootstrap_path(self, T:int, rng:np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        N = len(self._risky)
        B = max(1, self.bootstrap_block)
        r = np.empty(T, float); s = np.empty(T, float)
        t = 0; hi = max(1, N - B + 1)
        while t < T:
            start = rng.integers(0, hi)
            take = min(B, T - t)
            r[t:t+take] = self._risky[start:start+take]
            s[t:t+take] = self._safe[start:start+take]
            t += take
        return r, s

    # ----- API -----
    def reset(self, W0: Optional[float] = None, seed: Optional[int] = None):
        """Supports reset(W0=...), reset(seed=...), reset(W0=..., seed=...)."""
        self.t = 0
        self.W = float(self.W0 if W0 is None else W0)

        # When seed is provided, use it directly to ensure reproducibility across eval paths.
        # Otherwise, advance a deterministic stream based on seed_base + path_counter.
        rng_seed = int(seed) if (seed is not None) else (self.seed_base + self._path_counter)
        rng = np.random.default_rng(rng_seed)
        self._path_counter += 1

        if self.market_mode == 'bootstrap':
            self.path_risky, self.path_safe = self._bootstrap_path(self.T, rng)
        else:
            self.path_risky = rng.normal(0.06/12, 0.18/np.sqrt(12), size=self.T)
            self.path_safe  = np.full(self.T, 0.02/12)
        return self._obs()

    def _obs(self) -> np.ndarray:
        # normalized time plus wealth
        return np.array([self.t / max(1, self.T - 1), self.W], dtype=float)

    def _state(self) -> np.ndarray:
        # backward-compat shim for older code
        return self._obs()

    def step(self, *args, **kwargs):
        """
        Supports:
        - step(q=..., w=...)
        - step(q, w)
        - step([q, w]) / step((q, w)) / step(np.array([q, w]))
        Returns: (obs, reward, done, trunc, info)  # gymnasium-style 5-tuple

        변경점:
        - 헤지 비용은 hedge_active=True 인 달에만 1회 차감 (이중 차감 제거)
        - 모든 중간 변수(k, hedge_active, r_risky_eff) 항상 초기화
        - 안전한 클리핑 및 에피소드 종료 가드
        """
        # ---- parse (q, w) ----
        if len(args) == 1 and not kwargs:
            act = args[0]
            try:
                q = float(act[0]); w = float(act[1])
            except Exception as e:
                raise TypeError("step(action) expects sequence-like [q,w]") from e
        elif len(args) >= 2:
            q = float(args[0]); w = float(args[1])
        else:
            if 'q' in kwargs and 'w' in kwargs:
                q = float(kwargs['q']); w = float(kwargs['w'])
            else:
                raise TypeError("step requires (q, w) or action=[q,w]")

        # ---- guard: episode already ended ----
        if self.t >= self.T:
            return self._obs(), 0.0, True, False, {}

        # 1) clip action
        q = max(float(getattr(self, "q_floor", 0.0) or 0.0), _clip01(q))
        w = _clip01(min(w, float(getattr(self, "w_max", 1.0))))

        # 2) consumption
        c = q * self.W
        W_after_c = max(self.W - c, 0.0)

        # 3) returns (with optional hedge)
        r_risky = float(self.path_risky[self.t])
        r_safe  = float(self.path_safe[self.t])

        # hedge defaults
        k = 0.0
        hedge_active = False
        r_risky_eff = r_risky

        if str(getattr(self, "hedge", "off")).lower() == "on":
            # strength in [0,1]
            k = float(max(0.0, min(1.0, float(getattr(self, "hedge_sigma_k", 0.0)))))
            mode = str(getattr(self, "hedge_mode", "sigma")).lower()

            if mode == "sigma":
                # 항상 안전자산과 혼합해 변동성 완화
                r_risky_eff = (1.0 - k) * r_risky + k * r_safe
                hedge_active = True
            elif mode == "downside":
                # 하락 구간에서만 완화
                if r_risky < 0.0:
                    r_risky_eff = (1.0 - k) * r_risky + k * r_safe
                    hedge_active = True
            # elif mode == "mu":  # 필요 시 추가
            #     ...

        gross = 1.0 + (w * r_risky_eff + (1.0 - w) * r_safe)
        W_after_ret = W_after_c * gross

        # --- 헤지 비용: '해당 스텝에 실제 헤지 동작'이 있었을 때만 1회 차감 ---
        hedge_cost_m = float(getattr(self, "hedge_cost_m", 0.0))
        if hedge_active and hedge_cost_m > 0.0:
            # 헤지 강도/헤지된 위험노출에 비례해 차감(여기선 w를 프록시로 사용)
            W_after_ret *= (1.0 - hedge_cost_m * w)

        # 4) 운용 수수료
        fee_m = float(getattr(self, "fee_m", 0.0))
        fee = fee_m * W_after_ret
        self.W = max(W_after_ret - fee, 0.0)

        # 보상(효용 + 생존 보너스)
        u_scale = float(getattr(self, "u_scale", 0.0))
        gamma = float(getattr(self, "crra_gamma", 3.0))
        survive_bonus = float(getattr(self, "survive_bonus", 0.0))
        reward = u_scale * _crra_u(c, gamma) + survive_bonus

        # advance time & termination
        self.t += 1
        done = (self.t >= self.T) or (self.W <= 0.0)
        trunc = False

        # info for diagnostics
        info = {
            "consumption": float(c),
            "W": float(self.W),
            "q": float(q),
            "w": float(w),
            "r_risky": float(r_risky),
            "r_risky_eff": float(r_risky_eff),
            "r_safe": float(r_safe),
            "hedge": str(getattr(self, "hedge", "off")).lower(),
            "hedge_mode": str(getattr(self, "hedge_mode", "sigma")).lower(),
            "hedge_active": bool(hedge_active),
            "hedge_k": float(k),
        }
        return self._obs(), float(reward), bool(done), bool(trunc), info
