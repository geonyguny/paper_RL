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
    CSV columns (required): date, risky_nom, tbill_nom, cpi
    - cpi가 '지수(level)'이면 월간률로 변환 (r_t = CPI_t / CPI_{t-1} - 1)
    - cpi가 이미 '월간 인플레율'이면 그대로 사용
    - use_real_rf == 'on' 이면 실질 월간수익률로 변환하여 반환
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        names = {n.lower() for n in data.dtype.names or ()}
        required = {'risky_nom', 'tbill_nom', 'cpi'}
        if not required.issubset(names):
            raise ValueError(f"CSV missing columns: {sorted(required - names)}")

        risky_nom = np.asarray(data['risky_nom'], dtype=float)
        tbill_nom = np.asarray(data['tbill_nom'], dtype=float)
        cpi_col   = np.asarray(data['cpi'],       dtype=float)

        def _to_monthly_rate(x: np.ndarray) -> np.ndarray:
            # CPI가 지수면 월간률로, 이미 월간률이면 그대로.
            x = np.asarray(x, dtype=float)
            is_index_like = (np.nanmax(x) > 5.0) or (np.nanmedian(np.abs(x)) > 0.2)
            if is_index_like and x.size >= 2:
                r = np.empty_like(x, dtype=float)
                r[1:] = x[1:] / x[:-1] - 1.0
                r[0] = r[1] if x.size > 1 and np.isfinite(x[1]) else 0.0
                return r
            return x.astype(float)

        cpi_rate = _to_monthly_rate(np.nan_to_num(cpi_col, nan=0.0))

        if str(use_real_rf).lower() == 'on':
            risky = (1.0 + np.nan_to_num(risky_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
            safe  = (1.0 + np.nan_to_num(tbill_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
        else:
            risky = np.nan_to_num(risky_nom, nan=0.0)
            safe  = np.nan_to_num(tbill_nom, nan=0.0)

        return risky.astype(float), safe.astype(float)
    except Exception:
        # 안전한 최종 fallback (parametric i.i.d.)
        rng = np.random.default_rng(7)
        risky = rng.normal(0.06/12, 0.18/np.sqrt(12), size=6000)
        safe  = np.full(6000, 0.02/12)
        return risky, safe


# ---------- Environment ----------
class RetirementEnv:
    """
    Minimal retirement decumulation env:
      - state: (t_norm, W_t)
      - action: (q, w) in [0,1]^2
      - order: clip -> consume -> returns(+hedge) -> fee -> reward

    Notes
    - __init__는 cfg 객체 **또는** 키워드 인자(**kwargs)** 둘 다 지원.
    - step()은 Gymnasium 스타일 **5-튜플** 반환: (obs, reward, done, trunc, info).
    - 헤지 비용은 '헤지 발동(hedge_active=True)'인 스텝에만 1회 차감.
    """

    # --- cfg/kwargs 통합 접근자 ---
    @staticmethod
    def _get(cfg: Any, kwargs: dict, name: str, default: Any) -> Any:
        if kwargs and (name in kwargs):
            return kwargs[name]
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
        return default

    def __init__(self, cfg: Any = None, **kwargs):
        # --- time / wealth / prefs ---
        self.steps_per_year = int(max(1, self._get(cfg, kwargs, 'steps_per_year', 12)))
        self.T = int(max(1, self._get(cfg, kwargs, 'horizon_years', 15))) * self.steps_per_year
        self.W0 = float(self._get(cfg, kwargs, 'W0', 1.0))
        self.w_max = float(self._get(cfg, kwargs, 'w_max', 1.0))
        self.q_floor = float(self._get(cfg, kwargs, 'q_floor', 0.0))
        self.fee_annual = float(self._get(cfg, kwargs, 'fee_annual', 0.004))
        self.fee_m = self.fee_annual / self.steps_per_year  # 월 단순 비례
        self.survive_bonus = float(self._get(cfg, kwargs, 'survive_bonus', 0.0))
        self.u_scale = float(self._get(cfg, kwargs, 'u_scale', 0.05))
        self.gamma = float(self._get(cfg, kwargs, 'crra_gamma', 3.0))

        # --- market sources ---
        self.market_mode = str(self._get(cfg, kwargs, 'market_mode', 'bootstrap') or 'bootstrap').lower()
        self.market_csv = str(self._get(cfg, kwargs, 'market_csv', '') or '')
        self.bootstrap_block = int(max(1, self._get(cfg, kwargs, 'bootstrap_block', 24) or 24))
        self.use_real_rf = str(self._get(cfg, kwargs, 'use_real_rf', 'on') or 'on').lower()

        # --- hedge params ---
        self.hedge = str(self._get(cfg, kwargs, "hedge", "off") or "off").lower()               # 'on'|'off'
        self.hedge_mode = str(self._get(cfg, kwargs, "hedge_mode", "sigma") or "sigma").lower() # 'sigma'|'downside'
        self.hedge_sigma_k = float(self._get(cfg, kwargs, "hedge_sigma_k", 0.50))
        self.hedge_sigma_k = float(max(0.0, min(1.0, self.hedge_sigma_k)))

        # 비용 정책: '발동 시에만' 차감되는 월 프리미엄(=activated cost).
        premium_annual = self._get(cfg, kwargs, "hedge_premium", None)
        if premium_annual is None:
            premium_annual = self._get(cfg, kwargs, "hedge_cost", 0.005)  # backward-compat alias
        self.hedge_premium_annual = float(max(0.0, premium_annual))
        self.hedge_premium_m = self.hedge_premium_annual / self.steps_per_year
        # alias (구코드 호환)
        self.hedge_cost = self.hedge_premium_annual
        self.hedge_cost_m = self.hedge_premium_m

        # (선택) 발동 시 추가 수수료(거래/슬리피지 등)
        self.hedge_tx_annual = float(max(0.0, self._get(cfg, kwargs, "hedge_tx", 0.0)))
        self.hedge_tx_m = self.hedge_tx_annual / self.steps_per_year

        # --- seeding / path counter ---
        seed_attr = self._get(cfg, kwargs, "seed", None)
        if seed_attr is not None:
            self.seed_base = int(seed_attr)
        else:
            seeds = self._get(cfg, kwargs, "seeds", [0]) or [0]
            self.seed_base = int(seeds[0] if len(seeds) > 0 else 0)
        self._path_counter = 0  # increments each reset for iid reproducibility

        # --- preload market arrays if bootstrap ---
        if self.market_mode == 'bootstrap' and os.path.exists(self.market_csv):
            self._risky, self._safe = _load_market_arrays(self.market_csv, self.use_real_rf)
        else:
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

        # seed 지정 시 그대로 사용, 아니면 seed_base+path_counter 진행
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

        Returns (gymnasium-style 5-tuple):
          (obs, reward, done, trunc, info)

        변경점:
        - 헤지 비용은 hedge_active=True 인 달에만 1회 차감 (이중 차감 방지)
        - downside: 상승 미개입 / 하락만 (1-k)배로 완화 / 손실을 양수로 뒤집지 않음
        - sigma: 위험자산-안전자산 convex mix(상시 발동)
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
        r_risky_raw = float(self.path_risky[self.t])
        r_safe      = float(self.path_safe[self.t])
        r_pos = max(r_risky_raw, 0.0)
        r_neg = min(r_risky_raw, 0.0)

        # hedge defaults
        k = 0.0
        hedge_active = False
        r_risky_eff = r_risky_raw

        if str(getattr(self, "hedge", "off")).lower() == "on":
            k = float(max(0.0, min(1.0, float(getattr(self, "hedge_sigma_k", 0.0)))))
            mode = str(getattr(self, "hedge_mode", "sigma")).lower()

            if mode == "sigma":
                # 상시 완화
                r_risky_eff = (1.0 - k) * r_risky_raw + k * r_safe
                hedge_active = True

            elif mode in ("downside", "down"):
                # 하락 구간만 완화: r_eff = r_pos + (1-k) * r_neg
                if r_risky_raw < 0.0 and k > 0.0:
                    r_risky_eff = r_pos + (1.0 - k) * r_neg
                    hedge_active = True
                else:
                    r_risky_eff = r_risky_raw

            # 필요시 추가 모드 확장...

        # 불변조건 강제: 상승 이득 금지 / 손실 뒤집기 금지
        r_risky_eff = max(r_neg, min(r_risky_eff, r_pos))

        # 포트폴리오 월수익률 (헤지 비용 제외)
        r_port = w * r_risky_eff + (1.0 - w) * r_safe

        # --- 헤지 비용: '실제 헤지 동작'이 있었을 때만 1회 차감 ---
        hedge_cost_m = float(getattr(self, "hedge_cost_m", 0.0))
        if hedge_active and hedge_cost_m > 0.0:
            # 헤지된 위험노출(w)에 비례한 drag (원하면 w*k로 변경 가능)
            r_port -= w * hedge_cost_m
            if getattr(self, "hedge_tx_m", 0.0) > 0.0:
                r_port -= w * float(getattr(self, "hedge_tx_m"))

        gross = 1.0 + r_port
        W_after_ret = W_after_c * gross

        # 4) 운용 수수료 (소비→수익률→fee 순서 유지)
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

        # 진단 플래그(버그 탐지용)
        flip_neg_to_pos = (r_risky_raw < 0.0 and r_risky_eff >= 0.0)
        up_drift = (r_risky_raw >= 0.0 and r_risky_eff > r_risky_raw)

        # info for diagnostics
        info = {
            "consumption": float(c),
            "W": float(self.W),
            "q": float(q),
            "w": float(w),
            "r_risky": float(r_risky_raw),
            "r_risky_eff": float(r_risky_eff),
            "r_safe": float(r_safe),
            "hedge": str(getattr(self, "hedge", "off")).lower(),
            "hedge_mode": str(getattr(self, "hedge_mode", "sigma")).lower(),
            "hedge_active": bool(hedge_active),
            "hedge_k": float(k),
            # 버그 감시용 지표
            "FlipNegToPos": bool(flip_neg_to_pos),
            "UpDriftRate": bool(up_drift),
        }
        return self._obs(), float(reward), bool(done), False, info
