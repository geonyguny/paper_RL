# -*- coding: utf-8 -*-
# project/env/retirement_env.py
from __future__ import annotations
from typing import Tuple, Any, Optional
import numpy as np
import pandas as pd
import os, math

# ---------- helpers ----------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _crra_u(c: float, gamma: float) -> float:
    c = max(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return math.log(c)
    return (c**(1.0 - gamma) - 1.0) / (1.0 - gamma)

def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """지수로 보이면 전월대비율로 변환, 아니면 그대로(월간률 가정)."""
    x = np.asarray(x, dtype=float)
    is_index_like = (np.nanmax(x) > 5.0) or (np.nanmedian(np.abs(x)) > 0.2)
    if is_index_like and x.size >= 2:
        r = np.empty_like(x, dtype=float)
        r[1:] = x[1:] / x[:-1] - 1.0
        r[0] = r[1] if x.size > 1 and np.isfinite(x[1]) else 0.0
        return r
    return x.astype(float)

def _load_market_arrays(csv_path: str, use_real_rf: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CSV columns (required): date, risky_nom, tbill_nom, cpi

    반환:
      risky    : 월간 위험자산 수익률 (use_real_rf='on'이면 실질, 아니면 명목)
      safe     : 월간 무위험자산 수익률 (동일 규칙)
      cpi_rate : 월간 CPI 상승률(월간률)

    use_real_rf == 'on'이면 risky/safe를 CPI로 실질화.
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        names = {n.lower() for n in (data.dtype.names or ())}
        required = {'risky_nom', 'tbill_nom', 'cpi'}
        if not required.issubset(names):
            raise ValueError(f"CSV missing columns: {sorted(required - names)}")

        risky_nom = np.asarray(data['risky_nom'], dtype=float)
        tbill_nom = np.asarray(data['tbill_nom'], dtype=float)
        cpi_col   = np.asarray(data['cpi'],       dtype=float)

        cpi_rate = _to_monthly_rate_like(np.nan_to_num(cpi_col, nan=0.0))

        if str(use_real_rf).lower() == 'on':
            risky = (1.0 + np.nan_to_num(risky_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
            safe  = (1.0 + np.nan_to_num(tbill_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
        else:
            risky = np.nan_to_num(risky_nom, nan=0.0)
            safe  = np.nan_to_num(tbill_nom, nan=0.0)

        return risky.astype(float), safe.astype(float), cpi_rate.astype(float)
    except Exception:
        # 안전한 최종 fallback (parametric i.i.d.) + CPI=0%
        rng = np.random.default_rng(7)
        risky = rng.normal(0.06/12, 0.18/np.sqrt(12), size=6000)
        safe  = np.full(6000, 0.02/12)
        cpi_rate = np.zeros(6000, dtype=float)
        return risky, safe, cpi_rate


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
    - 연초 플래그/인플레 연율 노출: self.is_new_year, self.cpi_yoy
    - [ANN] 연금 오버레이: t=0에 1회 매입(P 차감) 후, 월 고정지급 self.y_ann을 소비에 **가산**(자산 W에서 차감하지 않음).
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

        # --- [ANN] annuity overlay flags/params ---
        self.ann_on = str(self._get(cfg, kwargs, 'ann_on', 'off') or 'off').lower()
        self.ann_alpha = float(self._get(cfg, kwargs, 'ann_alpha', 0.0) or 0.0)
        self.ann_L = float(self._get(cfg, kwargs, 'ann_L', 0.0) or 0.0)
        self.ann_d = int(self._get(cfg, kwargs, 'ann_d', 0) or 0)
        self.ann_index = str(self._get(cfg, kwargs, 'ann_index', 'real') or 'real')
        self.y_ann = float(max(0.0, self._get(cfg, kwargs, 'y_ann', 0.0)))
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

        # --- demo용 나이 메타 (정책/로깅에서 사용) ---
        self.age0 = int(self._get(cfg, kwargs, 'age0', 65))
        self.age_years = float(self.age0)  # reset/step에서 갱신

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

        premium_annual = self._get(cfg, kwargs, "hedge_premium", None)
        if premium_annual is None:
            premium_annual = self._get(cfg, kwargs, "hedge_cost", 0.005)  # backward-compat alias
        self.hedge_premium_annual = float(max(0.0, premium_annual))
        self.hedge_premium_m = self.hedge_premium_annual / self.steps_per_year
        # alias
        self.hedge_cost = self.hedge_premium_annual
        self.hedge_cost_m = self.hedge_premium_m

        self.hedge_tx_annual = float(max(0.0, self._get(cfg, kwargs, "hedge_tx", 0.0)))
        self.hedge_tx_m = self.hedge_tx_annual / self.steps_per_year

        # --- mortality / rf (정책에서 읽어감) ---
        self.life_table: Optional[pd.DataFrame] = None
        self.mort_table_df: Optional[pd.DataFrame] = None
        self.r_f_real_annual: Optional[float] = None
        self._init_mortality_if_any(cfg, kwargs)

        # --- seeding / path counter ---
        seed_attr = self._get(cfg, kwargs, "seed", None)
        if seed_attr is not None:
            self.seed_base = int(seed_attr)
        else:
            seeds = self._get(cfg, kwargs, "seeds", [0]) or [0]
            self.seed_base = int(seeds[0] if len(seeds) > 0 else 0)
        self._path_counter = 0  # increments each reset for iid reproducibility

        # --- preload market arrays ---
        if self.market_mode == 'bootstrap' and os.path.exists(self.market_csv):
            self._risky, self._safe, self._cpi_rate = _load_market_arrays(self.market_csv, self.use_real_rf)
        else:
            rng = np.random.default_rng(7)
            self._risky = rng.normal(0.06/12, 0.18/np.sqrt(12), size=6000)
            self._safe  = np.full(6000, 0.02/12)
            self._cpi_rate = np.zeros(6000, dtype=float)  # CPI 없으면 0%

        # ---- yearly flags (연초 트리거 & CPI YoY) ----
        self.is_new_year: bool = True
        self.cpi_yoy: float = 0.0  # 최근 12개월 누적 인플레율

        self.reset()

    # ----- mortality init -----
    def _init_mortality_if_any(self, cfg: Any, kwargs: dict):
        """cfg 설정에 따라 생명표/실질 rf를 로드하여 액터가 쓰도록 노출."""
        # ---- life table ----
        mortality_on = bool(str(self._get(cfg, kwargs, "mortality", "off")).lower() == "on")
        if mortality_on:
            path = self._get(cfg, kwargs, "mort_table", None)
            if isinstance(path, str) and os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    has_age = "age" in df.columns
                    has_px  = any(c in df.columns for c in ["px", "Px"])
                    has_qx  = "qx" in df.columns
                    has_mf  = all(c in df.columns for c in ["male", "female"])
                    if not has_age:
                        raise ValueError("mort_table must contain 'age' column")

                    if has_qx:
                        lt = df[["age", "qx"]].copy()
                    elif has_px:
                        col = "px" if "px" in df.columns else "Px"
                        lt = df[["age", col]].rename(columns={col: "px"}).copy()
                    elif has_mf:
                        sex = self._get(cfg, kwargs, "sex", "M")
                        col = "male" if str(sex).upper() == "M" else "female"
                        lt = df[["age", col]].rename(columns={col: "qx"}).copy()
                    else:
                        raise ValueError("expected ('age' + 'qx') or ('age' + 'px/Px') or ('age' + 'male,female')")

                    lt["age"] = lt["age"].astype(int)
                    lt = lt.sort_values("age").reset_index(drop=True)
                    self.life_table = lt
                    self.mort_table_df = lt
                    print(f"[env:mort] life_table loaded: rows={len(lt)}, cols={list(lt.columns)}")
                except Exception as e:
                    print(f"[env:mort] failed to load/parse mort_table: {e}")

        # ---- real rf ----
        rf_from_cfg = self._get(cfg, kwargs, "r_f_real_annual", None)
        if isinstance(rf_from_cfg, (int, float)):
            self.r_f_real_annual = float(rf_from_cfg)
        else:
            self.r_f_real_annual = 0.02  # 프로젝트 기본치

    # ----- market path builders -----
    def _bootstrap_path(self, T:int, rng:np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(self._risky)
        B = max(1, self.bootstrap_block)
        r = np.empty(T, float); s = np.empty(T, float); p = np.empty(T, float)
        t = 0; hi = max(1, N - B + 1)
        while t < T:
            start = rng.integers(0, hi)
            take = min(B, T - t)
            r[t:t+take] = self._risky[start:start+take]
            s[t:t+take] = self._safe[start:start+take]
            p[t:t+take] = self._cpi_rate[start:start+take]
            t += take
        return r, s, p

    # ----- annuity init at reset -----
    def _annuity_init_if_any(self):
        """life table 존재 + ann_on=on + ann_alpha>0 인 경우 t=0 일시 매입을 적용."""
        if not (self.ann_on == "on" and self.ann_alpha > 0.0):
            # 사용자가 y_ann을 cfg로 직접 전달했을 수도 있으니 그대로 둠
            return

        # life table 필요 (MVP)
        if self.life_table is None or len(self.life_table) == 0:
            return

        # overlay 모듈이 없을 수도 있으므로 방어
        try:
            from ..annuity.overlay import AnnuityConfig, init_annuity  # type: ignore
        except Exception:
            return

        try:
            cfg = AnnuityConfig(
                on=True, alpha=self.ann_alpha, L=self.ann_L, d=self.ann_d, index=self.ann_index
            )
            W_after, st = init_annuity(self.W, cfg, self.age0, self.life_table)
            # 적용
            self.W = float(W_after)
            self.y_ann = float(st.y_ann)
            self.ann_purchased = bool(st.purchased)
            self.ann_P = float(st.P)
            self.ann_a_factor = float(st.a_factor)
        except Exception:
            # 실패 시 조용히 무시 (env는 계속 동작)
            pass

    # ----- API -----
    def reset(self, W0: Optional[float] = None, seed: Optional[int] = None):
        """Supports reset(W0=...), reset(seed=...), reset(W0=..., seed=...)."""
        self.t = 0
        self.W = float(self.W0 if W0 is None else W0)
        self.age_years = float(self.age0)  # t=0 기준 나이
        self.is_new_year = True            # 첫 스텝 직전은 연초로 간주
        self.cpi_yoy = 0.0                 # 최근 12개월 누적 CPI (초기 0)

        # ann status 초기화 (y_ann은 cfg에서 넘어온 값 보존)
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

        rng_seed = int(seed) if (seed is not None) else (self.seed_base + self._path_counter)
        rng = np.random.default_rng(rng_seed)
        self._path_counter += 1

        if self.market_mode == 'bootstrap':
            self.path_risky, self.path_safe, self.path_cpi = self._bootstrap_path(self.T, rng)
        else:
            self.path_risky = rng.normal(0.06/12, 0.18/np.sqrt(12), size=self.T)
            self.path_safe  = np.full(self.T, 0.02/12)
            self.path_cpi   = np.zeros(self.T, dtype=float)  # iid 모드 기본 CPI=0%

        # --- [ANN] t=0 한 번 매입 → W 차감 / y_ann 설정
        self._annuity_init_if_any()

        return self._obs()

    def _obs(self) -> np.ndarray:
        # normalized time plus wealth
        return np.array([self.t / max(1, self.T - 1), self.W], dtype=float)

    def _state(self) -> np.ndarray:
        # backward-compat shim for older code
        return self._obs()

    # ----- hedge -----
    def _apply_hedge(self, r_risky_raw: float, r_safe: float, w: float) -> Tuple[float, bool]:
        """헤지 모드/강도에 따른 r_risky_eff와 hedge_active 플래그를 계산."""
        k = float(max(0.0, min(1.0, float(getattr(self, "hedge_sigma_k", 0.0)))))
        mode = str(getattr(self, "hedge_mode", "sigma")).lower()
        r_pos = max(r_risky_raw, 0.0)
        r_neg = min(r_risky_raw, 0.0)

        hedge_active = False
        r_risky_eff = r_risky_raw

        if str(getattr(self, "hedge", "off")).lower() == "on":
            if mode == "sigma":
                r_risky_eff = (1.0 - k) * r_risky_raw + k * r_safe
                hedge_active = True
            elif mode in ("downside", "down"):
                if r_risky_raw < 0.0 and k > 0.0:
                    r_risky_eff = r_pos + (1.0 - k) * r_neg
                    hedge_active = True
                else:
                    r_risky_eff = r_risky_raw

        # 불변조건: 상승 이득 금지 / 손실 뒤집기 금지
        r_risky_eff = max(r_neg, min(r_risky_eff, r_pos))
        return r_risky_eff, hedge_active

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
        - 매 스텝 후 self.is_new_year / self.cpi_yoy 갱신
        - [ANN] 소비식: c = y_ann + q*W (연금은 외부 유입, 자산에서 차감하지 않음)
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

        # 2) consumption  ---------------------------------------------------
        y_ann = float(getattr(self, "y_ann", 0.0) or 0.0)   # [ANN]
        c = y_ann + q * self.W
        # 연금은 외부 유입(보험), 자산 W에서 차감하지 않음.
        W_after_c = max(self.W - q * self.W, 0.0)

        # 3) returns (with optional hedge) ----------------------------------
        r_risky_raw = float(self.path_risky[self.t])
        r_safe      = float(self.path_safe[self.t])

        r_risky_eff, hedge_active = self._apply_hedge(r_risky_raw, r_safe, w)
        r_port = w * r_risky_eff + (1.0 - w) * r_safe

        # --- 헤지 비용: '실제 헤지 동작'이 있었을 때만 1회 차감 ---
        hedge_cost_m = float(getattr(self, "hedge_cost_m", 0.0))
        if hedge_active and hedge_cost_m > 0.0:
            r_port -= w * hedge_cost_m
            if getattr(self, "hedge_tx_m", 0.0) > 0.0:
                r_port -= w * float(getattr(self, "hedge_tx_m"))

        gross = 1.0 + r_port
        W_after_ret = W_after_c * gross

        # 4) 운용 수수료 (소비→수익률→fee 순서 유지) -------------------------
        fee_m = float(getattr(self, "fee_m", 0.0))
        fee = fee_m * W_after_ret
        self.W = max(W_after_ret - fee, 0.0)

        # 보상(효용 + 생존 보너스)
        u_scale = float(getattr(self, "u_scale", 0.0))
        gamma = float(getattr(self, "crra_gamma", 3.0))
        survive_bonus = float(getattr(self, "survive_bonus", 0.0))
        reward = u_scale * _crra_u(c, gamma) + survive_bonus

        # advance time & termination ----------------------------------------
        self.t += 1

        # --- 나이 & 연초 플래그 ---
        spm = int(getattr(self, "steps_per_year", 12) or 12)
        self.age_years = float(self.age0) + (self.t / max(1, spm))
        self.is_new_year = (self.t % spm == 0)

        # --- CPI YoY (최근 12개월 누적) ---
        if self.t >= spm:
            window = self.path_cpi[self.t-spm:self.t]
            try:
                self.cpi_yoy = float(np.prod(1.0 + window) - 1.0)
            except Exception:
                self.cpi_yoy = 0.0
        else:
            self.cpi_yoy = 0.0

        done = (self.t >= self.T) or (self.W <= 0.0)

        # 진단 플래그(버그 탐지용)
        flip_neg_to_pos = (r_risky_raw < 0.0 and r_risky_eff >= 0.0)
        up_drift = (r_risky_raw >= 0.0 and r_risky_eff > r_risky_raw)

        # info for diagnostics
        info = {
            "consumption": float(c),
            "y_ann": float(y_ann),               # [ANN]
            "ann_on": (self.ann_on == "on"),
            "ann_purchased": bool(self.ann_purchased),
            "ann_P": float(self.ann_P),
            "ann_a_factor": float(self.ann_a_factor),
            "W": float(self.W),
            "q": float(q),
            "w": float(w),
            "r_risky": float(r_risky_raw),
            "r_risky_eff": float(r_risky_eff),
            "r_safe": float(r_safe),
            "hedge": str(getattr(self, "hedge", "off")).lower(),
            "hedge_mode": str(getattr(self, "hedge_mode", "sigma")).lower(),
            "hedge_active": bool(hedge_active),
            "hedge_k": float(getattr(self, "hedge_sigma_k", 0.0)),
            # 디버깅 보조
            "cpi_yoy": float(self.cpi_yoy),
            "is_new_year": bool(self.is_new_year),
            "life_table": bool(self.life_table is not None),
            "FlipNegToPos": bool(flip_neg_to_pos),
            "UpDriftRate": bool(up_drift),
        }
        return self._obs(), float(reward), bool(done), False, info
