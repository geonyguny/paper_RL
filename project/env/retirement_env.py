# -*- coding: utf-8 -*-
# project/env/retirement_env.py
from __future__ import annotations
from typing import Tuple, Any, Optional, Dict

import os
import math
import numpy as np
import pandas as pd

# ---------- helpers ----------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _crra_u(c: float, gamma: float) -> float:
    c = max(float(c), 1e-12)
    if abs(float(gamma) - 1.0) < 1e-12:
        return math.log(c)
    return (c ** (1.0 - float(gamma)) - 1.0) / (1.0 - float(gamma))


def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """지수면 전월대비율로, 이미 월간률이면 그대로."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    # 지수/율 구분 휴리스틱
    is_index_like = (np.nanmax(x) > 5.0) or (np.nanmedian(np.abs(x)) > 0.2)
    if is_index_like and x.size >= 2:
        r = np.empty_like(x, dtype=float)
        r[1:] = x[1:] / x[:-1] - 1.0
        r[0] = r[1] if x.size > 1 and np.isfinite(x[1]) else 0.0
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return r
    return np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def _nan_guard_arr(a: np.ndarray, *, fill: float = 0.0, clip: Tuple[float, float] | None = None) -> np.ndarray:
    """배열 NaN/Inf 정화(+선택적 클리핑)."""
    arr = np.nan_to_num(np.asarray(a, dtype=float), nan=fill, posinf=fill, neginf=fill)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        arr = np.clip(arr, lo, hi)
    # 비정상 값 전체가 들어오면 최소한 0 배열 보장
    if not np.isfinite(arr).all():
        arr = np.zeros_like(arr, dtype=float)
    return arr


def _safe_float(x: Any, default: float = 0.0) -> float:
    """스칼라 NaN/Inf 방호."""
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(default)
    except Exception:
        return float(default)


def _load_market_arrays(csv_path: str, use_real_rf: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CSV columns (required): date, risky_nom, tbill_nom, cpi
    반환: risky, safe, cpi_rate (모두 월간률). use_real_rf='on'이면 CPI로 실질화.
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = {n.lower() for n in (data.dtype.names or ())}
        required = {"risky_nom", "tbill_nom", "cpi"}
        if not required.issubset(names):
            raise ValueError(f"CSV missing columns: {sorted(required - names)}")

        risky_nom = np.asarray(data["risky_nom"], dtype=float)
        tbill_nom = np.asarray(data["tbill_nom"], dtype=float)
        cpi_col = np.asarray(data["cpi"], dtype=float)

        cpi_rate = _to_monthly_rate_like(np.nan_to_num(cpi_col, nan=0.0))

        if str(use_real_rf).lower() == "on":
            risky = (1.0 + np.nan_to_num(risky_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
            safe = (1.0 + np.nan_to_num(tbill_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
        else:
            risky = np.nan_to_num(risky_nom, nan=0.0)
            safe = np.nan_to_num(tbill_nom, nan=0.0)

        return _nan_guard_arr(risky), _nan_guard_arr(safe), _nan_guard_arr(cpi_rate)
    except Exception:
        # 안전한 최종 fallback (parametric i.i.d.) + CPI=0%
        rng = np.random.default_rng(7)
        risky = rng.normal(0.06 / 12, 0.18 / np.sqrt(12), size=6000)
        safe = np.full(6000, 0.02 / 12)
        cpi_rate = np.zeros(6000, dtype=float)
        return risky, safe, cpi_rate


# ---------- Environment ----------
class RetirementEnv:
    """
    Retirement decumulation env (월 리밸런스)
      - state: (t_norm, W_t)  as ndarray([t/(T-1), W])
      - action: (q, w) ∈ [0,1]^2
      - order: clip → consumption → returns(+hedge) → fee

    주입 지원:
      cfg/kwargs에 data_ret_series, data_rf_series, data_cpi 가 있으면 CSV 대신 이를 사용.
      market_mode='bootstrap'일 때 블록부트스트랩, 'iid'면 파라메트릭 IID.

    step() 반환: (obs, reward, done, trunc, info)
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
        self.steps_per_year = int(max(1, self._get(cfg, kwargs, "steps_per_year", 12)))
        self.T = int(max(1, self._get(cfg, kwargs, "horizon_years", 15))) * self.steps_per_year
        self.W0 = _safe_float(self._get(cfg, kwargs, "W0", 1.0), 1.0)
        self.w_max = _safe_float(self._get(cfg, kwargs, "w_max", 1.0), 1.0)

        # None이면 0.0 처리 (핵심 패치)
        _qf = self._get(cfg, kwargs, "q_floor", 0.0)
        self.q_floor = _safe_float(0.0 if _qf is None else _qf, 0.0)

        self.fee_annual = _safe_float(
            self._get(cfg, kwargs, "phi_adval", self._get(cfg, kwargs, "fee_annual", 0.004)), 0.004
        )
        self.fee_m = self.fee_annual / self.steps_per_year
        self.survive_bonus = _safe_float(self._get(cfg, kwargs, "survive_bonus", 0.0), 0.0)
        self.u_scale = _safe_float(self._get(cfg, kwargs, "u_scale", 0.05), 0.05)
        self.gamma = _safe_float(self._get(cfg, kwargs, "crra_gamma", 3.0), 3.0)

        # --- [ANN] annuity overlay params ---
        self.ann_on = str(self._get(cfg, kwargs, "ann_on", "off") or "off").lower()
        self.ann_alpha = _safe_float(self._get(cfg, kwargs, "ann_alpha", 0.0), 0.0)
        self.ann_L = _safe_float(self._get(cfg, kwargs, "ann_L", 0.0), 0.0)
        self.ann_d = int(self._get(cfg, kwargs, "ann_d", 0) or 0)
        self.ann_index = str(self._get(cfg, kwargs, "ann_index", "real") or "real")
        self.y_ann = max(0.0, _safe_float(self._get(cfg, kwargs, "y_ann", 0.0), 0.0))
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

        # --- meta ---
        self.age0 = int(self._get(cfg, kwargs, "age0", 65))
        self.age_years = float(self.age0)

        # --- market sources ---
        self.market_mode = str(self._get(cfg, kwargs, "market_mode", "bootstrap") or "bootstrap").lower()
        self.market_csv = str(self._get(cfg, kwargs, "market_csv", "") or "")
        self.bootstrap_block = int(max(1, self._get(cfg, kwargs, "bootstrap_block", 24)))
        self.use_real_rf = str(self._get(cfg, kwargs, "use_real_rf", "on") or "on").lower()

        # --- hedge params ---
        self.hedge = str(self._get(cfg, kwargs, "hedge", "off") or "off").lower()
        self.hedge_mode = str(self._get(cfg, kwargs, "hedge_mode", "sigma") or "sigma").lower()
        self.hedge_sigma_k = float(np.clip(self._get(cfg, kwargs, "hedge_sigma_k", 0.50), 0.0, 1.0))

        premium_annual = self._get(cfg, kwargs, "hedge_premium", self._get(cfg, kwargs, "hedge_cost", 0.005))
        self.hedge_premium_annual = _safe_float(max(0.0, float(premium_annual)), 0.0)
        self.hedge_premium_m = self.hedge_premium_annual / self.steps_per_year
        self.hedge_cost = self.hedge_premium_annual  # alias
        self.hedge_cost_m = self.hedge_premium_m

        self.hedge_tx_annual = _safe_float(self._get(cfg, kwargs, "hedge_tx", 0.0), 0.0)
        self.hedge_tx_m = self.hedge_tx_annual / self.steps_per_year

        # --- mortality / rf (정책에서 읽음) ---
        self.life_table: Optional[pd.DataFrame] = None
        self.mort_table_df: Optional[pd.DataFrame] = None
        self.r_f_real_annual: Optional[float] = None
        self._init_mortality_if_any(cfg, kwargs)

        # --- seeding / RNG ---
        seeds = self._get(cfg, kwargs, "seeds", [0]) or [0]
        seed_attr = self._get(cfg, kwargs, "seed", None)
        base = int(seed_attr) if seed_attr is not None else int(seeds[0])
        from numpy.random import SeedSequence, default_rng
        self._ss = SeedSequence(base)
        self.rng = default_rng(self._ss)
        self._path_counter = 0  # increments each reset

        # --- preload market arrays / injection support ---
        inj_ret = self._get(cfg, kwargs, "data_ret_series", None)
        inj_rf = self._get(cfg, kwargs, "data_rf_series", None)
        inj_cpi = self._get(cfg, kwargs, "data_cpi", None)

        if inj_ret is not None and inj_rf is not None:
            self._risky = _nan_guard_arr(inj_ret)
            self._safe = _nan_guard_arr(inj_rf)
            self._cpi_rate = _nan_guard_arr(inj_cpi if inj_cpi is not None else np.zeros_like(self._risky))
        elif self.market_mode == "bootstrap" and os.path.exists(self.market_csv):
            self._risky, self._safe, self._cpi_rate = _load_market_arrays(self.market_csv, self.use_real_rf)
        else:
            tmp_rng = np.random.default_rng(7)
            self._risky = tmp_rng.normal(0.06 / 12, 0.18 / np.sqrt(12), size=6000)
            self._safe = np.full(6000, 0.02 / 12)
            self._cpi_rate = np.zeros(6000, dtype=float)

        # ---- yearly flags ----
        self.is_new_year: bool = True
        self.cpi_yoy: float = 0.0

        self.reset()

    # ----- mortality init -----
    def _init_mortality_if_any(self, cfg: Any, kwargs: dict):
        """cfg 설정에 따라 생명표/실질 rf 로드."""
        mortality_on = bool(str(self._get(cfg, kwargs, "mortality", "off")).lower() == "on")
        if mortality_on:
            path = self._get(cfg, kwargs, "mort_table", None)
            if isinstance(path, str) and os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    has_age = "age" in df.columns
                    has_px = any(c in df.columns for c in ["px", "Px"])
                    has_qx = "qx" in df.columns
                    has_mf = all(c in df.columns for c in ["male", "female"])
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

        rf_from_cfg = self._get(cfg, kwargs, "r_f_real_annual", None)
        self.r_f_real_annual = _safe_float(rf_from_cfg, 0.02) if isinstance(rf_from_cfg, (int, float)) else 0.02

    # ----- market path builders -----
    def _bootstrap_path(self, T: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """블록 부트스트랩으로 길이 T의 (risky, safe, cpi) 경로 생성."""
        N = len(self._risky)
        B = max(1, self.bootstrap_block)
        r = np.empty(T, float)
        s = np.empty(T, float)
        p = np.empty(T, float)
        t = 0
        hi = max(1, N - B + 1)
        while t < T:
            start = int(rng.integers(0, hi))
            take = min(B, T - t)
            r[t : t + take] = self._risky[start : start + take]
            s[t : t + take] = self._safe[start : start + take]
            p[t : t + take] = self._cpi_rate[start : start + take]
            t += take
        # 경로 레벨에서도 잔여 NaN 방호
        return _nan_guard_arr(r), _nan_guard_arr(s), _nan_guard_arr(p)

    # ----- annuity init at reset -----
    def _annuity_init_if_any(self):
        """life table + ann_on=on + ann_alpha>0 → t=0 1회 매입 & y_ann 설정."""
        if not (self.ann_on == "on" and self.ann_alpha > 0.0):
            return
        if self.life_table is None or len(self.life_table) == 0:
            return
        try:
            from ..annuity.overlay import AnnuityConfig, init_annuity  # type: ignore
        except Exception:
            return
        try:
            cfg = AnnuityConfig(on=True, alpha=self.ann_alpha, L=self.ann_L, d=self.ann_d, index=self.ann_index)
            W_after, st = init_annuity(self.W, cfg, self.age0, self.life_table)
            self.W = float(W_after)
            self.y_ann = float(st.y_ann)
            self.ann_purchased = bool(st.purchased)
            self.ann_P = float(st.P)
            self.ann_a_factor = float(st.a_factor)
        except Exception:
            # 실패 시 조용히 무시
            pass

    # ----- API -----
    def reset(self, W0: Optional[float] = None, seed: Optional[int] = None):
        """
        Supports reset(W0=...), reset(seed=...), reset(W0=..., seed=...).
        경로 RNG는 SeedSequence.spawn()으로 경로별 분기.
        """
        # 외부 seed가 주어지면 base 갱신
        if seed is not None:
            from numpy.random import SeedSequence, default_rng
            self._ss = SeedSequence(int(seed))
            self.rng = default_rng(self._ss)

        # 새 에피소드용 RNG 분기
        child = self._ss.spawn(1)[0]
        from numpy.random import default_rng
        self.rng = default_rng(child)
        self._path_counter += 1

        # 초기 상태
        self.t = 0
        self.W = _safe_float(self.W0 if W0 is None else W0, self.W0)
        self.age_years = float(self.age0)
        self.is_new_year = True
        self.cpi_yoy = 0.0

        # [ANN] 초기화
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

        # 경로 생성
        if self.market_mode == "bootstrap":
            self.path_risky, self.path_safe, self.path_cpi = self._bootstrap_path(self.T, self.rng)
        else:
            # IID 파라메트릭
            self.path_risky = self.rng.normal(0.06 / 12, 0.18 / np.sqrt(12), size=self.T)
            self.path_safe = np.full(self.T, 0.02 / 12)
            self.path_cpi = np.zeros(self.T, dtype=float)

        # t=0 연금 매입(옵션)
        self._annuity_init_if_any()

        return self._obs()

    def _obs(self) -> np.ndarray:
        """정규화 시간과 현재 자산을 ndarray로 반환."""
        t_norm = (self.t / max(1, self.T - 1)) if self.T > 1 else 0.0
        return np.array([_safe_float(t_norm, 0.0), _safe_float(self.W, 0.0)], dtype=float)

    def _state(self) -> np.ndarray:
        """과거 호환용 shim."""
        return self._obs()

    # ----- hedge -----
    def _apply_hedge(self, r_risky_raw: float, r_safe: float, w: float) -> Tuple[float, bool]:
        """헤지 모드/강도에 따른 r_risky_eff와 hedge_active 플래그."""
        k = float(np.clip(_safe_float(getattr(self, "hedge_sigma_k", 0.0), 0.0), 0.0, 1.0))
        mode = str(getattr(self, "hedge_mode", "sigma")).lower()

        rr = _safe_float(r_risky_raw, 0.0)
        rf = _safe_float(r_safe, 0.0)
        hedge_active = False
        r_risky_eff = rr

        if str(getattr(self, "hedge", "off")).lower() == "on":
            if mode == "sigma":
                r_risky_eff = (1.0 - k) * rr + k * rf
                hedge_active = True
            elif mode in ("downside", "down"):
                if rr < 0.0 and k > 0.0:
                    r_risky_eff = (1.0 - k) * rr  # 하락만 완화
                    hedge_active = True

        # 상승 이득 증폭 금지 / 손실 뒤집기 금지
        if rr < 0:
            r_risky_eff = min(0.0, r_risky_eff)
        else:
            r_risky_eff = min(r_risky_eff, rr)

        return _safe_float(r_risky_eff, 0.0), bool(hedge_active)

    def step(self, *args, **kwargs):
        """
        Supports:
          - step(q=..., w=...)
          - step(q, w)
          - step([q, w]) / step((q, w)) / step(np.array([q, w]))
        Returns: (obs, reward, done, trunc, info)
        변경점 요약:
          - 헤지 비용은 hedge_active=True 스텝에만 1회 차감
          - 매 스텝 후 is_new_year / cpi_yoy 갱신
          - [ANN] 소비식: c = y_ann + q*W (연금은 외부 유입)
          - 모든 입력/중간값 NaN/Inf 방호
        """
        # ---- parse (q, w) ----
        if len(args) == 1 and not kwargs:
            act = args[0]
            try:
                q = float(act[0])
                w = float(act[1])
            except Exception as e:
                raise TypeError("step(action) expects sequence-like [q,w]") from e
        elif len(args) >= 2:
            q = float(args[0])
            w = float(args[1])
        else:
            if "q" in kwargs and "w" in kwargs:
                q = float(kwargs["q"])
                w = float(kwargs["w"])
            else:
                raise TypeError("step requires (q, w) or action=[q,w]")

        q = _safe_float(q, 0.0)
        w = _safe_float(w, 0.0)

        # 에피소드 종료 후 호출 방지
        if self.t >= self.T:
            return self._obs(), 0.0, True, False, {}

        # 1) clip action ----------------------------------------------------
        q = max(_safe_float(getattr(self, "q_floor", 0.0), 0.0), _clip01(q))
        w = _clip01(min(w, _safe_float(getattr(self, "w_max", 1.0), 1.0)))

        # 2) consumption ----------------------------------------------------
        y_ann = _safe_float(getattr(self, "y_ann", 0.0), 0.0)
        W_now = max(_safe_float(self.W, 0.0), 0.0)
        c = _safe_float(y_ann + q * W_now, 0.0)
        W_after_c = max(W_now - q * W_now, 0.0)  # 연금은 외부 유입

        # 3) returns (+hedge) ----------------------------------------------
        r_risky_raw = _safe_float(self.path_risky[self.t], 0.0)
        r_safe = _safe_float(self.path_safe[self.t], 0.0)
        r_risky_eff, hedge_active = self._apply_hedge(r_risky_raw, r_safe, w)
        r_port = _safe_float(w * r_risky_eff + (1.0 - w) * r_safe, 0.0)

        # 헤지 비용/거래비용 (hedge_active일 때만)
        hc = _safe_float(getattr(self, "hedge_cost_m", 0.0), 0.0)
        htx = _safe_float(getattr(self, "hedge_tx_m", 0.0), 0.0)
        if hedge_active and hc > 0.0:
            r_port -= w * hc
            if htx > 0.0:
                r_port -= w * htx

        gross = _safe_float(1.0 + r_port, 1.0)
        W_after_ret = _safe_float(W_after_c * gross, W_after_c)

        # 4) fee ------------------------------------------------------------
        fee_m = _safe_float(getattr(self, "fee_m", 0.0), 0.0)
        fee = _safe_float(fee_m * W_after_ret, 0.0)
        self.W = max(_safe_float(W_after_ret - fee, 0.0), 0.0)

        # 보상(효용 + 생존)
        reward = _safe_float(getattr(self, "u_scale", 0.0), 0.0) * _crra_u(
            c, _safe_float(getattr(self, "crra_gamma", 3.0), 3.0)
        ) + _safe_float(getattr(self, "survive_bonus", 0.0), 0.0)

        # advance time ------------------------------------------------------
        self.t += 1
        spm = int(getattr(self, "steps_per_year", 12) or 12)
        self.age_years = float(self.age0) + (self.t / max(1, spm))
        self.is_new_year = (self.t % spm == 0)

        # CPI YoY
        if self.t >= spm:
            window = _nan_guard_arr(self.path_cpi[self.t - spm : self.t], fill=0.0)
            try:
                self.cpi_yoy = float(np.prod(1.0 + window) - 1.0)
            except Exception:
                self.cpi_yoy = 0.0
        else:
            self.cpi_yoy = 0.0

        done = (self.t >= self.T) or (self.W <= 0.0)

        # 진단 플래그 (정합성 체크)
        flip_neg_to_pos = (r_risky_raw < 0.0 and r_risky_eff >= 0.0)
        up_drift = (r_risky_raw >= 0.0 and r_risky_eff > r_risky_raw)

        info: Dict[str, Any] = {
            "consumption": float(c),
            "y_ann": float(y_ann),
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
            "cpi_yoy": float(_safe_float(self.cpi_yoy, 0.0)),
            "is_new_year": bool(self.is_new_year),
            "life_table": bool(self.life_table is not None),
            "FlipNegToPos": bool(flip_neg_to_pos),
            "UpDriftRate": bool(up_drift),
        }
        return self._obs(), float(reward), bool(done), False, info
