# project/env.py
from __future__ import annotations

import numpy as _np
from numpy.random import Generator as _Generator, default_rng as _np_default_rng, SeedSequence as _SeedSequence
from typing import Optional, Tuple, Dict, Any

from .config import SimConfig
from .market import IIDNormalMarket, BootstrapMarket
from .mortality import MortalitySampler


# ============================================================
# RNG utilities: Global SeedSequence master + factory helpers
# ============================================================
_SS_MASTER: Optional[_SeedSequence] = None
_SS_COUNTER: int = 0  # 재현 가능한 호출 카운터

def _init_ss_master(seed_like: Optional[int]) -> None:
    """최초 1회만 전역 SeedSequence를 초기화. seed_like=None이면 OS 엔트로피."""
    global _SS_MASTER
    if _SS_MASTER is None:
        _SS_MASTER = _SeedSequence(int(seed_like)) if seed_like is not None else _SeedSequence()

def _next_child_ss() -> _SeedSequence:
    """전역 마스터에서 자식 SeedSequence 하나를 뽑아온다(호출마다 상이)."""
    global _SS_MASTER, _SS_COUNTER
    if _SS_MASTER is None:
        _init_ss_master(None)
    _SS_COUNTER += 1
    # master entropy + counter를 섞어서 child 시퀀스 생성
    return _SeedSequence(entropy=(_SS_MASTER.entropy, _SS_COUNTER))  # type: ignore[arg-type]

def _default_rng(seed_like: Optional[object] = None) -> _Generator:
    """
    통일 RNG 팩토리:
    - seed_like is None           → 전역 마스터에서 child 시퀀스로 RNG
    - seed_like is int            → 그 정수 시드로 고정 RNG
    - seed_like is SeedSequence   → 그 SS에서 child를 spawn하여 RNG
    """
    if seed_like is None:
        return _np_default_rng(_next_child_ss())
    if isinstance(seed_like, _SeedSequence):
        return _np_default_rng(seed_like.spawn(1)[0])
    # 정수/정수형태 문자열 등
    return _np_default_rng(int(seed_like))


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

    def __init__(
        self,
        cfg: SimConfig,
        rng: Optional[_Generator] = None,          # env 내부 난수
        market_rng: Optional[_Generator] = None,   # 마켓 백엔드 난수
    ):
        self.cfg = cfg
        self.m = cfg.monthly()  # {'mu_m','sigma_m','rf_m','phi_m','p_m','g_m','beta_m'}
        self.T = int(cfg.horizon_years) * int(cfg.steps_per_year)

        # ----- RNG 배선: base_seed → [env, market] 서브스트림 분리 -----
        base_seed = getattr(cfg, "seed", None)
        if rng is not None or market_rng is not None:
            # 외부에서 직접 주입 시 그대로 사용(없으면 전역 child)
            self._rng = rng or _default_rng()
            self._rng_market = market_rng or _default_rng()
        else:
            # cfg.seed가 있으면 그 값으로 master 초기화 → child 2개 생성
            _init_ss_master(base_seed)
            self._rng = _np_default_rng(_next_child_ss())
            self._rng_market = _np_default_rng(_next_child_ss())

        # Market backend (iid / bootstrap)
        mode = str(getattr(cfg, "market_mode", "iid")).lower()
        if mode == "bootstrap":
            self.market = BootstrapMarket(cfg)  # 시장 쪽에 rng 주입은 아래에서 시도
        else:
            self.market = IIDNormalMarket(cfg)

        # 가능하면 market에 RNG 주입(호환성 고려)
        self._try_wire_market_rng(self._rng_market)

        # Hedge params (monthly)
        self._hedge_on: bool = bool(getattr(cfg, "hedge_on", getattr(cfg, "hedge", "off") == "on"))
        self._hedge_mode: str = str(getattr(cfg, "hedge_mode", "mu"))
        hedge_cost_annual = float(getattr(cfg, "hedge_cost", 0.0) or 0.0)
        steps = max(int(cfg.steps_per_year), 1)
        self._hedge_cost_m: float = (1.0 + hedge_cost_annual) ** (1.0 / steps) - 1.0
        k = float(getattr(cfg, "hedge_sigma_k", 0.0) or 0.0)
        self._hedge_sigma_k: float = float(min(max(k, 0.0), 1.0))
        self._mu_m: float = float(self.m["mu_m"])

        # Mortality
        mort_flag = (str(getattr(cfg, "mortality", "off")).lower() == "on") or bool(getattr(cfg, "mortality_on", False))
        self.mortality_on = bool(mort_flag)
        self.age0 = int(getattr(cfg, "age0", 65))
        self.sex  = str(getattr(cfg, "sex", "M"))
        self.mort = MortalitySampler(getattr(cfg, "mort_table", None)) if self.mortality_on else None

        # Bequest (optional)
        self.bequest_kappa = float(getattr(cfg, "bequest_kappa", 0.0) or 0.0)
        self.bequest_gamma = float(getattr(cfg, "bequest_gamma", 1.0) or 1.0)

        # ----- Data-injected series (optional) -----
        self._series_ret  = getattr(cfg, "data_ret_series", None)
        self._series_rf   = getattr(cfg, "data_rf_series", None)
        self._series_ok   = (
            self._series_ret is not None and self._series_rf is not None
            and len(_np.asarray(self._series_ret)) > 0 and len(_np.asarray(self._series_rf)) > 0
        )
        self._boot_block  = int(getattr(cfg, "bootstrap_block", 24) or 24)

        # Paths
        self.path_risky: _np.ndarray = _np.zeros(self.T, dtype=float)
        self.path_safe:  _np.ndarray = _np.zeros(self.T, dtype=float)

        self.reset()

    # --------------------
    # RNG wiring helpers
    # --------------------
    def _try_wire_market_rng(self, rng: Optional[_Generator]) -> None:
        """가능하면 market 쪽에 RNG를 주입. (호환성: seed()/rng 속성/인자)"""
        if rng is None:
            # cfg.seed만 있는 경우 market이 자체 seed 메서드를 지원하면 그걸로 초기화
            seed = getattr(self.cfg, "seed", None)
            if seed is not None and hasattr(self.market, "seed"):
                try:
                    self.market.seed(int(seed))
                except Exception:
                    pass
            return

        # 우선순위: 속성 할당 → set_rng 메서드 → seed 대체
        try:
            setattr(self.market, "rng", rng)
            return
        except Exception:
            pass
        if hasattr(self.market, "set_rng"):
            try:
                self.market.set_rng(rng)
                return
            except Exception:
                pass
        # 마지막 수단: seed 메서드가 있으면 임의 정수로 설정
        if hasattr(self.market, "seed"):
            try:
                tmp = _default_rng().integers(0, 2**31 - 1, dtype=_np.int64)
                self.market.seed(int(tmp))
            except Exception:
                pass

    def reseed(self, base_seed: Optional[int]) -> None:
        """런 타임에 base_seed로 env/market RNG를 재구성."""
        # master를 base_seed로(또는 OS 엔트로피로) 초기화하고 child 2개 뽑음
        _init_ss_master(base_seed)
        self._rng = _np_default_rng(_next_child_ss())
        self._rng_market = _np_default_rng(_next_child_ss())
        self._try_wire_market_rng(self._rng_market)

    # --------------------
    # internals
    # --------------------
    def _apply_hedge(self, r_path: _np.ndarray) -> _np.ndarray:
        if not self._hedge_on:
            return r_path
        if self._hedge_mode == "mu":
            return r_path - self._hedge_cost_m
        # sigma-mode: shrink deviations around μ_m
        return self._mu_m + (1.0 - self._hedge_sigma_k) * (r_path - self._mu_m)

    def _make_mbb_path(self, rng: _Generator, series: _np.ndarray, T: int, B: int) -> _np.ndarray:
        """Moving-Block Bootstrap로 길이 T 경로 생성."""
        arr = _np.asarray(series, dtype=float)
        n = int(arr.shape[0])
        if n <= 0:
            raise ValueError("empty series for bootstrap")
        B = max(1, int(B))
        out = _np.empty(T, dtype=float)
        filled = 0
        while filled < T:
            i0 = int(rng.integers(0, max(n - B, 1)))
            take = min(B, T - filled)
            out[filled:filled + take] = arr[i0:i0 + take]
            filled += take
        return out

    def _obs(self) -> Dict[str, Any]:
        # 간단 dict 관측치
        return dict(t=self.t, W=self.W, W0=self.W0, peakW=self.peakW, age=self.age)

    # --------------------
    # API
    # --------------------
    def reset(self, seed: Optional[int] = None, W0: float = 1.0) -> Dict[str, Any]:
        # reseed (internal RNG + backend) – seed가 주어지면 그걸로 child 재구성
        if seed is not None:
            self.reseed(int(seed))

        self.t = 0
        self.W = float(W0)
        self.W0 = float(W0)
        self.peakW = float(W0)
        self.age = float(self.age0)

        use_bootstrap = str(getattr(self.cfg, "market_mode", "iid")).lower() == "bootstrap"
        used_injected = False

        # Data-injected 우선
        if use_bootstrap and self._series_ok:
            B = self._boot_block
            risky = self._make_mbb_path(self._rng, self._series_ret, self.T, B)
            safe  = self._make_mbb_path(self._rng, self._series_rf,  self.T, B)
            # sanity: all-non-finite or all-zero → fallback
            if _np.all(~_np.isfinite(risky)) or _np.allclose(risky, 0.0):
                risky = None
            if _np.all(~_np.isfinite(safe)) or _np.allclose(safe, 0.0):
                safe = None
            if risky is not None and safe is not None:
                self.path_risky = self._apply_hedge(_np.asarray(risky, dtype=float))
                self.path_safe  = _np.asarray(safe, dtype=float)
                used_injected = True

        # Backend fallback
        if not used_injected:
            if isinstance(self.market, BootstrapMarket):
                risky, safe = self.market.sample_paths(self.T, block=getattr(self.cfg, "bootstrap_block", 24))
                self.path_risky = self._apply_hedge(_np.asarray(risky, dtype=float))
                self.path_safe  = _np.asarray(safe, dtype=float)
            else:
                base_path = self.market.sample_risky(self.T)
                self.path_risky = self._apply_hedge(_np.asarray(base_path, dtype=float))
                self.path_safe  = _np.full(self.T, float(self.m["rf_m"]), dtype=float)

        # one-line debug (quiet=off)
        if str(getattr(self.cfg, "quiet", "on")).lower() != "on":
            def _cs(arr: _np.ndarray) -> float:
                arr = _np.asarray(arr, dtype=float)
                return float(_np.nanmean(arr[:16])) if arr.size else float("nan")
            print(
                f"[env] injected={used_injected} "
                f"ret_cs={_cs(self.path_risky):.6f} rf_cs={_cs(self.path_safe):.6f} "
                f"T={self.T} B={self._boot_block}"
            )

        return self._obs()

    def step(self, q: float, w: float) -> Tuple[Dict, float, bool, bool, Dict]:
        info: Dict[str, Any] = {}

        # 1) clipping
        w = float(_np.clip(float(w), 0.0, self.cfg.w_max))
        q = float(_np.clip(float(q), 0.0, 1.0))
        if getattr(self.cfg, "floor_on", False) and getattr(self.cfg, "f_min_real", 0.0) > 0.0 and self.W > 0.0:
            q_min = min(1.0, float(self.cfg.f_min_real) / self.W)
            q = max(q, q_min)

        # 2) consumption
        c_t = float(q * self.W)
        W_net = self.W - c_t

        # 3) returns
        idx = min(max(self.t, 0), self.T - 1)
        R_risk = float(self.path_risky[idx])
        R_safe = float(self.path_safe[idx])

        # print first two steps (quiet=off)
        if str(getattr(self.cfg, "quiet", "on")).lower() != "on" and self.t < 2:
            print(f"[env] t={self.t} R_risk={R_risk:.6f} R_safe={R_safe:.6f}")

        gross = 1.0 + (w * R_risk) + ((1.0 - w) * R_safe)
        W_next = W_net * gross

        # 4) fee on beginning-of-step wealth
        W_next = W_next - float(self.m["phi_m"]) * self.W

        # clip / bookkeeping
        hjb_W_max = getattr(self.cfg, "hjb_W_max", None)
        if hjb_W_max is not None:
            W_next = float(_np.clip(W_next, 0.0, float(hjb_W_max)))
        else:
            W_next = float(max(W_next, 0.0))

        self.t += 1
        early_ruin = bool(W_next <= 0.0 and self.t < self.T)

        self.W = W_next
        self.peakW = max(self.peakW, self.W)

        # 5) mortality (internal RNG)
        died = False
        bequest_val = 0.0
        if self.mortality_on and (self.W > 0.0):
            p = float(self.mort.p_death_month(self.age)) if self.mort is not None else 0.0
            if self._rng.random() < max(0.0, min(p, 1.0)):
                died = True
                kappa = self.bequest_kappa
                gamma = self.bequest_gamma
                if kappa > 0.0:
                    if abs(gamma - 1.0) < 1e-9:
                        bequest_val = kappa * _np.log(max(self.W, 1e-12))
                    else:
                        bequest_val = kappa * ((max(self.W, 0.0) ** (1.0 - gamma) - 1.0) / (1.0 - gamma))
        self.age += 1.0 / float(self.cfg.steps_per_year)

        info["consumption"] = c_t
        info["W_T"] = float(self.W)

        # hedge info
        if self._hedge_on:
            info["hedge_active"] = True
            info["hedge_k"] = float(self._hedge_sigma_k if self._hedge_mode == "sigma" else self._hedge_cost_m)
            info["w"] = float(w)

        if died:
            info["death"] = True
            info["bequest"] = float(bequest_val)

        done = bool(self.t >= self.T or self.W <= 0.0 or died)
        return self._obs(), c_t, done, early_ruin, info
