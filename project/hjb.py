# -*- coding: utf-8 -*-
# project/hjb.py
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as _np
from .config import SimConfig


def _make_rng(seed: Optional[int]) -> _np.random.Generator:
    """SeedSequence 기반 default_rng 생성 (seed=None이면 OS 엔트로피)."""
    if seed is None:
        return _np.random.default_rng()
    ss = _np.random.SeedSequence(int(seed))
    return _np.random.default_rng(ss)


class HJBSolver:
    """
    Backward DP on (t, W) with discrete controls (q, w).

    - q-grid: 5 points [0, 0.25*q4, 0.5*q4, 0.75*q4, q4]  (q4: 연 4%를 월로 환산)
    - Terminal CVaR (RU-dual):
        V_T(W) = - λ * [ η + (1/(1-α)) * max(F - W - η, 0) ]
      (lambda_term<=0이면 η 탐색 생략; η=0 고정)
    - (dev) stage tail penalty / w^2 penalty: 반드시 lambda_term로 스케일
    - (dev) w_min_dev: 너무 낮은 risky 비중 제외(정책 분산 확보용)
    - (dev) dev_split_w_grid: λ=0 / λ>0 에 서로 다른 w-grid (정책 분리 가시화)
    - Hedge: cfg.hedge, hedge_mode, hedge_cost/hedge_sigma_k를 Solver에도 동일 반영
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.m = cfg.monthly()

        # Wealth grid
        W_min = float(getattr(cfg, "hjb_W_min", 0.0))
        W_max = float(getattr(cfg, "hjb_W_max", 2.0))
        W_n   = int(getattr(cfg, "hjb_W_grid", 33))
        self.W_grid = _np.linspace(W_min, W_max, W_n)
        self.T = int(cfg.horizon_years) * int(cfg.steps_per_year)

        # ---------- w-actions (중복 제거 + dev 스위치 반영) ----------
        base_grid = list(getattr(cfg, "hjb_w_grid", [0.0, 0.25, 0.5, 0.75, 1.0]))
        base_grid = [float(w) for w in base_grid if 0.0 <= float(w) <= float(cfg.w_max)]
        if not base_grid:
            base_grid = [0.0, min(0.5, float(cfg.w_max)), float(cfg.w_max)]

        if bool(getattr(cfg, "dev_split_w_grid", False)):
            if float(getattr(cfg, "lambda_term", 0.0)) > 0.0:
                wa = [w for w in base_grid if w <= 0.5] or [min(max(base_grid[0], 0.0), float(cfg.w_max))]
            else:
                wa = [w for w in base_grid if w >= 0.5] or [min(max(base_grid[-1], 0.0), float(cfg.w_max))]
        else:
            wmin = float(getattr(cfg, "w_min_dev", 0.0) or 0.0)
            wa = [w for w in base_grid if w >= wmin] or [min(max(base_grid[-1], 0.0), float(cfg.w_max))]

        self.w_actions = _np.array(sorted(set(wa)), dtype=float)

        # ---------- q-actions (정확히 5점) ----------
        # 연 4%를 steps_per_year 기준으로 월 전환해 최대치 근사
        spm = int(getattr(cfg, "steps_per_year", 12))
        q4 = 1.0 - (1.0 - 0.04) ** (1.0 / max(spm, 1))
        self.q_actions = _np.array([0.0, 0.25*q4, 0.5*q4, 0.75*q4, q4], dtype=float)

    # --- Utility (CRRA γ=3; 필요하면 cfg.crra_gamma로 전환 가능) ---
    def util(self, c: _np.ndarray) -> _np.ndarray:
        gamma = 3.0
        eps = 1e-9
        if abs(gamma - 1.0) < 1e-12:
            return _np.log(_np.maximum(c, eps))
        return (_np.maximum(c, eps) ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    # --- RU-dual terminal penalty helper ---
    def _cvar_terminal_penalty(
        self,
        W: _np.ndarray,
        F: float,
        alpha: float,
        eta: float,
        lam: float
    ) -> _np.ndarray:
        if lam <= 0.0:
            return _np.zeros_like(W, dtype=float)
        inv = 1.0 / max(1e-12, (1.0 - alpha))
        return lam * (eta + inv * _np.maximum(F - W - eta, 0.0))

    def solve(self, seed: Optional[int] = None, rng: Optional[_np.random.Generator] = None) -> Dict[str, Any]:
        """
        Parameters
        ----------
        seed : Optional[int]
            None 이면 OS 엔트로피 기반 RNG, 정수면 재현 가능.
        rng  : Optional[np.random.Generator]
            외부에서 RNG를 넘겨주면 그대로 사용.

        Returns
        -------
        dict: { "Pi_w": (T×|W|), "Pi_q": (T×|W|), "eta": float, "W_grid": np.ndarray }
        """
        # --- RNG 초기화(핵심 수정) ---
        rng_local = rng if rng is not None else _make_rng(seed)

        # 월간 파라미터 (키명 정합성: sigma_m 사용)
        mu    = float(self.m.get("mu_m", 0.0))
        rf    = float(self.m.get("rf_m", 0.0))
        phi   = float(self.m.get("phi_m", 0.0))
        beta  = float(self.m.get("beta_m", 1.0))
        # sigma 추정치: 주어지지 않으면 연σ를 월로 환산(보수적 기본값)
        sigma = float(
            self.m.get(
                "sigma_m",
                float(getattr(self.cfg, "sigma_annual", 0.18)) / (float(getattr(self.cfg, "steps_per_year", 12)) ** 0.5)
            )
        )

        # --- Hedge를 Solver에도 반영(Env와 동일 규칙) ---
        if str(getattr(self.cfg, "hedge", getattr(self.cfg, "hedge_on", "off"))).lower() == "on":
            mode = str(getattr(self.cfg, "hedge_mode", "mu")).lower()
            if mode == "mu":
                # 수익률 haircut
                mu = mu - float(getattr(self.cfg, "hedge_cost", 0.0) or 0.0)
            elif mode == "sigma":
                k = float(getattr(self.cfg, "hedge_sigma_k", 0.0) or 0.0)
                sigma = max(0.0, sigma * (1.0 - k))

        # eta-grid: lambda_term<=0이면 탐색 생략
        lam_term = float(getattr(self.cfg, "lambda_term", 0.0) or 0.0)
        eta_values = tuple(getattr(self.cfg, "hjb_eta_grid", (0.0,))) if (lam_term > 0.0) else (0.0,)

        # 기대값 근사 샘플
        Nshock = int(getattr(self.cfg, "hjb_Nshock", 32) or 32)
        shocks = rng_local.normal(loc=mu, scale=sigma, size=Nshock)  # risky per-period return samples

        best_eta: float = 0.0
        best_obj: float = -1e18
        best_policy_w = None
        best_policy_q = None

        # tail 비율: α=0.95 → 하위 5%
        alpha = float(getattr(self.cfg, "alpha", 0.95))
        tail_ratio = max(1e-9, 1.0 - alpha)
        k_tail = max(1, int(round(tail_ratio * shocks.size)))

        # tie-break tolerance
        tie_eps = 1e-9

        F = float(getattr(self.cfg, "F_target", 1.0) or 1.0)

        for eta in eta_values:
            V   = _np.zeros((self.T + 1, self.W_grid.size), dtype=float)
            PiW = _np.zeros((self.T,     self.W_grid.size), dtype=float)
            PiQ = _np.zeros((self.T,     self.W_grid.size), dtype=float)

            # ---------- Terminal value with RU-dual penalty ----------
            pen_T = self._cvar_terminal_penalty(self.W_grid, F, alpha, float(eta), lam_term)
            V[self.T, :] = -pen_T  # terminal boundary를 음수 보상으로 반영

            # ---------------------- Backward induction ----------------------
            for t in reversed(range(self.T)):
                for i, W in enumerate(self.W_grid):
                    # Floor-aware q_min
                    q_min = 0.0
                    if bool(getattr(self.cfg, "floor_on", False)) and float(getattr(self.cfg, "f_min_real", 0.0)) > 0.0 and W > 0.0:
                        q_min = min(1.0, float(getattr(self.cfg, "f_min_real")) / W)
                    q_grid = _np.maximum(self.q_actions, q_min)

                    best_val = -1e18
                    bw = float(self.w_actions[0])
                    bq = float(q_grid[0])

                    for w in self.w_actions:
                        w = float(min(max(w, 0.0), float(self.cfg.w_max)))
                        for q in q_grid:
                            # ---- Transition: clip → consume → returns → fee ----
                            c = q * W
                            W_net = W - c

                            # risky/safe mix (using MC samples `shocks`)
                            gross = 1.0 + (w * shocks) + ((1.0 - w) * rf)
                            W_next = W_net * gross - phi * W  # fee base=W_t
                            W_next = _np.clip(W_next, 0.0, float(getattr(self.cfg, "hjb_W_max", self.W_grid[-1])))

                            # Interpolate V_{t+1}(W_next) (vectorized)
                            idx = _np.clip(_np.searchsorted(self.W_grid, W_next) - 1, 0, self.W_grid.size - 2)
                            wl = self.W_grid[idx]
                            wr = self.W_grid[idx + 1]
                            wgt = _np.where(wr > wl, (W_next - wl) / (wr - wl), 0.0)
                            Vn = (1.0 - wgt) * V[t + 1, idx] + wgt * V[t + 1, idx + 1]

                            # -------- dev-only penalties (λ 스케일 필수) --------
                            risk_pen = 0.0
                            if bool(getattr(self.cfg, "dev_cvar_stage", False)) and (F > 0.0) and (lam_term > 0.0):
                                # 하위 tail 평균이 F보다 낮으면 페널티
                                tail = _np.partition(W_next, k_tail - 1)[:k_tail]
                                loss_tail = max(F - float(tail.mean()), 0.0)
                                kappa = float(getattr(self.cfg, "dev_cvar_kappa", 0.0) or 0.0)
                                risk_pen = lam_term * kappa * loss_tail

                            w2_coeff = float(getattr(self.cfg, "dev_w2_penalty", 0.0) or 0.0)
                            w2_pen = lam_term * w2_coeff * (w ** 2)

                            # Backup
                            val = self.util(_np.array([c]))[0] + beta * float(Vn.mean()) - risk_pen - w2_pen

                            # tie-break: 값 동률이면 더 보수적 w(작은 w) 선택
                            if (val > best_val + tie_eps) or (abs(val - best_val) <= tie_eps and w < bw):
                                best_val = float(val)
                                bw, bq = float(w), float(q)

                    V[t, i]  = best_val
                    PiW[t, i] = bw
                    PiQ[t, i] = bq

            # 초기부(예: W0=1.0) 목적함수로 eta 선택
            j = int(_np.clip(_np.searchsorted(self.W_grid, 1.0), 0, self.W_grid.size - 1))
            obj = float(V[0, j])
            if obj > best_obj:
                best_obj = obj
                best_eta = float(eta)
                best_policy_w = PiW
                best_policy_q = PiQ

        # Fallbacks (should be rare)
        if best_policy_w is None or best_policy_q is None:
            const_w = float(min(max(self.w_actions.mean(), 0.0), float(self.cfg.w_max)))
            const_q = float(self.q_actions[-1])
            best_policy_w = _np.full((self.T, self.W_grid.size), const_w, dtype=float)
            best_policy_q = _np.full((self.T, self.W_grid.size), const_q, dtype=float)

        return {"Pi_w": best_policy_w, "Pi_q": best_policy_q, "eta": best_eta, "W_grid": self.W_grid}
