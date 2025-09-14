import numpy as _np
from .config import SimConfig

class HJBSolver:
    """
    Backward DP on (t, W) with discrete controls (q, w).

    - q-grid: 5 points [0, 0.25*q4, 0.5*q4, 0.75*q4, q4] (dev: 빠르고 보수적)
    - Terminal CVaR (RU-dual): V_T(W) = - lambda * [eta + (1/(1-alpha)) * max(F - W - eta, 0)]
      (lambda_term<=0이면 eta 탐색 생략; eta=0 단일값)
    - (dev) stage tail penalty: 반드시 lambda_term로 스케일
    - (dev) w^2 penalty: 반드시 lambda_term로 스케일
    - (dev) w_min_dev: 너무 낮은 risky 비중 임시 제외(정책 분산 확보)
    - (dev) dev_split_w_grid: λ=0/λ>0에서 서로 다른 w-grid 사용(정책 분리 가시화)
    - Hedge: cfg.hedge_on일 때 mu/sigma를 본 Solver에도 동일 반영(mu haircut or sigma shrink)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.m = cfg.monthly()
        self.W_grid = _np.linspace(cfg.hjb_W_min, cfg.hjb_W_max, cfg.hjb_W_grid)
        self.T = cfg.horizon_years * cfg.steps_per_year

        # ---------- w-actions (중복 제거 + dev 스위치 반영) ----------
        base_grid = [w for w in cfg.hjb_w_grid if 0.0 <= w <= cfg.w_max]
        if getattr(cfg, "dev_split_w_grid", False):
            # λ에 따라 서로 다른 w-grid로 정책 분리 가시화 (개발용)
            if cfg.lambda_term > 0.0:
                wa = [w for w in base_grid if w <= 0.5]
                if not wa:  # safety
                    wa = [min(max(base_grid[0], 0.0), cfg.w_max)]
            else:
                wa = [w for w in base_grid if w >= 0.5]
                if not wa:
                    wa = [min(max(base_grid[-1], 0.0), cfg.w_max)]
        else:
            # 기본: w_min_dev 이상만 사용(개발용 분산 확보). 본실험 전 0으로 복구 권장.
            wmin = float(getattr(cfg, "w_min_dev", 0.0))
            wa = [w for w in base_grid if w >= wmin]
            if not wa:
                wa = [min(max(base_grid[-1], 0.0), cfg.w_max)]
        self.w_actions = _np.array(sorted(set(wa)), dtype=float)

        # ---------- q-actions (정확히 5점) ----------
        q4 = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
        self.q_actions = _np.array([0.0, 0.25*q4, 0.5*q4, 0.75*q4, q4], dtype=float)

    # --- Utility (CRRA γ=3) ---
    def util(self, c: _np.ndarray) -> _np.ndarray:
        gamma = 3.0
        eps = 1e-9
        if gamma == 1.0:
            return _np.log(_np.maximum(c, eps))
        return (_np.maximum(c, eps) ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    # --- RU-dual terminal penalty helper ---
    def _cvar_terminal_penalty(self, W: _np.ndarray, F: float, alpha: float, eta: float, lam: float) -> _np.ndarray:
        if lam <= 0.0:
            return _np.zeros_like(W, dtype=float)
        inv = 1.0 / max(1e-12, (1.0 - alpha))
        return lam * (eta + inv * _np.maximum(F - W - eta, 0.0))

    def solve(self, seed: int = 0):
        rng = _np.random.default_rng(seed)

        # 월간 파라미터 (키명 정합성: sigma_m 사용)
        mu = float(self.m.get("mu_m", 0.0))
        rf = float(self.m.get("rf_m", 0.0))
        phi = float(self.m.get("phi_m", 0.0))
        beta = float(self.m.get("beta_m", 1.0))
        sigma = float(self.m.get("sigma_m", self.cfg.sigma_annual / (self.cfg.steps_per_year ** 0.5)))

        # --- Hedge를 Solver에도 반영(Env와 동일 규칙) ---
        if getattr(self.cfg, "hedge_on", False):
            mode = getattr(self.cfg, "hedge_mode", "mu")
            if mode == "mu":
                # 수익률 haircut
                mu = mu - float(getattr(self.cfg, "hedge_cost", 0.0))
            elif mode == "sigma":
                k = float(getattr(self.cfg, "hedge_sigma_k", 0.0))
                sigma = max(0.0, sigma * (1.0 - k))

        # eta-grid: lambda_term<=0이면 탐색 생략
        eta_values = self.cfg.hjb_eta_grid if (self.cfg.lambda_term > 0.0) else (0.0,)

        # 기대값 근사 샘플
        Nshock = int(getattr(self.cfg, "hjb_Nshock", 32))
        shocks = rng.normal(loc=mu, scale=sigma, size=Nshock)  # risky per-period returns

        best_eta, best_obj = 0.0, -1e18
        best_policy_w = None
        best_policy_q = None

        # tail 비율을 α로 연동 (예: α=0.95 → 하위 5%)
        tail_ratio = max(1e-9, 1.0 - float(self.cfg.alpha))
        k_tail = max(1, int(tail_ratio * shocks.size))

        # === tie-break tolerance ===
        tie_eps = 1e-9

        for eta in eta_values:
            V = _np.zeros((self.T + 1, self.W_grid.size))
            Pi_w = _np.zeros((self.T, self.W_grid.size))
            Pi_q = _np.zeros((self.T, self.W_grid.size))

            # ---------- Terminal value with RU-dual penalty (핵심 패치) ----------
            F = float(self.cfg.F_target or 1.0)
            pen_T = self._cvar_terminal_penalty(self.W_grid, F, float(self.cfg.alpha), float(eta), float(self.cfg.lambda_term))
            V[self.T, :] = -pen_T  # terminal boundary에 음수 보상으로 반영

            # ---------------------- Backward induction ----------------------
            for t in reversed(range(self.T)):
                for i, W in enumerate(self.W_grid):
                    # Floor-aware q_min
                    q_min = 0.0
                    if self.cfg.floor_on and self.cfg.f_min_real > 0 and W > 0:
                        q_min = min(1.0, self.cfg.f_min_real / W)
                    q_grid = _np.maximum(self.q_actions, q_min)

                    best_val = -1e18
                    bw, bq = float(self.w_actions[0]), float(q_grid[0])

                    for w in self.w_actions:
                        w = float(min(max(w, 0.0), self.cfg.w_max))
                        for q in q_grid:
                            # -------- Transition (순서: clip → cons → returns → fee) --------
                            c = q * W
                            W_net = W - c
                            # mix risky/safe (shocks: risky return samples)
                            gross = 1.0 + w * shocks + (1.0 - w) * rf
                            W_next = W_net * gross - phi * W  # fee base=W_t
                            W_next = _np.clip(W_next, 0.0, self.cfg.hjb_W_max)

                            # Interpolate V_{t+1}(W_next) (vectorized)
                            idx = _np.clip(_np.searchsorted(self.W_grid, W_next) - 1, 0, self.W_grid.size - 2)
                            wl = self.W_grid[idx]
                            wr = self.W_grid[idx + 1]
                            wgt = _np.where(wr > wl, (W_next - wl) / (wr - wl), 0.0)
                            Vn = (1.0 - wgt) * V[t + 1, idx] + wgt * V[t + 1, idx + 1]

                            # ---------------- dev-only penalties (λ 스케일 필수) ----------------
                            risk_pen = 0.0
                            if getattr(self.cfg, "dev_cvar_stage", False) and (F > 0.0):
                                tail = _np.partition(W_next, k_tail - 1)[:k_tail]
                                loss_tail = max(F - float(tail.mean()), 0.0)
                                kappa = float(getattr(self.cfg, "dev_cvar_kappa", 0.0))
                                risk_pen = float(self.cfg.lambda_term) * kappa * loss_tail

                            w2_coeff = float(getattr(self.cfg, "dev_w2_penalty", 0.0))
                            w2_pen = float(self.cfg.lambda_term) * w2_coeff * (w ** 2)

                            # Backup
                            val = self.util(c) + beta * float(Vn.mean()) - risk_pen - w2_pen

                            # === tie-break: 값 동률이면 더 보수적 w(작은 w) 선택 ===
                            if (val > best_val + tie_eps) or (abs(val - best_val) <= tie_eps and w < bw):
                                best_val = float(val)
                                bw, bq = float(w), float(q)

                    V[t, i] = best_val
                    Pi_w[t, i] = bw
                    Pi_q[t, i] = bq

            # 초기부(예: W0=1.0) 목적함수로 eta 선택
            j = int(_np.clip(_np.searchsorted(self.W_grid, 1.0), 0, self.W_grid.size - 1))
            obj = float(V[0, j])
            if obj > best_obj:
                best_obj = obj
                best_eta = float(eta)
                best_policy_w = Pi_w
                best_policy_q = Pi_q

        # Fallbacks (should be rare)
        if best_policy_w is None or best_policy_q is None:
            const_w = float(min(max(self.w_actions.mean(), 0.0), self.cfg.w_max))
            const_q = float(self.q_actions[-1])
            best_policy_w = _np.full((self.T, self.W_grid.size), const_w, dtype=float)
            best_policy_q = _np.full((self.T, self.W_grid.size), const_q, dtype=float)

        return dict(Pi_w=best_policy_w, Pi_q=best_policy_q, eta=best_eta, W_grid=self.W_grid)
