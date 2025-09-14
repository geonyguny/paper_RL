# project/trainer/rl_a2c.py
import time, csv, math, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from project.env import RetirementEnv

try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except Exception:
    pass

# ---------- Gym-like shim ----------
class _GymShim:
    def __init__(self, base_env: RetirementEnv):
        self.base = base_env

    def _obs_vec(self, s):
        if isinstance(s, dict):
            t  = float(s.get("t", s.get("age", 0.0)))
            W  = float(s.get("W", s.get("W_t", 0.0)))
            T  = max(getattr(self.base, "T", 1), 1)
            W0 = float(getattr(self.base, "W0", 1.0)) or 1.0
            age = float(s.get("age", 0.0))
            age_norm = age / 120.0
            return np.array([t/float(T), W/ W0, age_norm, 0.0], dtype=np.float32)
        return np.asarray(s, dtype=np.float32)

    def reset(self, seed=None):
        try: out = self.base.reset(seed=seed)
        except TypeError: out = self.base.reset()
        s = out[0] if (isinstance(out, tuple) and len(out)>=1) else out
        return self._obs_vec(s), {}

    def step(self, action):
        q = float(action[0]); w = float(action[1])
        out = self.base.step(q, w)
        if not isinstance(out, tuple):
            raise RuntimeError("RetirementEnv.step must return a tuple")
        if len(out)==5:
            s_next, r, done, trunc, info = out
        elif len(out)==4:
            s_next, r, done, info = out; trunc=False
        else:
            raise RuntimeError(f"Unexpected step() return length: {len(out)}")
        obs = self._obs_vec(s_next)
        if isinstance(info, dict) and "W_T" not in info:
            try:
                W_T = float(getattr(self.base, "W", None) or (s_next.get("W") if isinstance(s_next, dict) else 0.0))
                info["W_T"] = W_T
            except Exception:
                pass
        return obs, float(r), bool(done), bool(trunc), info

    def close(self):
        if hasattr(self.base, "close"): self.base.close()

# ---------- Utility ----------
def u_crra(c: float, gamma: float = 3.0, eps: float = 1e-8) -> float:
    c = max(float(c), 0.0) + eps
    if abs(gamma - 1.0) < 1e-9:
        return math.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

# ---------- Policy ----------
class BetaHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc_a = nn.Linear(in_dim, 1)
        self.fc_b = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.fc_a.weight); nn.init.zeros_(self.fc_a.bias)
        nn.init.xavier_uniform_(self.fc_b.weight); nn.init.zeros_(self.fc_b.bias)
    def forward(self, h):
        alpha = F.softplus(self.fc_a(h)) + 1.001
        beta  = F.softplus(self.fc_b(h)) + 1.001
        return alpha, beta

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head_q = BetaHead(hidden)
        self.head_w = BetaHead(hidden)
        self.v_head  = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.v_head.weight); nn.init.zeros_(self.v_head.bias)
    def forward(self, obs):
        h = self.backbone(obs)
        a_q, b_q = self.head_q(h)
        a_w, b_w = self.head_w(h)
        v = self.v_head(h).squeeze(-1)
        return (a_q, b_q), (a_w, b_w), v
    def act(self, obs, cfg, eval_mode=False):
        (a_q, b_q), (a_w, b_w), v = self.forward(obs)
        dist_q = Beta(a_q, b_q); dist_w = Beta(a_w, b_w)
        if eval_mode:
            raw_q = (a_q / (a_q + b_q)).clamp(1e-4, 1-1e-4)
            raw_w = (a_w / (a_w + b_w)).clamp(1e-4, 1-1e-4)
        else:
            raw_q = dist_q.rsample().clamp(1e-4, 1-1e-4)
            raw_w = dist_w.rsample().clamp(1e-4, 1-1e-4)

        # map to constraints
        q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
        w_max   = float(getattr(cfg, "w_max", 1.0) or 1.0)
        q_cap   = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)

        q = q_floor + (1.0 - q_floor) * raw_q
        w = w_max * raw_w

        # q_cap both in train & eval
        if q_cap > 0.0:
            q = torch.minimum(q, torch.as_tensor(q_cap, dtype=q.dtype, device=q.device))

        logp = dist_q.log_prob(raw_q) + dist_w.log_prob(raw_w)
        ent  = dist_q.entropy().squeeze(-1) + dist_w.entropy().squeeze(-1)
        return q.squeeze(-1), w.squeeze(-1), logp.squeeze(-1), ent, v, (dist_q, dist_w)

# ---------- GAE ----------
def compute_gae(rews, vals, dones, gamma, lam):
    T = len(rews)
    adv = torch.zeros(T, device=vals.device)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rews[t] + gamma * (vals[t+1] if t+1 < T else 0.0) * nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals[:T]
    return adv, ret

# ---------- Dual CVaR ----------
class DualCVaR:
    def __init__(self, alpha=0.95, eta_init=0.0, tau=0.3):
        self.alpha = alpha; self.eta = eta_init; self.tau = tau
    def update_eta_with_batch(self, L_batch):
        q_alpha = np.quantile(L_batch, self.alpha)
        self.eta = (1.0 - self.tau) * self.eta + self.tau * float(q_alpha)
    def terminal_penalty(self, L):
        return self.eta + max(L - self.eta, 0.0) / (1.0 - self.alpha)

# ---------- Dual CVaR (stage-wise for consumption shortfall) ----------
class DualCVaRStage:
    def __init__(self, alpha=0.95, eta_init=0.0, tau=0.3):
        self.alpha = alpha; self.eta = eta_init; self.tau = tau
    def update_eta_with_batch(self, Ls):
        # Ls: list or np.ndarray of per-step shortfall losses
        if len(Ls) == 0:
            return
        q_alpha = np.quantile(Ls, self.alpha)
        self.eta = (1.0 - self.tau) * self.eta + self.tau * float(q_alpha)
    def penalty(self, L):
        return self.eta + max(L - self.eta, 0.0) / (1.0 - self.alpha)

# ---------- Teacher policy ----------
def teacher_action(cfg):
    # 4% rule monthly, w≈0.6(clip to w_max)
    q4_m = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
    q_cap = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)
    q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    w_teacher = min(0.60, float(getattr(cfg, "w_max", 1.0)))
    q_teacher = q4_m
    if q_cap > 0.0: q_teacher = min(q_teacher, q_cap)
    q_teacher = max(q_teacher, q_floor)
    return float(q_teacher), float(w_teacher)

# ---------- Rollout ----------
def rollout(env, policy, cfg, steps, gamma, lam, device,
            cvar_hook, lw_scale, survive_bonus, teacher_eps,
            stage_cvar: DualCVaRStage = None, cstar_mode: str = "annuity"):
    obs_list, act_q, act_w, logp_list, ent_list, val_list, rew_list, done_list = ([] for _ in range(8))
    stage_L_list = []  # for stage-wise CVaR eta update
    obs, _ = env.reset(seed=None)

    q_cap = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)
    q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    w_max   = float(getattr(cfg, "w_max", 1.0) or 1.0)
    gamma_crra = float(getattr(cfg, "crra_gamma", 3.0))
    u_scale    = float(getattr(cfg, "u_scale", 0.0))

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q, w, logp, ent, v, (dist_q, dist_w) = policy.act(obs_t, cfg, eval_mode=False)

        # teacher forcing (on-policy logp for teacher action)
        if random.random() < max(0.0, min(1.0, float(teacher_eps))):
            tq, tw = teacher_action(cfg)
            if q_cap > 0.0: tq = min(tq, q_cap)
            tq = max(tq, q_floor); tw = min(max(tw, 0.0), w_max)
            denom_q = max(1e-8, 1.0 - q_floor)
            raw_q_t = np.clip((tq - q_floor) / denom_q, 1e-4, 1-1e-4)
            raw_w_t = np.clip(tw / max(w_max, 1e-8), 1e-4, 1-1e-4)
            raw_q_t = torch.tensor([raw_q_t], dtype=torch.float32, device=device)
            raw_w_t = torch.tensor([raw_w_t], dtype=torch.float32, device=device)
            logp = (dist_q.log_prob(raw_q_t) + dist_w.log_prob(raw_w_t)).squeeze(-1)
            q = torch.tensor([tq], dtype=torch.float32, device=device)
            w = torch.tensor([tw], dtype=torch.float32, device=device)

        if q_cap > 0.0:
            q = torch.clamp(q, max=q_cap)

        obs2, c_t, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))

        # reward shaping
        rew = 0.0
        # utility of consumption
        if u_scale != 0.0:
            rew += u_scale * u_crra(c_t, gamma=gamma_crra)

        # stage-wise CVaR on consumption shortfall
        if (stage_cvar is not None) and (getattr(cfg, "cvar_stage_on", False)):
            # define c_star_t
            mode = str(cstar_mode or "annuity")
            if mode == "fixed":
                c_star = float(getattr(cfg, "cstar_m", 0.04/12)) * 1.0  # wealth-normalized obs에서 1.0은 W0
                c_star = float(c_star) * float(obs[1])  # obs[1]=W/W0
            elif mode == "vpw":
                # very light VPW proxy: use cfg.monthly annuity factor at starting g
                m = cfg.monthly(); g = m['g_m']; Nm = env.base.T - env.base.t
                a = (1.0 - (1.0+g)**(-Nm))/g if g > 0 else max(Nm, 1)
                q_m = min(1.0, 1.0 / a)
                c_star = q_m * float(obs[1])  # wealth-normalized
            else:
                # annuity-style: p_m * W_t
                c_star = float(cfg.monthly()['p_m']) * float(obs[1])
            L_t = max(c_star - float(c_t), 0.0)
            stage_L_list.append(L_t)
            lam_s = float(getattr(cfg, "lambda_stage", 0.0) or 0.0)
            if lam_s > 0.0:
                rew -= lam_s * stage_cvar.penalty(L_t)

        if not (done or trunc) and survive_bonus != 0.0:
            rew += float(survive_bonus)
        if (done or trunc):
            if cvar_hook is not None:
                rew += float(cvar_hook(info))           # −λ·CVaR_dual(L) on terminal wealth
            lw = float(getattr(cfg, "lw_scale", 0.0) or 0.0)
            if lw != 0.0:
                rew += lw * float(info.get("W_T", 0.0))
            # (optional) bequest at death is already included only if user shapes it here
            # ex) rew += bequest_weight * info.get("bequest", 0.0)

        # keep graph
        obs_list.append(obs_t.squeeze(0))
        act_q.append(q.detach()); act_w.append(w.detach())
        logp_list.append(logp)    # keep grad
        ent_list.append(ent)
        val_list.append(v)
        rew_list.append(torch.tensor(rew, dtype=torch.float32, device=device))
        done_list.append(done or trunc)
        obs = obs2
        if done or trunc:
            obs, _ = env.reset(seed=None)

    obs_t = torch.stack(obs_list)
    logp  = torch.stack(logp_list)
    ent   = torch.stack(ent_list)
    val   = torch.stack(val_list)
    rews  = torch.stack(rew_list)
    dones = torch.tensor(done_list, device=device, dtype=torch.bool)
    adv, ret = compute_gae(rews, val, dones, gamma, lam)
    return {"obs": obs_t, "q": torch.stack(act_q), "w": torch.stack(act_w),
            "logp": logp, "ent": ent, "val": val, "adv": adv.detach(), "ret": ret.detach(),
            "stage_L": np.asarray(stage_L_list, dtype=np.float64)}

# ---------- Eval (mean-policy) ----------
def evaluate_mean_policy(make_env_fn, policy, cfg, n_paths=300, device="cpu"):
    Ws = []
    with torch.no_grad():
        for _ in range(n_paths):
            env = make_env_fn()
            obs, _ = env.reset(seed=None)
            done = False; trunc = False
            while not (done or trunc):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q, w, _, _, _, _ = policy.act(obs_t, cfg, eval_mode=True)
                obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            W_T = info.get("W_T", None)
            if W_T is None and hasattr(env, "W"): W_T = float(env.W)
            Ws.append(float(W_T)); env.close()
    Ws = np.array(Ws, dtype=np.float64)
    EW = float(Ws.mean())
    k = max(1, int(0.05 * len(Ws)))
    tail_idx = np.argsort(Ws)[:k]
    ES95_loss = float(np.mean(np.maximum(getattr(cfg, "F_target", 1.0) - Ws[tail_idx], 0.0)))
    Ruin = float((Ws < 1e-9).mean())
    return {"EW": EW, "ES95": ES95_loss, "Ruin": Ruin, "mean_WT": float(Ws.mean())}

# ---------- XAI helpers ----------
def make_policy_heatmaps(policy, cfg, outputs, device="cpu"):
    import matplotlib.pyplot as plt
    out = Path(outputs) / "xai"; out.mkdir(parents=True, exist_ok=True)
    W_grid = np.linspace(0.1, 2.0, 81, dtype=np.float32)
    Tn = 20  # 20 time slices
    Q = np.zeros((Tn, len(W_grid)), dtype=np.float32)
    Wp = np.zeros_like(Q)
    with torch.no_grad():
        policy.eval()
        for ti in range(Tn):
            t_norm = ti / (Tn-1)
            for j, wv in enumerate(W_grid):
                obs = np.array([t_norm, wv, 65.0/120.0, 0.0], dtype=np.float32)
                q_m, w_m, _, _, _, _ = policy.act(torch.tensor(obs).unsqueeze(0).to(device), cfg, eval_mode=True)
                Q[ti, j] = float(q_m.item()); Wp[ti, j] = float(w_m.item())
    for name, A in [("Pi_q_heatmap.png", Q), ("Pi_w_heatmap.png", Wp)]:
        plt.figure()
        plt.imshow(A, aspect="auto", origin="lower",
                   extent=[W_grid.min(), W_grid.max(), 0.0, 1.0])
        plt.colorbar()
        plt.xlabel("Wealth (W/W0)"); plt.ylabel("t_norm")
        plt.title(name.replace(".png",""))
        plt.tight_layout()
        plt.savefig(out / name); plt.close()

def collect_occupancy(make_env_fn, policy, cfg, outputs, n_paths=300, device="cpu"):
    import matplotlib.pyplot as plt
    out = Path(outputs) / "xai"; out.mkdir(parents=True, exist_ok=True)
    hist = np.zeros((50, 50), dtype=np.int32)
    with torch.no_grad():
        for _ in range(n_paths):
            env = make_env_fn()
            obs, _ = env.reset(seed=None)
            done = False; trunc = False
            while not (done or trunc):
                t_norm = float(obs[0]); Wn = float(obs[1])
                i = min(49, max(0, int(t_norm*50)))
                j = min(49, max(0, int((Wn/2.0)*50)))  # W in [0,2]
                hist[i, j] += 1
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q, w, _, _, _, _ = policy.act(obs_t, cfg, eval_mode=True)
                obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            env.close()
    plt.figure()
    plt.imshow(hist, aspect="auto", origin="lower")
    plt.colorbar(); plt.title("Occupancy (t_norm vs W/W0)")
    plt.tight_layout(); plt.savefig(out / "occupancy.png"); plt.close()

def replay_tail_paths(make_env_fn, policy, cfg, outputs, k=8, alpha=0.05, device="cpu"):
    import matplotlib.pyplot as plt
    out = Path(outputs) / "xai"; out.mkdir(parents=True, exist_ok=True)
    paths = []
    with torch.no_grad():
        for _ in range(200):
            env = make_env_fn()
            obs, _ = env.reset(seed=None)
            Wts, qs, ws = [], [], []
            done = False; trunc = False
            while not (done or trunc):
                Wts.append(float(obs[1]))
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q, w, _, _, _, _ = policy.act(obs_t, cfg, eval_mode=True)
                qs.append(float(q.item())); ws.append(float(w.item()))
                obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            WT = info.get("W_T", float("nan"))
            paths.append((WT, np.array(Wts), np.array(qs), np.array(ws)))
            env.close()
    paths.sort(key=lambda x: x[0])
    sel = paths[:k]
    for idx, (WT, Wts, qs, ws) in enumerate(sel):
        t = np.arange(len(Wts))
        plt.figure(); plt.plot(t, Wts); plt.title(f"tail path {idx} WT={WT:.3f}"); plt.tight_layout()
        plt.savefig(out / f"tail_{idx}_W.png"); plt.close()
        plt.figure(); plt.plot(t, qs); plt.title(f"q(t) tail {idx}"); plt.tight_layout()
        plt.savefig(out / f"tail_{idx}_q.png"); plt.close()
        plt.figure(); plt.plot(t, ws); plt.title(f"w(t) tail {idx}"); plt.tight_layout()
        plt.savefig(out / f"tail_{idx}_w.png"); plt.close()

# ---------- CSV logger ----------
def append_metrics_csv(outputs_dir, fields):
    out_logs = Path(outputs_dir) / "_logs"; out_logs.mkdir(parents=True, exist_ok=True)
    dest = out_logs / "metrics.csv"; write_header = (not dest.exists())
    with dest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields.keys())
        if write_header: w.writeheader()
        w.writerow(fields)

# ---------- Entrypoint ----------
def train_rl(cfg, seed_list, outputs, n_paths_eval=300, rl_epochs=60, steps_per_epoch=2048,
             lr=3e-4, gamma=None, gae_lambda=0.95, entropy_coef=0.01, value_coef=0.5,
             max_grad_norm=0.5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gamma = gamma or getattr(cfg, "beta", 0.996)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    def _make_env_local(): return _GymShim(RetirementEnv(cfg))

    # CVaR dual
    cvar = DualCVaR(alpha=getattr(cfg, "alpha", 0.95), eta_init=0.0, tau=0.3)
    lambda_term = float(getattr(cfg, "lambda_term", 0.0) or 0.0)
    F_target    = float(getattr(cfg, "F_target", 1.0) or 1.0)

    def cvar_hook(info):
        if lambda_term == 0.0: return 0.0
        W_T = info.get("W_T", None)
        if W_T is None: return 0.0
        L = max(F_target - float(W_T), 0.0)
        return - lambda_term * cvar.terminal_penalty(L)

    # Stage-wise CVaR
    stage_on = bool(getattr(cfg, "cvar_stage_on", False))
    stage_cvar = DualCVaRStage(alpha=getattr(cfg, "alpha_stage", 0.95), eta_init=0.0, tau=0.3) if stage_on else None
    cstar_mode = str(getattr(cfg, "cstar_mode", "annuity"))

    # shaping params
    lw_scale       = float(getattr(cfg, "lw_scale", 0.0) or 0.0)
    survive_bonus  = float(getattr(cfg, "survive_bonus", 0.0) or 0.0)
    teacher_eps0   = float(getattr(cfg, "teacher_eps0", 0.0) or 0.0)
    teacher_decay  = float(getattr(cfg, "teacher_decay", 1.0) or 1.0)

    # build once to get obs_dim
    env0 = _make_env_local(); obs0, _ = env0.reset(seed=None)
    obs_dim = int(np.asarray(obs0, dtype=np.float32).shape[0]); env0.close()

    policy = PolicyNet(obs_dim).to(device)
    optim_all = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(rl_epochs):
        eps = teacher_eps0 * (teacher_decay ** epoch)
        batch = rollout(_make_env_local(), policy, cfg, steps_per_epoch, gamma, gae_lambda,
                        device, cvar_hook, lw_scale, survive_bonus, eps,
                        stage_cvar=stage_cvar, cstar_mode=cstar_mode)
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        pi_loss = -(batch["logp"] * adv).mean()
        v_loss  = 0.5 * ((batch["val"] - batch["ret"])**2).mean()
        ent_b   = batch["ent"].mean()
        loss = pi_loss + value_coef * v_loss - entropy_coef * ent_b

        optim_all.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optim_all.step()

        # update eta using small mean-policy rollouts (terminal CVaR)
        Ws_tmp = []
        for _ in range(8):
            env = _make_env_local()
            obs, _ = env.reset(seed=None)
            done = False; trunc = False
            with torch.no_grad():
                while not (done or trunc):
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q, w, _, _, _, _ = policy.act(obs_t, cfg, eval_mode=True)
                    obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            W_T = info.get("W_T", None)
            if W_T is None and hasattr(env, "W"): W_T = float(env.W)
            Ws_tmp.append(float(W_T)); env.close()
        if lambda_term > 0.0 and len(Ws_tmp) > 0:
            Ls = np.maximum(F_target - np.array(Ws_tmp, dtype=np.float64), 0.0)
            cvar.update_eta_with_batch(Ls)

        # stage-wise CVaR eta update
        if stage_on and stage_cvar is not None:
            stage_cvar.update_eta_with_batch(batch.get("stage_L", []))

    # evaluation (mean-policy)
    metrics = evaluate_mean_policy(_make_env_local, policy, cfg, n_paths=n_paths_eval, device=device)

    # XAI (optional)
    if bool(getattr(cfg, "xai_on", True)):
        try:
            make_policy_heatmaps(policy, cfg, outputs, device=device)
            collect_occupancy(_make_env_local, policy, cfg, outputs, n_paths=200, device=device)
            replay_tail_paths(_make_env_local, policy, cfg, outputs, k=6, device=device)
        except Exception:
            pass

    ts = time.strftime("%y%m%dT%H%M%S")
    fields = dict(
        ts=ts, asset=getattr(cfg, "asset", "US"), method="rl", baseline="",
        es_mode="loss", F_target=F_target, w_max=getattr(cfg, "w_max", 1.0),
        hedge_on=getattr(cfg, "hedge_on", False), hedge_mode=getattr(cfg, "hedge_mode", ""),
        hedge_sigma_k=getattr(cfg, "hedge_sigma_k", 0.0), lambda_term=lambda_term,
        fee_annual=getattr(cfg, "phi_adval", 0.0), floor_on=getattr(cfg, "floor_on", False),
        f_min_real=getattr(cfg, "f_min_real", 0.0), EW=metrics["EW"], ES95=metrics["ES95"],
        Ruin=metrics["Ruin"], mean_WT=metrics["mean_WT"], seeds=" ".join(map(str, seed_list)),
        n_paths_eval=n_paths_eval, outputs=str(outputs),
        mortality_on=getattr(cfg, "mortality_on", False),
        market_mode=getattr(cfg, "market_mode", "iid"),
        cvar_stage_on=getattr(cfg, "cvar_stage_on", False),
    )
    append_metrics_csv(outputs, fields)

    # optional XAI 1D slice (t=0, W∈[0.1,2.0]) NPZ
    out_xai = Path(outputs) / "xai"
    out_xai.mkdir(parents=True, exist_ok=True)
    try:
        W_grid = np.linspace(0.1, 2.0, 121, dtype=np.float32)
        policy.eval()
        with torch.no_grad():
            Pi_q, Pi_w = [], []
            for wv in W_grid:
                obs = np.array([0.0, wv, 65.0/120.0, 0.0], dtype=np.float32)
                q_m, w_m, _, _, _, _ = policy.act(torch.tensor(obs).unsqueeze(0).to(device), cfg, eval_mode=True)
                Pi_q.append(float(q_m.item())); Pi_w.append(float(w_m.item()))
        np.savez(str(out_xai / "policy_rl_mean.npz"), Pi_w=np.array(Pi_w), Pi_q=np.array(Pi_q), W_grid=W_grid)
    except Exception:
        pass

    return fields
