import os
import sys
import csv
import argparse
import numpy as np

# --- make "project" package importable when this script lives under outputs/xai/ ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Now safe to import project modules
from project.config import SimConfig
from project.env import RetirementEnv


def load_policy(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"policy dump not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=False)
    for k in ("Pi_w", "Pi_q", "W_grid"):
        if k not in d:
            raise KeyError(f"missing key '{k}' in {npz_path}")
    Pw, Pq, Wg = d["Pi_w"], d["Pi_q"], d["W_grid"]
    if Pw.shape != Pq.shape:
        raise ValueError(f"Pi_w/Pi_q shape mismatch: {Pw.shape} vs {Pq.shape}")
    return Pw, Pq, Wg


def make_cfg(args) -> SimConfig:
    cfg = SimConfig(asset=args.asset)
    # market
    cfg.mu_annual = float(args.mu_annual)
    cfg.sigma_annual = float(args.sigma_annual)
    cfg.rf_annual = float(args.rf_annual)
    # horizon/controls
    cfg.horizon_years = int(args.horizon_years)
    cfg.w_max = float(args.w_max)
    # objective
    cfg.F_target = float(args.F_target)
    cfg.alpha = float(args.alpha)
    cfg.phi_adval = float(args.fee_annual)
    # misc (for safety/stability)
    cfg.hjb_W_max = float(args.hjb_W_max)
    # hedge options
    cfg.hedge_on = (args.hedge == "on")
    cfg.hedge_mode = args.hedge_mode
    cfg.hedge_sigma_k = float(args.hedge_sigma_k)
    # floor (off by default here)
    cfg.floor_on = bool(args.floor_on)
    cfg.f_min_real = float(args.f_min_real)
    return cfg


def make_actor(Pq: np.ndarray, Pw: np.ndarray, Wg: np.ndarray):
    T_pol = int(Pw.shape[0])

    def actor(s):
        t_idx = min(int(s["t"]), T_pol - 1)
        i = int(np.clip(np.searchsorted(Wg, s["W"]) - 1, 0, Wg.size - 2))
        q = float(Pq[t_idx, i])
        w = float(Pw[t_idx, i])
        return q, w

    return actor


def simulate(cfg: SimConfig, actor, seeds, K: int):
    env = RetirementEnv(cfg)
    T = env.T
    WT = []
    Q = np.full((T, len(seeds) * K), np.nan, dtype=float)
    W = np.full((T, len(seeds) * K), np.nan, dtype=float)
    col = 0
    for sd in seeds:
        for k in range(K):
            env.reset(seed=sd * 100000 + k)
            for t in range(T):
                st = env._state()
                q, w = actor(st)
                Q[t, col] = q
                W[t, col] = w
                _, _, done, _ = env.step(q, w)
                if done:
                    break
            WT.append(env.W)
            col += 1
    return np.array(WT), Q, W


def save_bands_and_hist(WT: np.ndarray, Q: np.ndarray, W: np.ndarray, outdir: str, bins: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # percentile bands
    q_p = (5, 10, 50, 90, 95)
    w_p = (5, 10, 50, 90, 95)
    q_bands = {p: np.nanpercentile(Q, p, axis=1) for p in q_p}
    w_bands = {p: np.nanpercentile(W, p, axis=1) for p in w_p}
    x = np.arange(Q.shape[0])

    # q bands
    plt.plot(x, q_bands[50])
    plt.fill_between(x, q_bands[10], q_bands[90], alpha=0.3)
    plt.fill_between(x, q_bands[5], q_bands[95], alpha=0.15)
    plt.title("q_t bands")
    plt.xlabel("t")
    plt.ylabel("q")
    plt.savefig(os.path.join(outdir, "q_t_bands.png"))
    plt.clf()

    # w bands
    plt.plot(x, w_bands[50])
    plt.fill_between(x, w_bands[10], w_bands[90], alpha=0.3)
    plt.fill_between(x, w_bands[5], w_bands[95], alpha=0.15)
    plt.title("w_t bands")
    plt.xlabel("t")
    plt.ylabel("w")
    plt.savefig(os.path.join(outdir, "w_t_bands.png"))
    plt.clf()

    # terminal histogram
    plt.hist(WT, bins=bins)
    plt.title("WT histogram")
    plt.xlabel("W_T")
    plt.ylabel("freq")
    plt.savefig(os.path.join(outdir, "wt_hist.png"))
    plt.clf()

    # quantiles csv
    qs = [5, 10, 50, 90, 95]
    vals = [float(np.percentile(WT, q)) for q in qs]
    mean = float(WT.mean())
    with open(os.path.join(outdir, "wt_quantiles.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p5", "p10", "median", "p90", "p95", "mean"])
        w.writerow([vals[0], vals[1], vals[2], vals[3], vals[4], mean])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_npz", default=r".\outputs\xai\policy_us15_k050_l1.npz")
    ap.add_argument("--outdir", default=r".\outputs\xai")

    # market/horizon defaults (US case in your runs)
    ap.add_argument("--asset", default="US")
    ap.add_argument("--mu_annual", type=float, default=0.065)
    ap.add_argument("--sigma_annual", type=float, default=0.16)
    ap.add_argument("--rf_annual", type=float, default=0.02)
    ap.add_argument("--horizon_years", type=int, default=15)
    ap.add_argument("--w_max", type=float, default=1.0)

    # objective/fees
    ap.add_argument("--F_target", type=float, default=1.1)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--fee_annual", type=float, default=0.004)
    ap.add_argument("--hjb_W_max", type=float, default=5.0)

    # hedge/floor
    ap.add_argument("--hedge", choices=["on", "off"], default="on")
    ap.add_argument("--hedge_mode", choices=["mu", "sigma"], default="sigma")
    ap.add_argument("--hedge_sigma_k", type=float, default=0.5)
    ap.add_argument("--floor_on", action="store_true")
    ap.add_argument("--f_min_real", type=float, default=0.0)

    # simulation size
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_paths", type=int, default=300)
    ap.add_argument("--bins", type=int, default=40)

    args = ap.parse_args()

    Pw, Pq, Wg = load_policy(args.policy_npz)
    cfg = make_cfg(args)
    actor = make_actor(Pq, Pw, Wg)

    WT, Q, W = simulate(cfg, actor, args.seeds, args.n_paths)

    os.makedirs(args.outdir, exist_ok=True)
    save_bands_and_hist(WT, Q, W, args.outdir, args.bins)

    print(f"XAI OK | WT n={WT.size} | outdir={args.outdir}")


if __name__ == "__main__":
    main()
