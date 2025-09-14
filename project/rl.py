
import numpy as _np
from .config import SimConfig

class BetaPolicy:
    def __init__(self, cfg: SimConfig, hidden=64):
        self.cfg = cfg

    def act(self, state):
        W = state['W']; t = state['t']; T = self.cfg.horizon_years * self.cfg.steps_per_year
        q = 0.02
        if self.cfg.floor_on and self.cfg.f_min_real > 0:
            q = max(q, min(1.0, self.cfg.f_min_real/max(W,1e-9)))
        age_factor = 1.0 - (t / max(T,1))
        w = min(self.cfg.w_max, max(0.0, 0.5 * age_factor + 0.2))
        return q, w

class A2CTrainer:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.policy = BetaPolicy(cfg, hidden=cfg.rl_hidden)

    def train(self, seed=0):
        _ = _np.random.default_rng(seed)
        return self.policy
