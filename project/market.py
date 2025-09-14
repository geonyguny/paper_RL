import numpy as np
import csv, os

class IIDNormalMarket:
    def __init__(self, cfg):
        m = cfg.monthly()
        self.mu_m = float(m["mu_m"])
        self.sigma_m = max(1e-12, float(m["sigma_m"]))
        self.rf_m = float(m["rf_m"])
        self.rng = np.random.RandomState(getattr(cfg, "seed", 0))

    def seed(self, s):
        self.rng = np.random.RandomState(int(s))

    def sample_risky(self, T: int):
        return self.rng.normal(loc=self.mu_m, scale=self.sigma_m, size=int(T))

class BootstrapMarket:
    """
    CSV 스키마(월별): date, risky_nom, tbill_nom, cpi
    실질 변환: r_real = (1+r_nom)/(1+inf)-1,  inf = cpi_t/cpi_{t-1}-1
    """
    def __init__(self, cfg):
        path = getattr(cfg, "market_csv", None)
        if (path is None) or (not os.path.exists(path)):
            raise FileNotFoundError("BootstrapMarket requires --market_csv CSV file")
        self.use_real_rf = bool(getattr(cfg, "use_real_rf", True))
        self.block = max(1, int(getattr(cfg, "bootstrap_block", 24)))
        data = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                risky_nom = float(row["risky_nom"])
                rf_nom    = float(row["tbill_nom"])
                cpi       = float(row["cpi"])
                data.append((risky_nom, rf_nom, cpi))
        arr = np.asarray(data, dtype=np.float64)
        risky_nom = arr[:,0]; rf_nom = arr[:,1]; cpi = arr[:,2]
        inf = np.zeros_like(cpi)
        inf[1:] = (cpi[1:] / np.maximum(1e-12, cpi[:-1])) - 1.0
        risky_real = (1.0 + risky_nom) / (1.0 + inf) - 1.0
        rf_real    = (1.0 + rf_nom) / (1.0 + inf) - 1.0
        self.risky_real = risky_real
        self.rf_real    = rf_real
        self.rng = np.random.RandomState(getattr(cfg, "seed", 0))

    def seed(self, s):
        self.rng = np.random.RandomState(int(s))

    def sample_paths(self, T: int, block: int = None):
        block = block or self.block
        n = len(self.risky_real)
        out_risky = []
        out_safe  = []
        while len(out_risky) < T:
            start = self.rng.randint(0, n)
            idx = [(start + i) % n for i in range(block)]
            out_risky.extend(self.risky_real[idx].tolist())
            out_safe.extend(self.rf_real[idx].tolist())
        return np.asarray(out_risky[:T], dtype=np.float64), np.asarray(out_safe[:T], dtype=np.float64)
