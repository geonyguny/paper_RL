# project/market.py
from __future__ import annotations

import os
import csv
import numpy as np
from typing import Optional, Tuple, Sequence, Dict, Any


def _make_rng(seed: Optional[int]) -> np.random.Generator:
    """
    통일된 RNG 생성 유틸.
    - seed is None → OS entropy 기반 default_rng()
    - seed is int  → SeedSequence(seed) 기반 default_rng()
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence(int(seed)))


class IIDNormalMarket:
    """
    정규(IID) 실질수익률 백엔드.
    cfg.monthly()에서 월별 μ, σ, r_f를 읽고, 위험자산 경로를 N(μ, σ)로 시뮬레이션.
    """
    def __init__(self, cfg: Any, rng: Optional[np.random.Generator] = None) -> None:
        m = cfg.monthly()
        self.mu_m = float(m["mu_m"])
        self.sigma_m = max(1e-12, float(m["sigma_m"]))
        self.rf_m = float(m["rf_m"])
        # ▼ line 10/13 대체: RandomState → default_rng
        self.rng: np.random.Generator = rng if rng is not None else _make_rng(getattr(cfg, "seed", None))

    # ▼ line 13 대체: RandomState → default_rng
    def seed(self, s: Optional[int]) -> None:
        """외부에서 재현성 있는 재시드."""
        self.rng = _make_rng(int(s) if s is not None else None)

    def sample_risky(self, T: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """길이 T 위험자산 실질수익률 경로."""
        g = rng if rng is not None else self.rng
        return g.normal(loc=self.mu_m, scale=self.sigma_m, size=int(T)).astype(np.float64)


class BootstrapMarket:
    """
    CSV 스키마(월별): date, risky_nom, tbill_nom, cpi
    실질 변환: r_real = (1+r_nom)/(1+inf) - 1,  inf = cpi_t/cpi_{t-1} - 1

    Moving-Block Bootstrap(MBB)로 길이 T 경로를 만든다.
    부족 시 래핑(모듈로)로 채운다.
    """
    REQUIRED_COLS = ("risky_nom", "tbill_nom", "cpi")

    def __init__(self, cfg: Any, rng: Optional[np.random.Generator] = None) -> None:
        path = getattr(cfg, "market_csv", None)
        if (path is None) or (not os.path.exists(path)):
            raise FileNotFoundError("BootstrapMarket requires --market_csv CSV file")

        self.use_real_rf = str(getattr(cfg, "use_real_rf", "on")).lower() == "on"
        self.block = max(1, int(getattr(cfg, "bootstrap_block", 24)))

        # ▼ line 45 대체: RandomState → default_rng
        self.rng: np.random.Generator = rng if rng is not None else _make_rng(getattr(cfg, "seed", None))

        # 데이터 적재
        risky_nom, rf_nom, cpi = self._load_csv(path)
        risky_real, rf_real = self._to_real_returns(risky_nom, rf_nom, cpi)
        self.risky_real = risky_real.astype(np.float64)
        self.rf_real = rf_real.astype(np.float64)

        if self.risky_real.size < 2:
            raise ValueError("bootstrap input series too short")

    # ▼ line 48 대체: RandomState → default_rng
    def seed(self, s: Optional[int]) -> None:
        """외부에서 재현성 있는 재시드."""
        self.rng = _make_rng(int(s) if s is not None else None)

    # ---------- I/O & Transform ----------

    def _load_csv(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            missing = [c for c in self.REQUIRED_COLS if c not in r.fieldnames]
            if missing:
                raise ValueError(f"CSV missing columns: {missing} (required={self.REQUIRED_COLS})")
            for row in r:
                try:
                    risky_nom = float(row["risky_nom"])
                    rf_nom = float(row["tbill_nom"])
                    cpi = float(row["cpi"])
                    rows.append((risky_nom, rf_nom, cpi))
                except Exception:
                    # 잘못된 행은 스킵
                    continue
        if not rows:
            raise ValueError("empty/invalid CSV for bootstrap market")

        arr = np.asarray(rows, dtype=np.float64)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    @staticmethod
    def _to_real_returns(risky_nom: np.ndarray, rf_nom: np.ndarray, cpi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 인플레율
        cpi = np.asarray(cpi, dtype=np.float64)
        inf = np.zeros_like(cpi)
        # cpi[t]/cpi[t-1]-1 (첫 달은 0으로 둠)
        denom = np.maximum(cpi[:-1], 1e-12)
        inf[1:] = (cpi[1:] / denom) - 1.0

        risky_real = (1.0 + risky_nom) / (1.0 + inf) - 1.0
        rf_real = (1.0 + rf_nom) / (1.0 + inf) - 1.0
        return risky_real, rf_real

    # ---------- Sampling ----------

    def sample_paths(
        self,
        T: int,
        block: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        wrap: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Moving-Block Bootstrap 경로 생성.
        - block: 블록 길이(기본 self.block)
        - wrap : 블록이 끝을 넘으면 모듈로 래핑 허용
        """
        g = rng if rng is not None else self.rng
        B = int(block or self.block)
        T = int(T)
        n = int(self.risky_real.size)

        if n <= 0:
            raise ValueError("empty series for bootstrap")
        if B <= 0:
            raise ValueError("block size must be positive")

        out_risky = np.empty(T, dtype=np.float64)
        out_safe = np.empty(T, dtype=np.float64)
        filled = 0

        while filled < T:
            if wrap:
                start = int(g.integers(0, n))  # 0..n-1
                length = min(B, T - filled)
                # 래핑 허용: 모듈로 인덱싱으로 한 번에 채움
                idx = (start + np.arange(length)) % n
                out_risky[filled:filled + length] = self.risky_real[idx]
                out_safe[filled:filled + length] = self.rf_real[idx]
                filled += length
            else:
                # non-wrap: 시작점은 0..(n-B)
                if n < B:
                    raise ValueError("series shorter than block (wrap=False)")
                start = int(g.integers(0, n - B + 1))
                length = min(B, T - filled)
                out_risky[filled:filled + length] = self.risky_real[start:start + length]
                out_safe[filled:filled + length] = self.rf_real[start:start + length]
                filled += length

        return out_risky, out_safe
