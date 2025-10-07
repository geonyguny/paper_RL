# tests/test_cvar_interp.py
import numpy as np
import math
from project.runner.cli import _cvar_fallback

def es_interp_reference(L, alpha):
    L = np.sort(np.asarray(L, dtype=float))
    n = L.size
    if n == 0:
        return 0.0
    a = max(min(float(alpha), 1.0 - 1e-12), 1e-12)
    j = int(math.floor(n * a))
    if j >= n:
        j = n - 1
    theta = n * a - j
    Lj1 = float(L[j])
    tail_sum = float(L[j+1:].sum())
    return ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))

def test_cvar_fallback_matches_reference():
    rng = np.random.default_rng(0)
    for n in [10, 53, 100, 257]:
        L = rng.random(n)  # 임의 손실
        for a in [0.90, 0.95, 0.975, 0.99]:
            got = _cvar_fallback(L, a)
            ref = es_interp_reference(L, a)
            assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)
