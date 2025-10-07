# D:\01_simul\project\utils\metrics_utils.py
import numpy as np

def terminal_losses(W_T, F_target: float) -> np.ndarray:
    """L_i = max(F_target - W_T_i, 0)"""
    W_T = np.asarray(W_T, dtype=float).reshape(-1)
    return np.maximum(float(F_target) - W_T, 0.0)

def cvar_alpha(losses: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR_α(loss) = E[loss | loss ≥ VaR_α(loss)]"""
    losses = np.asarray(losses, dtype=float).reshape(-1)
    if losses.size == 0:
        return 0.0
    # 보수적 추정 원하면 method="higher" 사용(NumPy 1.22+)
    try:
        q = np.quantile(losses, alpha, method="higher")
    except TypeError:
        q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    return float(tail.mean()) if tail.size else 0.0
