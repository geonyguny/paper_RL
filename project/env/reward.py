# project/env/reward.py
import numpy as np
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


def _is_torch(x):
    return _HAS_TORCH and hasattr(x, "device")


def _clip_pos(xp, x, eps=1e-12):
    # numpy / torch 공용 clip(min) 구현
    return xp.clip(x, eps, None) if xp is np else (x.clamp(min=eps))


def _u_crra(x, gamma, xp):
    x = _clip_pos(xp, x)
    if abs(gamma - 1.0) < 1e-12:
        return xp.log(x) if xp is np else torch.log(x)
    return (x ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _u_prime_crra(c, gamma, xp):
    c = _clip_pos(xp, c)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)


def terminal_loss_utility(W_T, F_target, cfg):
    """
    L_T^u = [ b(F) - b(W_T) ]_+  (utility level)
    또는 currency L_T = [F - W_T]_+ 에 u'(c̄) 곱해 utility로 변환.
    반환 타입: 입력 타입과 동일(np.ndarray or torch.Tensor)
    """
    if _is_torch(W_T):
        xp = torch
        relu = torch.relu
        F = torch.as_tensor(F_target, dtype=W_T.dtype, device=W_T.device)
    else:
        xp = np
        relu = lambda z: np.maximum(z, 0.0)
        F = float(F_target)

    if getattr(cfg, "cvar_unit", "utility") == "utility":
        # 기본: b(.) = u(.)
        bF = _u_crra(F,   cfg.crra_gamma, xp)
        bW = _u_crra(W_T, cfg.crra_gamma, xp)
        L_u = relu(bF - bW)
    else:
        # currency → utility 변환: u'(c̄) * [F - W_T]_+
        L_cur = relu(F - W_T)
        if getattr(cfg, "uprime_cbar_mode", "annuity") == "annuity":
            c_bar = getattr(cfg, "cstar_m", 0.0033333333333333335) * F  # 월 정액
        else:
            c_bar = getattr(cfg, "uprime_cbar_value", 1.0)
            if _is_torch(W_T):
                c_bar = torch.as_tensor(c_bar, dtype=W_T.dtype, device=W_T.device)
        upr = _u_prime_crra(c_bar, cfg.crra_gamma, xp)
        L_u = upr * L_cur

    # 보상과 동일 스케일
    return (getattr(cfg, "u_scale", 1.0)) * L_u
