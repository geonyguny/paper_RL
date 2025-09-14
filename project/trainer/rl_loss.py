# project/trainer/rl_loss.py
# -*- coding: utf-8 -*-
"""
A2C 손실 + 효용레벨 CVaR 듀얼 패널티(터미널) 모듈
- 목적함수 축을 '효용(utility) 레벨'로 통일
- terminal_loss_utility()를 통해 L_T^u 계산
- RU(Rockafellar–Uryasev) 듀얼 변수 η 업데이트(EMA/SGD 하이브리드)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from project.env.reward import terminal_loss_utility


# ---------- 유틸 ----------

def _is_tensor(x) -> bool:
    return _HAS_TORCH and hasattr(x, "device")


def _to_torch(x, like):
    if _is_tensor(like):
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return x


# ---------- CVaR 듀얼 상태 ----------

@dataclass
class CVaRDualState:
    """CVaR 듀얼 변수 상태(η) 및 하이퍼파라미터"""
    eta: float = 0.0                 # 듀얼 변수 η (효용 단위)
    eta_lr: float = 0.05             # η 업데이트 스텝 (SGD 해석)
    eta_ema: float = 0.90            # η EMA blending (0~1, 1에 가까울수록 관성 큼)
    alpha: float = 0.95              # CVaR 신뢰수준
    last_R_hat: float = 0.0          # 최근 배치의 R(η) 추정값
    last_p_hat: float = 0.0          # 최근 배치에서 P(L_u > η) 추정

    def to_dict(self) -> Dict[str, float]:
        return {
            "eta": float(self.eta),
            "R_hat": float(self.last_R_hat),
            "p_tail": float(self.last_p_hat),
        }


# ---------- CVaR 듀얼 패널티 ----------

def cvar_dual_penalty(Lu_batch, eta, alpha):
    """
    R(η) = η + 1/(1-α) * E[(L_u - η)_+]
    입력: np.ndarray 또는 torch.Tensor
    출력: Lu와 동일한 프레임워크 스칼라
    """
    if _is_tensor(Lu_batch):
        eta_t = torch.as_tensor(eta, dtype=Lu_batch.dtype, device=Lu_batch.device)
        gap = torch.relu(Lu_batch - eta_t)
        return eta_t + (1.0 / (1.0 - alpha)) * gap.mean()
    else:
        gap = np.maximum(Lu_batch - eta, 0.0)
        return eta + (1.0 / (1.0 - alpha)) * np.mean(gap)


def update_eta(state: CVaRDualState, L_u) -> None:
    """
    ∂R/∂η = 1 - P(L_u > η)/(1-α)  를 이용한 SGD 업데이트 + EMA 블렌딩.
    """
    if _is_tensor(L_u):
        p_hat = (L_u > state.eta).float().mean().item()
    else:
        p_hat = float((L_u > state.eta).mean())

    grad = 1.0 - (p_hat / (1.0 - state.alpha))  # 최소화 방향
    eta_new = max(0.0, state.eta - state.eta_lr * grad)
    # EMA 블렌딩
    state.eta = state.eta_ema * state.eta + (1.0 - state.eta_ema) * eta_new
    state.last_p_hat = p_hat
    # R_hat은 호출 측에서 기록


def compute_terminal_cvar_penalty(
    W_T_batch,
    sum_step_rewards,
    cfg,
    state: CVaRDualState,
) -> Tuple:
    """
    터미널 효용손실 기반 CVaR 듀얼 패널티 계산 및 목적함수 결합.
    반환:
        objective, penalty, R_hat, L_u
        (Torch면 torch scalar, Numpy면 float 반환)
    """
    # 1) 터미널 효용손실 L_u 계산 (u_scale 내장)
    L_u = terminal_loss_utility(W_T_batch, cfg.F_target, cfg)

    # 2) 듀얼 패널티 R(η)
    R_hat = cvar_dual_penalty(L_u, state.eta, state.alpha)
    state.last_R_hat = float(R_hat.item() if _is_tensor(R_hat) else R_hat)

    # 3) η 업데이트
    update_eta(state, L_u)

    # 4) 최종 패널티 및 목적함수
    if _is_tensor(R_hat):
        penalty = cfg.lambda_term * R_hat
        objective = sum_step_rewards - penalty
    else:
        penalty = cfg.lambda_term * R_hat
        objective = float(sum_step_rewards) - float(penalty)

    return objective, penalty, R_hat, L_u


# ---------- A2C 손실(옵션: 트레이너에서 활용) ----------

def gae_advantages(
    rewards,
    values,
    dones,
    last_value,
    gamma: float,
    gae_lambda: float,
):
    """
    GAE(λ) 계산. 입력은 [T,B] 텐서 또는 넘파이.
    반환: advantages[T,B], returns[T,B]
    """
    xp = torch if _is_tensor(values) else np
    T = rewards.shape[0]
    B = rewards.shape[1]

    adv = xp.zeros_like(values)
    lastgaelam = xp.zeros((B,), dtype=values.dtype)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - (dones[t].float() if _is_tensor(dones) else dones[t].astype(values.dtype))
        next_value = values[t + 1] if t < T - 1 else (last_value)
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        adv[t] = lastgaelam

    returns = adv + values
    return adv, returns


def a2c_losses(
    logprob,           # [T,B]
    values,            # [T,B]
    entropy,           # [T,B] or [T,B,heads] -> 합산
    advantages,        # [T,B]
    returns,           # [T,B]
    entropy_coef: float,
    value_coef: float,
) -> Tuple:
    """
    표준 A2C 손실: policy, value, entropy.
    """
    xp = torch if _is_tensor(values) else np

    # 정책 손실
    pg_loss = - (logprob * advantages.detach()).mean() if _is_tensor(values) else -float(np.mean(logprob * advantages))

    # 가치함수 손실(MSE)
    if _is_tensor(values):
        v_loss = 0.5 * (returns.detach() - values).pow(2).mean()
    else:
        v_loss = 0.5 * float(np.mean((returns - values) ** 2))

    # 엔트로피 보너스
    if _is_tensor(entropy):
        if entropy.dim() > 2:
            ent = entropy.sum(-1)
        else:
            ent = entropy
        ent_bonus = entropy_coef * ent.mean()
        total = pg_loss + value_coef * v_loss - ent_bonus
    else:
        ent = entropy if entropy.ndim == 2 else entropy.sum(-1)
        ent_bonus = entropy_coef * float(np.mean(ent))
        total = pg_loss + value_coef * v_loss - ent_bonus

    return total, pg_loss, v_loss, ent_bonus


# ---------- 통합 헬퍼 ----------

def build_objective_with_cvar(
    *,
    sum_step_rewards,   # 전체 에피소드(또는 배치) 누적 보상 (torch scalar or float)
    W_T_batch,          # [B] 터미널 부
    cfg,
    state: CVaRDualState,
):
    """
    터미널 효용 CVaR 듀얼 패널티를 목적함수에 결합하여 반환.
    """
    objective, penalty, R_hat, L_u = compute_terminal_cvar_penalty(
        W_T_batch=W_T_batch,
        sum_step_rewards=sum_step_rewards,
        cfg=cfg,
        state=state,
    )
    stats = {
        "cvar_eta": state.eta,
        "cvar_R_hat": float(R_hat.item() if _is_tensor(R_hat) else R_hat),
        "cvar_penalty": float(penalty.item() if _is_tensor(penalty) else penalty),
    }
    return objective, stats, L_u
