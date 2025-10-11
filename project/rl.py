# project/rl.py
from __future__ import annotations

from typing import Any, Tuple
import numpy as _np
from .config import SimConfig


def _parse_state(state: Any, cfg: SimConfig) -> Tuple[float, float, int, int]:
    """
    관측(state)을 안전하게 파싱한다.
    지원 형태
      - dict: {'t': int, 'W': float} (+선택적으로 'T')
      - array-like: [t_norm, W]  (t_norm in [0,1], W: wealth)
    반환: (q_floor, W, t, T)
    """
    T = int(cfg.horizon_years) * int(cfg.steps_per_year)

    # dict 형태
    if isinstance(state, dict):
        W = float(state.get("W", 0.0))
        t = int(state.get("t", 0))
        return float(getattr(cfg, "q_floor", 0.0) or 0.0), W, t, T

    # 배열/시퀀스 형태 [t_norm, W]
    try:
        s = list(state)  # numpy array 등 시퀀스 지원
        if len(s) >= 2:
            t_norm = float(s[0])
            W = float(s[1])
            # t_norm은 0~1 사이 정규화된 시간(환경 설계에 따름)
            # T-1로 스케일링한 뒤 반올림하여 정수 스텝으로 복원
            t = int(round(t_norm * max(T - 1, 1)))
            t = max(0, min(t, max(T - 1, 0)))
            return float(getattr(cfg, "q_floor", 0.0) or 0.0), W, t, T
    except Exception:
        pass

    # 알 수 없는 형태 → 보수적 기본값
    return float(getattr(cfg, "q_floor", 0.0) or 0.0), 0.0, 0, T


class BetaPolicy:
    """
    매우 가벼운 휴리스틱 정책:
      - 소비률 q: 기본 2%, 바닥소비가 있으면 q >= f_min_real / W
      - 위험배분 w: 연령/잔여기간을 proxy한 age_factor로 선형 완충
    관측 입력이 dict/배열 모두에서 동작.
    """
    def __init__(self, cfg: SimConfig, hidden: int = 64):
        self.cfg = cfg
        self.hidden = int(hidden)

    def act(self, state: Any) -> Tuple[float, float]:
        q_floor, W, t, T = _parse_state(state, self.cfg)

        # 소비율 q (바닥소비 고려)
        q = 0.02
        if getattr(self.cfg, "floor_on", False) and float(getattr(self.cfg, "f_min_real", 0.0)) > 0.0:
            if W > 0.0:
                q = max(q, min(1.0, float(self.cfg.f_min_real) / max(W, 1e-9)))
            else:
                q = 1.0  # 자산이 0 이하이면 가능한 범위에서 소비 집행(보수적 처리)
        q = float(max(q_floor, min(1.0, max(0.0, q))))

        # 위험배분 w (잔여기간 기반 완충)
        age_factor = 1.0 - (float(t) / float(max(T, 1)))
        w_base = 0.5 * age_factor + 0.2
        w = float(min(float(self.cfg.w_max), max(0.0, w_base)))

        return q, w


class A2CTrainer:
    """
    A2C(스텁)의 트레이너 인터페이스 유지:
      - 전역 시드 변경 없이, 전달된 seed로만 내부 RNG를 준비(재현성 확보)
      - 여기서는 간단히 휴리스틱 정책을 반환 (학습 로직은 외부로 위임 가능)
    """
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.policy = BetaPolicy(cfg, hidden=getattr(cfg, "rl_hidden", 64))
        self._rng: _np.random.Generator | None = None

    def _init_rng(self, seed: int | None) -> None:
        # SeedSequence 기반 내부 RNG (전역 RNG에 영향 주지 않음)
        if seed is None:
            self._rng = _np.random.default_rng()
        else:
            ss = _np.random.SeedSequence(int(seed))
            self._rng = _np.random.default_rng(ss)

    def train(self, seed: int | None = None) -> BetaPolicy:
        """
        간단 스텁: seed가 주어지면 재현성을 가진 내부 RNG 초기화 후
        현재 BetaPolicy를 반환. (필요시 여기서 파라미터 샘플링/튜닝 가능)
        """
        self._init_rng(seed)
        # 예: self._rng 를 사용해서 하이퍼파라미터 탐색/초기화 등을 수행 가능
        # (지금은 가벼운 휴리스틱 정책이므로 바로 반환)
        return self.policy
