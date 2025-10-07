# project/trainer/policy_io.py
from __future__ import annotations
from typing import Any, Dict, Callable, Optional, Tuple
from types import SimpleNamespace

import torch

# 학습 시 사용한 것과 동일한 네트워크/어댑터 재사용
from .rl_a2c import PolicyNet, make_actor_from_policy


# -------------------------
# cfg shim (정규화/제약용)
# -------------------------
def _make_cfg_shim(
    cfg_hint: Optional[Any],
    ckpt_hints: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    evaluate()에서 actor가 동일한 정규화(T_ref, W0 등)를 쓰도록
    최소 속성만 가진 cfg shim을 만든다.
    - cfg_hint가 있으면 그대로 사용(최우선)
    - 없으면 ckpt_hints(구버전 저장본)로 채움
    - 둘 다 없으면 안전한 기본값으로 채움
    """
    if cfg_hint is not None:
        return cfg_hint

    hints = ckpt_hints or {}
    return SimpleNamespace(
        # evaluate()/PolicyNet.act에서 참조 가능한 안전 기본값들
        # T_ref = cfg.T or (horizon_years * steps_per_year)
        T=int(hints.get("T", 0)),
        horizon_years=int(hints.get("horizon_years", 35)),
        steps_per_year=int(hints.get("steps_per_year", 12)),
        W0=float(hints.get("W0", 1.0) or 1.0),
        q_floor=float(hints.get("q_floor", 0.0) or 0.0),
        w_max=float(hints.get("w_max", 1.0) or 1.0),
        rl_q_cap=float(hints.get("rl_q_cap", 0.0) or 0.0),
    )


# -------------------------
# ckpt 파싱/아키텍처 추론
# -------------------------
def _infer_arch_from_state_dict(sd: Dict[str, Any]) -> Tuple[int, int]:
    """
    state_dict에서 (obs_dim, hidden)을 유추.
    - 보통 첫 Linear 가중치 shape = [hidden, obs_dim]
    - 실패 시 (4, 128) 기본값
    """
    for k, v in sd.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            out_f, in_f = v.shape
            # backbone 첫 레이어일 가능성이 높음
            if in_f > 0 and out_f > 0:
                return int(in_f), int(out_f)
    return 4, 128


def _extract_state_dict_and_meta(ckpt: Any) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
    """
    체크포인트에서 (state_dict, arch, cfg_hints)를 최대한 유연하게 추출.
    지원 포맷:
      1) {"state_dict":..., "obs_dim": 4}
      2) {"state_dict":..., "arch": {"obs_dim": 4, "hidden": 128}, "cfg_hints": {...}}
      3) raw state_dict 자체(키가 layer.weight 형식) -> (obs_dim, hidden) 추론
    """
    if isinstance(ckpt, dict):
        # state_dict 찾기
        state_dict = ckpt.get("state_dict")
        if state_dict is None:
            # raw state_dict 또는 다른 키에 들어있을 수 있음
            if all(isinstance(k, str) for k in ckpt.keys()):
                # raw state_dict로 취급
                state_dict = ckpt
            else:
                for k in ("model_state", "policy_state", "policy", "model", "params"):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state_dict = ckpt[k]
                        break
        if state_dict is None:
            raise RuntimeError("Bad checkpoint: missing 'state_dict' (and not a raw state_dict)")

        # arch 처리
        arch = dict(ckpt.get("arch", {})) if isinstance(ckpt.get("arch"), dict) else {}
        if "obs_dim" not in arch:
            arch["obs_dim"] = int(ckpt.get("obs_dim", arch.get("obs_dim", 0) or 0))
        if not arch.get("obs_dim"):
            od, hd = _infer_arch_from_state_dict(state_dict)
            arch["obs_dim"] = od
            arch.setdefault("hidden", hd)
        else:
            arch.setdefault("hidden", 128)

        # cfg_hints
        cfg_hints = ckpt.get("cfg_hints", {}) if isinstance(ckpt.get("cfg_hints"), dict) else {}

        return state_dict, arch, cfg_hints

    # 그 외 포맷은 비지원
    raise RuntimeError("Unsupported checkpoint format (expected dict or state_dict-like)")


# -------------------------
# Public API
# -------------------------
def load_policy_as_actor(
    ckpt_path: str,
    cfg_hint: Optional[Any] = None,
) -> Callable[[Dict[str, Any]], Tuple[float, float]]:
    """
    policy.pt 체크포인트에서 PolicyNet을 복원하고,
    evaluate()가 바로 사용할 수 있는 actor(state)->(q,w)를 반환.

    Args
    ----
    ckpt_path : str
        torch.save(...)로 저장한 체크포인트 경로.
        - 신형: {"state_dict":..., "obs_dim": <int>}
        - 구형: {"state_dict":..., "arch": {"obs_dim":..., "hidden":...}, "cfg_hints": {...}}
        - (호환) 원시 state_dict도 허용
    cfg_hint : Any, optional
        정규화/제약(q_floor, w_max, rl_q_cap, T_ref, W0)을 위한 cfg 객체.
        주어지면 그대로 사용. 없으면 ckpt 내 cfg_hints 또는 안전 기본값으로 shim 생성.

    Returns
    -------
    actor : Callable[[state], (q, w)]
        evaluate()에서 호출 가능한 경량 actor
    """
    # CPU로 안전 로드 → 이후 device로 이동 (CUDA 유무와 무관하게 호환성 ↑)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict, arch, cfg_hints = _extract_state_dict_and_meta(ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyNet(arch.get("obs_dim", 4), hidden=arch.get("hidden", 128)).to(device)
    # 키 불일치 허용(strict=False) → 구버전과의 호환성 ↑
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # cfg shim 준비 (없으면 hints/기본값으로)
    cfg = _make_cfg_shim(cfg_hint, cfg_hints)

    # rl_a2c의 동일 어댑터 사용 → 관측 전처리/제약 일관성 보장
    return make_actor_from_policy(net, cfg=cfg, device=device)


# alias (run.py가 load_actor를 찾을 수도 있음)
def load_actor(ckpt_path: str, cfg_hint: Optional[Any] = None):
    return load_policy_as_actor(ckpt_path, cfg_hint)
