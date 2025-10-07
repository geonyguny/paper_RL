# project/runner/kod_sweeps.py
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, List, Tuple, Optional

from .cli import _build_arg_parser  # 기본값 네임스페이스 생성용
from .run import run_once, run_rl
from ..eval import evaluate

# -------------------------------
# 유틸
# -------------------------------

def _ns_from_defaults(**overrides) -> Any:
    """cli의 기본값을 가져와 일부만 덮어쓴 argparse.Namespace 생성"""
    parser = _build_arg_parser()
    ns = parser.parse_args([])  # 모든 기본값
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    # dict field union
    fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def _run_point(ns) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    method에 따라 run_* 호출.
    RL은 return_actor=on으로 받아 evaluate()를 여기서 수행해 W_T 기반 ES 재계산.
    """
    if ns.method == "rl":
        ns.return_actor = "on"
        res = run_rl(ns)
        # run_rl이 (cfg, actor) 튜플을 돌려줄 수도 있고 dict일 수도 있음
        if isinstance(res, tuple) and len(res) >= 2:
            cfg, actor = res[0], res[1]
            m, extra = evaluate(cfg, actor, es_mode=ns.es_mode, return_paths=True)  # type: ignore
            return m, extra or {}
        elif isinstance(res, dict) and "metrics" in res:
            return res["metrics"], res.get("extra", {})
        else:
            # 최소 호환
            return (res if isinstance(res, dict) else {"result": "ok"}), {}
    else:
        res = run_once(ns)
        if isinstance(res, dict) and "metrics" in res:
            return res["metrics"], res.get("extra", {})
        return (res if isinstance(res, dict) else {"result": "ok"}), {}

# -------------------------------
# 실험 사양
# -------------------------------

@dataclass
class BasePreset:
    # 공통
    asset: str = "KR"
    market_mode: str = "bootstrap"   # "iid" | "bootstrap"
    data_profile: str = "full"       # csv 자동 바인딩
    bootstrap_block: int = 24
    use_real_rf: str = "on"
    es_mode: str = "loss"
    F_target: float = 0.60
    outputs: str = "./outputs"
    quiet: str = "on"

    # RL 기본
    method: str = "rl"
    rl_epochs: int = 5
    rl_steps_per_epoch: int = 1024
    rl_n_paths_eval: int = 500
    seeds: Iterable[int] = (0, 1, 2, 3, 4)

    # 제약/수수료/하이퍼
    w_max: float = 0.70
    fee_annual: float = 0.004
    q_floor: float = 0.02
    rl_q_cap: float = 0.0

    # 생존/성별/개시연령
    mortality: str = "on"
    sex: str = "M"
    age0: int = 55

# -------------------------------
# 1) 단일 변수 스윕 (OAT)
# -------------------------------

def sweep_oat(tag: str = "exp_main", preset: Optional[BasePreset] = None) -> None:
    pr = preset or BasePreset()
    base_out = os.path.join(pr.outputs, tag)
    _ensure_dir(base_out)

    rows: List[Dict[str, Any]] = []

    # 실험 축 (필요 시 수정)
    w_max_grid      = [round(x/10, 1) for x in range(0, 11)]         # 0.0~1.0
    q_floor_grid    = [0.015, 0.020, 0.025]
    fee_grid        = [0.0, 0.00207, 0.00414, 0.00828]
    w_fixed_grid    = [round(x/10, 1) for x in range(0, 11)]         # rule baseline용

    # 0) RL baseline (기본 preset)
    for s in pr.seeds:
        ns = _ns_from_defaults(**asdict(pr),
                               tag=f"{tag}_RL_M{pr.sex}_A{pr.age0}_seed{s}",
                               seeds=[int(s)],
                               return_actor="on",
                               bands="on")
        m, extra = _run_point(ns)
        rows.append({
            "group": "RL_baseline",
            "sex": pr.sex, "age0": pr.age0, "seed": s,
            "w_max": pr.w_max, "q_floor": pr.q_floor, "fee_annual": pr.fee_annual,
            **m
        })

    # 1) w_max 스윕
    for v in w_max_grid:
        ns = _ns_from_defaults(**asdict(pr),
                               tag=f"{tag}_RL_wmax_{v:.2f}",
                               w_max=float(v),
                               seeds=[0], return_actor="on")
        m, extra = _run_point(ns)
        rows.append({"group": "w_max", "w_max": v, **m})

    # 2) q_floor 스윕
    for v in q_floor_grid:
        ns = _ns_from_defaults(**asdict(pr),
                               tag=f"{tag}_RL_qfloor_{v:.4f}",
                               q_floor=float(v),
                               seeds=[0], return_actor="on")
        m, extra = _run_point(ns)
        rows.append({"group": "q_floor", "q_floor": v, **m})

    # 3) fee 스윕
    for v in fee_grid:
        ns = _ns_from_defaults(**asdict(pr),
                               tag=f"{tag}_RL_fee_{v:.5f}",
                               fee_annual=float(v),
                               seeds=[0], return_actor="on")
        m, extra = _run_point(ns)
        rows.append({"group": "fee", "fee_annual": v, **m})

    # 4) 고정 w 규칙 기반(rule) 비교: q=4%룰 근사 + 고정 w
    for wfix in w_fixed_grid:
        ns = _ns_from_defaults(**asdict(pr),
                               method="rule",
                               tag=f"{tag}_RULE_w_{wfix:.2f}",
                               w_fixed=float(wfix),
                               seeds=[0])
        m, extra = _run_point(ns)
        rows.append({"group": "rule_w_fixed", "w_fixed": wfix, **m})

    _write_csv(os.path.join(base_out, "oat_results.csv"), rows)


# -------------------------------
# 2) 2D 히트맵 스윕 (예: w_max × fee, w_max × q_floor)
# -------------------------------

def sweep_grid(tag: str = "exp_grid", preset: Optional[BasePreset] = None) -> None:
    pr = preset or BasePreset()
    base_out = os.path.join(pr.outputs, tag)
    _ensure_dir(base_out)

    rows: List[Dict[str, Any]] = []

    w_max_grid   = [round(x/10, 1) for x in range(0, 11)]
    fee_grid     = [0.0, 0.00207, 0.00414, 0.00828]
    q_floor_grid = [0.015, 0.020, 0.025]

    # A) (w_max, fee)
    for wmax in w_max_grid:
        for fee in fee_grid:
            ns = _ns_from_defaults(**asdict(pr),
                                   tag=f"{tag}_RL_wmax_fee_{wmax:.2f}_{fee:.5f}",
                                   w_max=float(wmax), fee_annual=float(fee),
                                   seeds=[0], return_actor="on")
            m, _ = _run_point(ns)
            rows.append({"grid": "wmax_fee", "w_max": wmax, "fee_annual": fee, **m})

    # B) (w_max, q_floor)
    for wmax in w_max_grid:
        for qf in q_floor_grid:
            ns = _ns_from_defaults(**asdict(pr),
                                   tag=f"{tag}_RL_wmax_qfloor_{wmax:.2f}_{qf:.4f}",
                                   w_max=float(wmax), q_floor=float(qf),
                                   seeds=[0], return_actor="on")
            m, _ = _run_point(ns)
            rows.append({"grid": "wmax_qfloor", "w_max": wmax, "q_floor": qf, **m})

    _write_csv(os.path.join(base_out, "grid_results.csv"), rows)


# -------------------------------
# 3) 개시연령/성별 비교
# -------------------------------

def sweep_age_sex(tag: str = "exp_age_sex", preset: Optional[BasePreset] = None) -> None:
    pr = preset or BasePreset()
    base_out = os.path.join(pr.outputs, tag)
    _ensure_dir(base_out)

    rows: List[Dict[str, Any]] = []
    for sex in ("M", "F"):
        for age0 in range(55, 66):  # 55~65
            ns = _ns_from_defaults(**asdict(pr),
                                   tag=f"{tag}_RL_{sex}{age0}",
                                   sex=sex, age0=int(age0),
                                   seeds=[0], return_actor="on")
            m, _ = _run_point(ns)
            rows.append({"group": "age_sex", "sex": sex, "age0": age0, **m})

    _write_csv(os.path.join(base_out, "age_sex_results.csv"), rows)


# -------------------------------
# (선택) 확장 실험 파라미터 자리
#  - θ(ann_alpha), h_FX, α(자산배분) 등은 cfg에 패스만 하고,
#    엔진 반영은 env/market 쪽 TODO.
# -------------------------------

def sweep_extended_placeholder(tag: str = "exp_extended", preset: Optional[BasePreset] = None) -> None:
    pr = preset or BasePreset()
    base_out = os.path.join(pr.outputs, tag)
    _ensure_dir(base_out)

    rows: List[Dict[str, Any]] = []

    # 예: θ(ann_alpha) 0~1, h_FX 0~1
    for theta in [round(x/10, 1) for x in range(0, 11)]:
        for hfx in [round(x/10, 1) for x in range(0, 11)]:
            ns = _ns_from_defaults(**asdict(pr),
                                   tag=f"{tag}_RL_theta_hfx_{theta:.1f}_{hfx:.1f}",
                                   ann_on="on" if theta > 0 else "off",
                                   ann_alpha=float(theta),
                                   # 환헤지 비율: 엔진 미반영. cfg에 전달만 함.
                                   h_fx=float(hfx),
                                   seeds=[0], return_actor="on")
            m, _ = _run_point(ns)
            rows.append({
                "grid": "theta_hfx",
                "theta": theta, "h_FX": hfx,
                "_note": "h_FX/θ는 env/market 구현 필요",
                **m
            })

    _write_csv(os.path.join(base_out, "extended_placeholder.csv"), rows)


# -------------------------------
# 진입점
# -------------------------------

def main():
    # 기본 프리셋: 남 55, bootstrap/full, ES(loss) 기준
    pr = BasePreset(seeds=(0,))  # 샘플 실행 속도 위해 seed 1개
    sweep_oat(tag="exp_main", preset=pr)
    sweep_grid(tag="exp_grid", preset=pr)
    sweep_age_sex(tag="exp_age", preset=pr)
    # 필요시: sweep_extended_placeholder()

if __name__ == "__main__":
    main()
