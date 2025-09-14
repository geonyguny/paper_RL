import os, csv, datetime, inspect
import numpy as _np
from typing import Callable, Tuple, Optional, Dict, Any
from .env import RetirementEnv

# =========================
# Core episode & metrics
# =========================

def _reset_env(env: RetirementEnv, seed: int) -> None:
    """
    reset(seed=...) 지원/미지원 환경 모두에서 안전하게 초기화.
    1) reset에 seed 파라미터가 있으면 사용
    2) 없으면 reset()으로 폴백
    3) set_seed가 있으면 보조적으로 호출 (있더라도 reset이 경로를 다시 뽑는다면 무해)
    """
    try:
        sig = inspect.signature(env.reset)
        if "seed" in sig.parameters:
            env.reset(seed=seed)
            return
    except (TypeError, ValueError):
        # 시그니처 점검 불가 시 아래 폴백
        pass

    # 폴백 경로
    try:
        env.reset()
    finally:
        if hasattr(env, "set_seed"):
            try:
                env.set_seed(seed)  # 있으면 보조적으로 호출
            except Exception:
                pass

def run_episode(
    env: RetirementEnv,
    actor: Callable[[_np.ndarray], Tuple[float, float]],
    seed: int = 0,
) -> Tuple[_np.ndarray, bool, dict[str, float]]:
    """
    한 에피소드 실행.
    - actor(state) -> (q, w)
    - RetirementEnv.step(...)는 (obs, reward, done, trunc, info) 5-튜플을 반환.
    - '조기 파산(ruin)'은 마지막 스텝 전에 W<=0 으로 끝난 경우로 판정.
    Returns:
        W_hist: np.ndarray, 월별 W 기록
        early_hit: bool, 조기 파산 발생 여부
        ep_stats: dict, 헤지 계측 요약(히트율/평균 k/발동 시 평균 w 등)
    """
    # 새 경로로 초기화(재현 가능한 시드)
    env.reset(seed=seed)

    W_hist: list[float] = []
    early_hit = False

    # hedge counters
    hedge_hits = 0
    hedge_k_sum = 0.0
    hedge_active_w_sum = 0.0

    for i in range(env.T):
        state = env._obs()
        q, w = actor(state)
        _, _, done, _, info = env.step(q=q, w=w)

        W_hist.append(env.W)

        # 조기 파산 판정: 마지막 이전 스텝에서 W<=0으로 종료
        if env.W <= 0.0 and i < env.T - 1:
            early_hit = True

        # hedge 계측
        if isinstance(info, dict) and info.get("hedge_active", False):
            hedge_hits += 1
            hedge_k_sum += float(info.get("hedge_k", 0.0))
            # info에 w가 포함되어 있지 않더라도 현재 행동의 w로 보완
            hedge_active_w_sum += float(info.get("w", w))

        if done:
            break

    N = len(W_hist)
    ep_stats = {
        "hedge_hits": float(hedge_hits),
        "steps": float(N),
        "hedge_k_sum": float(hedge_k_sum),
        "hedge_active_w_sum": float(hedge_active_w_sum),
    }
    return _np.array(W_hist, dtype=float), early_hit, ep_stats

def metrics_wealth(WT_samples: _np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
    EW = float(WT_samples.mean())
    q = _np.quantile(WT_samples, 1.0 - alpha)  # 5th pct of wealth
    tail = WT_samples[WT_samples <= q]
    ES_tail_mean = float(tail.mean()) if tail.size > 0 else float(q)
    return dict(EW=EW, ES95=ES_tail_mean)


def metrics_loss(WT_samples: _np.ndarray, F: float = 1.0, alpha: float = 0.95) -> Dict[str, float]:
    # Loss = shortfall vs target F at terminal
    L = _np.maximum(F - WT_samples, 0.0)
    EL = float(L.mean())
    qL = _np.quantile(L, alpha)  # 95th pct of loss
    tail = L[L >= qL]
    ES = float(tail.mean()) if tail.size > 0 else float(qL)
    return dict(EW=float(WT_samples.mean()), EL=EL, ES95=ES)

def evaluate(cfg: Any, actor, es_mode: str = "wealth") -> Dict[str, float]:
    env = RetirementEnv(cfg)
    WT: list[float] = []
    early_flags: list[bool] = []

    # hedge aggregate
    agg_hedge_hits = 0.0
    agg_steps = 0.0
    agg_k_sum = 0.0
    agg_active_w_sum = 0.0

    seeds = getattr(cfg, "seeds", [0])
    n_eval = int(getattr(cfg, "n_paths_eval", getattr(cfg, "rl_n_paths_eval", 1)))

    for sd in seeds:
        base = int(sd) * 100_000
        for k in range(n_eval):
            W_hist, early, ep_stats = run_episode(env, actor, seed=base + k)
            WT.append(W_hist[-1] if getattr(W_hist, "size", 0) > 0 else 0.0)
            early_flags.append(bool(early))

            # 누적 헤지 통계
            agg_hedge_hits     += float(ep_stats.get("hedge_hits", 0.0))
            agg_steps          += float(ep_stats.get("steps", 0.0))
            agg_k_sum          += float(ep_stats.get("hedge_k_sum", 0.0))
            agg_active_w_sum   += float(ep_stats.get("hedge_active_w_sum", 0.0))

    WT_arr    = _np.asarray(WT, dtype=float)
    early_arr = _np.asarray(early_flags, dtype=bool)

    if WT_arr.size == 0:
        # 안전 가드: 평가 샘플이 전혀 없을 때
        m = dict(EW=0.0, ES95=0.0, EL=0.0, mean_WT=0.0)
        ruin_rate = 0.0
    else:
        # 파산율: 조기 종료이거나 최종부가 0 이하인 경우
        ruin_rate = float(_np.mean(_np.logical_or(early_arr, WT_arr <= 0.0)))

        if es_mode == "loss":
            F = float(getattr(cfg, "F_target", 1.0) or 1.0)
            m = metrics_loss(WT_arr, F=F, alpha=float(getattr(cfg, "alpha", 0.95)))
            m["mean_WT"] = float(WT_arr.mean())
        else:
            m = metrics_wealth(WT_arr, alpha=float(getattr(cfg, "alpha", 0.95)))
            m["mean_WT"] = m["EW"]

    m["Ruin"] = ruin_rate

    # 헤지 계측 요약
    m["HedgeHit"] = float(agg_hedge_hits / agg_steps) if agg_steps > 0 else 0.0
    if agg_hedge_hits > 0:
        m["HedgeKMean"]   = float(agg_k_sum / agg_hedge_hits)
        m["HedgeActiveW"] = float(agg_active_w_sum / agg_hedge_hits)
    else:
        m["HedgeKMean"]   = 0.0
        m["HedgeActiveW"] = 0.0

    return m


# =========================
# PR-3: Autosave Hook
# =========================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def save_metrics_autocsv(metrics: dict, cfg: Any, outputs: Optional[str] = None) -> str:
    """
    Append one row of metrics & config into outputs/_logs/metrics.csv
    Fields kept compact for gate checks and quick frontiers.
    """
    out_dir = outputs or getattr(cfg, "outputs", "./outputs")
    logs_dir = os.path.join(out_dir, "_logs")
    _ensure_dir(logs_dir)
    csv_path = os.path.join(logs_dir, "metrics.csv")

    # config pulls
    method = getattr(cfg, "method", None)
    es_mode = getattr(cfg, "es_mode", None)
    hedge_str = str(getattr(cfg, "hedge", "off")).lower()
    hedge_on_bool = (hedge_str == "on")
    hedge_mode = getattr(cfg, "hedge_mode", None)

    header = [
        "ts", "asset", "method", "es_mode",
        "alpha", "lambda_term", "F_target",
        "EW", "ES95", "EL", "Ruin", "mean_WT",
        "HedgeHit", "HedgeKMean", "HedgeActiveW",
        "fee_annual", "w_max", "floor_on", "f_min_real",
        "hedge_on", "hedge_mode",
        "horizon_years", "steps_per_year",
        "seeds", "n_paths_eval", "tag",
    ]

    seeds_val = getattr(cfg, "seeds", None)
    seeds_str = None
    if seeds_val:
        try:
            seeds_str = ",".join(str(s) for s in seeds_val)
        except Exception:
            seeds_str = str(seeds_val)

    row = {
        "ts": _now_iso(),
        "asset": getattr(cfg, "asset", None),
        "method": method,
        "es_mode": es_mode,
        "alpha": getattr(cfg, "alpha", None),
        "lambda_term": getattr(cfg, "lambda_term", None),
        "F_target": getattr(cfg, "F_target", None),
        "EW": metrics.get("EW"),
        "ES95": metrics.get("ES95"),
        "EL": metrics.get("EL"),
        "Ruin": metrics.get("Ruin"),
        "mean_WT": metrics.get("mean_WT"),
        "HedgeHit": metrics.get("HedgeHit"),
        "HedgeKMean": metrics.get("HedgeKMean"),
        "HedgeActiveW": metrics.get("HedgeActiveW"),
        "fee_annual": getattr(cfg, "fee_annual", None),  # fix: phi_adval -> fee_annual
        "w_max": getattr(cfg, "w_max", None),
        "floor_on": getattr(cfg, "floor_on", None),
        "f_min_real": getattr(cfg, "f_min_real", None),
        "hedge_on": bool(hedge_on_bool),
        "hedge_mode": hedge_mode,
        "horizon_years": getattr(cfg, "horizon_years", None),
        "steps_per_year": getattr(cfg, "steps_per_year", None),
        "seeds": seeds_str,
        "n_paths_eval": getattr(cfg, "n_paths_eval", getattr(cfg, "rl_n_paths_eval", None)),
        "tag": getattr(cfg, "tag", None),
    }

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    return csv_path


# =========================
# (Optional) Frontiers
# =========================

def plot_frontier_from_csv(csv_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """
    EW–ES95 frontier를 metrics.csv에서 그려 저장.
    - matplotlib이 없으면 조용히 패스하고 None 리턴
    """
    try:
        import csv as _csv
        import matplotlib.pyplot as plt  # optional

        xs, ys = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                try:
                    ew = float(row.get("EW", "nan"))
                    es = float(row.get("ES95", "nan"))
                    if _np.isfinite(ew) and _np.isfinite(es):
                        xs.append(ew); ys.append(es)
                except Exception:
                    continue

        if not xs:
            return None

        plt.figure()
        plt.scatter(xs, ys, s=16)
        plt.xlabel("EW (Expected Terminal Wealth)")
        plt.ylabel("ES95")
        plt.title("EW–ES95 frontier (from metrics.csv)")

        if out_path is None:
            base = os.path.dirname(csv_path)
            out_path = os.path.join(base, "frontier_EW_ES.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception:
        return None
