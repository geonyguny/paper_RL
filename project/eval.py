# project/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, csv, datetime, inspect
import numpy as _np
import pandas as pd
from typing import Callable, Tuple, Optional, Dict, Any

from .env import RetirementEnv


# =========================
# Constants (CSV header)
# =========================

_METRICS_HEADER: list[str] = [
    "ts", "asset", "method", "es_mode",
    "alpha", "lambda_term", "F_target",
    "EW", "ES95", "EL", "Ruin", "mean_WT",
    "HedgeHit", "HedgeKMean", "HedgeActiveW",
    "fee_annual", "w_max", "floor_on", "f_min_real",
    "hedge_on", "hedge_mode",
    "horizon_years", "steps_per_year",
    "seeds", "n_paths_eval", "tag",
    # consumption (bands & ES-like on consumption)
    "p10_c_last", "p50_c_last", "p90_c_last", "C_ES95_avg", "AlivePathRate",
    # annuity overlay (if present)
    "y_ann", "a_factor", "P",
    # convenience: sweep/overlay params persisted in logs
    "ann_alpha",
]


# =========================
# Helpers
# =========================

def _reset_env(env: RetirementEnv, seed: int) -> None:
    """
    reset(seed=...) 지원/미지원 환경 모두에서 안전하게 초기화.
    우선 reset(seed=...), 폴백으로 reset(), 보조적으로 set_seed(seed) 호출.
    """
    try:
        sig = inspect.signature(env.reset)
        if "seed" in sig.parameters:
            env.reset(seed=seed)
            return
    except (TypeError, ValueError):
        pass
    try:
        env.reset()
    finally:
        if hasattr(env, "set_seed"):
            try:
                env.set_seed(seed)
            except Exception:
                pass


def _has_life_table(env: RetirementEnv) -> bool:
    lt = getattr(env, "life_table", None)
    if isinstance(lt, pd.DataFrame) and not lt.empty:
        return True
    lt2 = getattr(env, "mort_table_df", None)
    return isinstance(lt2, pd.DataFrame) and not lt2.empty


def _get_life_table(env: RetirementEnv) -> Optional[pd.DataFrame]:
    lt = getattr(env, "life_table", None)
    if isinstance(lt, pd.DataFrame) and not lt.empty:
        return lt
    lt2 = getattr(env, "mort_table_df", None)
    if isinstance(lt2, pd.DataFrame) and not lt2.empty:
        return lt2
    return None


def _annual_q_from_row(row: pd.Series) -> float:
    """
    life_table 한 행에서 annual qx를 안전하게 꺼낸다.
    허용 컬럼: qx 또는 px/Px(생존확률).
    """
    if "qx" in row.index:
        q = float(row["qx"])
    elif "px" in row.index or "Px" in row.index:
        col = "px" if "px" in row.index else "Px"
        q = 1.0 - float(row[col])
    else:
        raise KeyError("life_table row must have 'qx' or 'px/Px'")
    if not _np.isfinite(q):
        q = 0.0
    return float(min(max(q, 0.0), 1.0 - 1e-12))


def _monthly_q_from_annual(q_annual: float, spm: int) -> float:
    """연간 q(사망확률) -> 월간 q_m 변환."""
    qm = 1.0 - (1.0 - float(q_annual)) ** (1.0 / max(spm, 1))
    return float(min(max(qm, 0.0), 1.0 - 1e-12))


def _sample_death_month(
    age0: int,
    spm: int,
    life_df: pd.DataFrame,
    rng: _np.random.Generator,
    max_extra_years: int = 60,
) -> Optional[int]:
    """
    생명표 기반 사망월 표본추출.
    - 반환: death_m (0-based, 해당 달 '이전'에 사망으로 간주) 또는 None(장수)
    """
    if life_df is None or life_df.empty:
        return None
    df = life_df.copy()
    df["age"] = df["age"].astype(int)
    df = df.sort_values("age").set_index("age")
    a_min, a_max = int(df.index.min()), int(df.index.max())

    age = int(age0)
    month = 0
    hard_cap = (max_extra_years + (a_max - age0) + 1) * spm

    while month < hard_cap:
        a_clamp = int(min(max(age, a_min), a_max))
        q_m = _monthly_q_from_annual(_annual_q_from_row(df.loc[a_clamp]), spm)
        if rng.random() < q_m:
            return month
        month += 1
        if (month % spm) == 0:
            age += 1
    return None


# =========================
# Core episode
# =========================

def run_episode(
    env: RetirementEnv,
    actor: Callable[[_np.ndarray], Tuple[float, float]],
    seed: int = 0,
) -> Tuple[_np.ndarray, _np.ndarray, bool, dict[str, float]]:
    """
    한 에피소드 실행(사망표본이 있으면 사망월에 조기종결).
    Returns:
        W_hist: 월별 W
        C_hist: 월별 소비
        early_hit: 조기 파산 여부(사망과 구분)
        ep_stats: 헤지 계측 요약
    """
    _reset_env(env, seed=seed)

    # mortality: death month sampling (optional)
    death_m: Optional[int] = None
    mort_on = str(getattr(env, "mortality", getattr(env, "mortality_on", "off"))).lower()
    if (mort_on == "on") and _has_life_table(env):
        spm = int(getattr(env, "steps_per_year", 12) or 12)
        age0 = int(getattr(env, "age0", 65) or 65)
        lt = _get_life_table(env)
        rng = _np.random.default_rng(int(seed) + 17)  # decoupled reproducible stream
        try:
            death_m = _sample_death_month(age0, spm, lt, rng)
        except Exception:
            death_m = None  # safe fallback

    W_hist: list[float] = []
    C_hist: list[float] = []
    early_hit = False

    # hedge counters
    hedge_hits = 0
    hedge_k_sum = 0.0
    hedge_active_w_sum = 0.0

    for i in range(env.T):
        # death reached: stop stepping; use last pre-death wealth as terminal
        if (death_m is not None) and (i >= death_m):
            break

        state = env._obs()
        q, w = actor(state)
        _, _, done, _, info = env.step(q=q, w=w)

        W_hist.append(env.W)
        C_hist.append(float((info or {}).get("consumption", 0.0)))

        # ruin (separate from death): W<=0 before last scheduled step
        if env.W <= 0.0 and i < env.T - 1:
            early_hit = True

        # hedge stats
        if isinstance(info, dict) and info.get("hedge_active", False):
            hedge_hits += 1
            hedge_k_sum += float(info.get("hedge_k", 0.0))
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
    return _np.array(W_hist, dtype=float), _np.array(C_hist, dtype=float), bool(early_hit), ep_stats


# =========================
# Metrics
# =========================

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


# =========================
# Evaluation (wealth/loss + consumption bands + mortality)
# =========================

def evaluate(cfg: Any, actor, es_mode: str = "wealth") -> Dict[str, float]:
    env = RetirementEnv(cfg)
    T = int(getattr(env, "T", 0))

    # (NEW) verify injected paths snapshot (quiet=off 때만 1회 출력)
    if str(getattr(cfg, "quiet", "on")).lower() != "on":
        def _cs(a):
            a = _np.asarray(a, dtype=float)
            return float(_np.nanmean(a[:16])) if a.size else float("nan")
        try:
            print(f"[eval] path_cs ret={_cs(getattr(env, 'path_risky', _np.array([], dtype=float))):.6f} "
                  f"rf={_cs(getattr(env, 'path_safe', _np.array([], dtype=float))):.6f}")
            pr = getattr(env, "path_risky", None)
            ps = getattr(env, "path_safe", None)
            if pr is not None and ps is not None:
                pr = _np.asarray(pr, dtype=float); ps = _np.asarray(ps, dtype=float)
                if pr.size >= 2 and ps.size >= 2:
                    print(f"[eval] head ret={pr[0]:.6f},{pr[1]:.6f} rf={ps[0]:.6f},{ps[1]:.6f}")
        except Exception:
            pass

    WT: list[float] = []
    early_flags: list[bool] = []

    # consumption series (NaN padded to T)
    C_all: list[_np.ndarray] = []

    # hedge aggregates
    agg_hedge_hits = 0.0
    agg_steps = 0.0
    agg_k_sum = 0.0
    agg_active_w_sum = 0.0

    seeds = getattr(cfg, "seeds", [0])
    n_eval = int(getattr(cfg, "n_paths_eval", getattr(cfg, "rl_n_paths_eval", 1)))

    for sd in seeds:
        base = int(sd) * 100_000
        for k in range(n_eval):
            W_hist, C_hist, early, ep_stats = run_episode(env, actor, seed=base + k)

            WT.append(W_hist[-1] if getattr(W_hist, "size", 0) > 0 else 0.0)
            early_flags.append(bool(early))

            if T <= 0:
                T = len(C_hist)
            row = _np.full(T, _np.nan, dtype=float)
            take = min(len(C_hist), T)
            if take > 0:
                row[:take] = C_hist[:take]
            C_all.append(row)

            agg_hedge_hits += float(ep_stats.get("hedge_hits", 0.0))
            agg_steps += float(ep_stats.get("steps", 0.0))
            agg_k_sum += float(ep_stats.get("hedge_k_sum", 0.0))
            agg_active_w_sum += float(ep_stats.get("hedge_active_w_sum", 0.0))

    WT_arr = _np.asarray(WT, dtype=float)
    early_arr = _np.asarray(early_flags, dtype=bool)

    # wealth/loss & ruin
    if WT_arr.size == 0:
        m = dict(EW=0.0, ES95=0.0, EL=0.0, mean_WT=0.0)
        ruin_rate = 0.0
    else:
        ruin_rate = float(_np.mean(_np.logical_or(early_arr, WT_arr <= 0.0)))
        if es_mode == "loss":
            F = float(getattr(cfg, "F_target", 1.0) or 1.0)
            m = metrics_loss(WT_arr, F=F, alpha=float(getattr(cfg, "alpha", 0.95)))
            m["mean_WT"] = float(WT_arr.mean())
        else:
            m = metrics_wealth(WT_arr, alpha=float(getattr(cfg, "alpha", 0.95)))
            m["mean_WT"] = m["EW"]
    m["Ruin"] = ruin_rate

    # hedge summary
    m["HedgeHit"] = float(agg_hedge_hits / agg_steps) if agg_steps > 0 else 0.0
    m["HedgeKMean"] = float(agg_k_sum / agg_hedge_hits) if agg_hedge_hits > 0 else 0.0
    m["HedgeActiveW"] = float(agg_active_w_sum / agg_hedge_hits) if agg_hedge_hits > 0 else 0.0

    # -----------------------------
    # Consumption: bands + ES-like
    # -----------------------------
    if len(C_all) > 0 and T > 0:
        C_mat = _np.vstack(C_all)  # (Npaths, T)

        # AlivePathRate: 소비가 한 번이라도 관측된 경로 비율
        alive_rate = float(_np.mean(~_np.all(_np.isnan(C_mat), axis=1)))
        m["AlivePathRate"] = alive_rate

        # (선택) 종료 이후 소비=0으로 간주하려면 다음 라인을 활성화
        # C_mat = _np.where(_np.isnan(C_mat), 0.0, C_mat)

        # 유효 열(전부 NaN이 아닌 시점)만 골라서 밴드 계산
        valid_cols = _np.where(~_np.all(_np.isnan(C_mat), axis=0))[0]
        if valid_cols.size > 0:
            last_idx = int(valid_cols[-1])

            # 유효 구간에 대해서만 백분위 계산
            p10_v = _np.nanpercentile(C_mat[:, valid_cols], 10, axis=0)
            p50_v = _np.nanpercentile(C_mat[:, valid_cols], 50, axis=0)
            p90_v = _np.nanpercentile(C_mat[:, valid_cols], 90, axis=0)

            # 마지막 유효 시점의 last 지표
            m["p10_c_last"] = float(_np.nanpercentile(C_mat[:, last_idx], 10))
            m["p50_c_last"] = float(_np.nanpercentile(C_mat[:, last_idx], 50))
            m["p90_c_last"] = float(_np.nanpercentile(C_mat[:, last_idx], 90))

            # ES-like on consumption: 경로별 평균소비의 하위 5% 분위수
            Cmean_paths = _np.nanmean(C_mat, axis=1)
            m["C_ES95_avg"] = float(_np.nanquantile(Cmean_paths, 0.05))

            # bands 저장은 --bands 토글(on일 때만 IO)
            if str(getattr(cfg, "bands", "on")).lower() == "on":
                bands_dir = os.path.join(getattr(cfg, "outputs", "./outputs"), "_bands")
                os.makedirs(bands_dir, exist_ok=True)
                bands = _np.full((3, T), _np.nan, dtype=float)
                bands[:, valid_cols] = _np.vstack([p10_v, p50_v, p90_v])
                pd.DataFrame({
                    "t": _np.arange(T, dtype=int),
                    "p10": bands[0], "p50": bands[1], "p90": bands[2]
                }).to_csv(os.path.join(bands_dir, "consumption_bands.csv"),
                         index=False, encoding="utf-8")
        else:
            # 모든 열이 NaN인 드문 케이스: 보수적으로 0 처리
            m.update({"p10_c_last": 0.0, "p50_c_last": 0.0, "p90_c_last": 0.0, "C_ES95_avg": 0.0})

    return m


# =========================
# Autosave (+ header auto-migration)
# =========================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _maybe_upgrade_header(csv_path: str, expected: list[str]) -> None:
    """
    기존 metrics.csv 헤더가 구버전이면 자동으로 백업 후 최신 헤더로 재작성.
    (기존 행은 유지, 새 컬럼은 공란으로 채움)
    """
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            first = f.readline().rstrip("\r\n")
        current = [c.strip() for c in first.split(",")] if first else []
        if current == expected:
            return

        # backup
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = f"{csv_path}.bak_{ts}"
        os.replace(csv_path, bak)

        # rewrite with expected header
        with open(bak, "r", encoding="utf-8", newline="") as fin, \
             open(csv_path, "w", encoding="utf-8", newline="") as fout:
            r = csv.DictReader(fin)
            w = csv.DictWriter(fout, fieldnames=expected)
            w.writeheader()
            for row in r:
                w.writerow({k: row.get(k, "") for k in expected})
        print(f"[metrics:migrate] header upgraded → {csv_path} (backup: {bak})")
    except Exception:
        # 어떤 문제든 조용히 패스 (append 시도는 아래에서 계속)
        pass


def save_metrics_autocsv(metrics: dict, cfg: Any, outputs: Optional[str] = None) -> str:
    """
    outputs/_logs/metrics.csv에 한 줄 append.
    - 최신 헤더와 다르면 자동으로 마이그레이션 수행(재발 방지).
    - 소비/연금/보조지표 컬럼 포함 → 논문 표/그림 바로 내보내기.
    """
    out_dir = outputs or getattr(cfg, "outputs", "./outputs")
    logs_dir = os.path.join(out_dir, "_logs")
    _ensure_dir(logs_dir)
    csv_path = os.path.join(logs_dir, "metrics.csv")

    # auto-migrate header if needed
    _maybe_upgrade_header(csv_path, _METRICS_HEADER)

    method = getattr(cfg, "method", None)
    es_mode = getattr(cfg, "es_mode", None)
    hedge_on = str(getattr(cfg, "hedge", "off")).lower() == "on"
    hedge_mode = getattr(cfg, "hedge_mode", None)

    seeds_val = getattr(cfg, "seeds", None)
    if seeds_val:
        try:
            seeds_str = ",".join(str(s) for s in seeds_val)
        except Exception:
            seeds_str = str(seeds_val)
    else:
        seeds_str = None

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
        "fee_annual": getattr(cfg, "fee_annual", None),
        "w_max": getattr(cfg, "w_max", None),
        "floor_on": getattr(cfg, "floor_on", None),
        "f_min_real": getattr(cfg, "f_min_real", None),
        "hedge_on": bool(hedge_on),
        "hedge_mode": hedge_mode,
        "horizon_years": getattr(cfg, "horizon_years", None),
        "steps_per_year": getattr(cfg, "steps_per_year", None),
        "seeds": seeds_str,
        "n_paths_eval": getattr(cfg, "n_paths_eval", getattr(cfg, "rl_n_paths_eval", None)),
        "tag": getattr(cfg, "tag", None),
        # consumption
        "p10_c_last": metrics.get("p10_c_last"),
        "p50_c_last": metrics.get("p50_c_last"),
        "p90_c_last": metrics.get("p90_c_last"),
        "C_ES95_avg": metrics.get("C_ES95_avg"),
        "AlivePathRate": metrics.get("AlivePathRate"),
        # annuity overlay
        "y_ann": metrics.get("y_ann"),
        "a_factor": metrics.get("a_factor"),
        "P": metrics.get("P"),
        # sweep/overlay params
        "ann_alpha": getattr(cfg, "ann_alpha", None),
    }

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_METRICS_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)

    return csv_path


# =========================
# (Optional) Frontier plot
# =========================

def plot_frontier_from_csv(csv_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """
    EW–ES95 frontier를 metrics.csv에서 그려 저장.
    matplotlib 미설치 시 None 반환.
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
