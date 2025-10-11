# project/runner/eta_utils.py
from __future__ import annotations

import json
import os
import time
import math
from typing import Any, Dict, List, Optional, Tuple

# ===== 공용 포맷터 =====
def fmt_hms(sec: float) -> str:
    try:
        total = int(round(float(sec)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"

def parse_hms_to_seconds(hms: Optional[str]) -> Optional[float]:
    if not hms:
        return None
    s = str(hms).strip()
    try:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = map(int, parts)
            return float(h) * 3600 + float(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = map(int, parts)
            return float(m) * 60 + float(sec)
        return float(s)
    except Exception:
        return None

# ===== DB 경로/입출력 =====
def eta_db_path(args) -> str:
    default = os.path.join(getattr(args, "outputs", "./outputs"), ".eta_history.json")
    return getattr(args, "eta_db", default) or default

def eta_load_db(path: str) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # 음수/0초 등 비정상 기록 제거
                return [r for r in data if float(r.get("time_total_s", 0) or 0) > 0]
    except Exception:
        pass
    return []

def eta_save_db(path: str, rows: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 최근 500건만 유지
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows[-500:], f, ensure_ascii=False, indent=0)
    except Exception:
        pass

# ===== 시그니처/규모지표 =====
def _seeds_count(args) -> int:
    s = getattr(args, "seeds", [])
    if isinstance(s, int):
        return 1
    if isinstance(s, (list, tuple)):
        return max(1, len(s))
    return 1

def eta_signature(args) -> Dict[str, Any]:
    return {
        "method": getattr(args, "method", None),
        "market_mode": getattr(args, "market_mode", None),
        "data_profile": getattr(args, "data_profile", None),
        "asset": getattr(args, "asset", None),
        "es_mode": getattr(args, "es_mode", None),

        # 공통 규모
        "n_paths": getattr(args, "n_paths", None),
        "seeds": _seeds_count(args),

        # RL 규모
        "rl_epochs": getattr(args, "rl_epochs", None),
        "rl_steps_per_epoch": getattr(args, "rl_steps_per_epoch", None),
        "rl_n_paths_eval": getattr(args, "rl_n_paths_eval", None),

        # HJB 그리드
        "hjb_W_grid": getattr(args, "hjb_W_grid", None),
        "hjb_Nshock": getattr(args, "hjb_Nshock", None),
        "hjb_eta_n": getattr(args, "hjb_eta_n", None),
    }

# ===== 내부: 규모 계산 =====
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
        return v if v > 0 else default
    except Exception:
        return default

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v > 0 else default
    except Exception:
        return default

def _hjb_grid_work(sig: Dict[str, Any]) -> float:
    W = _safe_int(sig.get("hjb_W_grid"), 0)
    N = _safe_int(sig.get("hjb_Nshock"), 0)
    E = _safe_int(sig.get("hjb_eta_n"), 0)
    if W == 0 or N == 0 or E == 0:
        return 0.0
    # 근사 복잡도: 격자 * 충격 * 방정식 반복
    return float(W) * float(N) * float(E)

def _hjb_eval_work(sig: Dict[str, Any]) -> float:
    paths = _safe_int(sig.get("n_paths"), 0)
    seeds = _safe_int(sig.get("seeds"), 1)
    return float(paths * max(1, seeds))

def _rl_train_work(sig: Dict[str, Any]) -> float:
    epochs = _safe_int(sig.get("rl_epochs"), 0)
    steps  = _safe_int(sig.get("rl_steps_per_epoch"), 0)
    seeds  = _safe_int(sig.get("seeds"), 1)
    return float(epochs * steps * max(1, seeds))

def _rl_eval_work(sig: Dict[str, Any]) -> float:
    eval_paths = _safe_int(sig.get("rl_n_paths_eval"), 0)
    seeds = _safe_int(sig.get("seeds"), 1)
    return float(eval_paths * max(1, seeds))

# ===== 내부: 후보 필터/매칭 =====
def _hard_filter(sig: Dict[str, Any], r_sig: Dict[str, Any]) -> bool:
    # method는 반드시 동일
    if r_sig.get("method") != sig.get("method"):
        return False
    # 데이터 프로파일은 강선호 → 다만 후보가 부족하면 나중에 느슨해짐(아래 선택 로직에서 처리)
    return True

def _match_score(sig: Dict[str, Any], r_sig: Dict[str, Any]) -> int:
    # 가중치 점수: method(필수), data_profile/market_mode/asset/es_mode 일치 수
    score = 0
    score += 1 if r_sig.get("data_profile") == sig.get("data_profile") else 0
    score += 1 if r_sig.get("market_mode") == sig.get("market_mode") else 0
    score += 1 if r_sig.get("asset") == sig.get("asset") else 0
    score += 1 if r_sig.get("es_mode") == sig.get("es_mode") else 0
    return score

def _log_ratio(a: float, b: float) -> float:
    # a/b의 로그 절대값(대칭거리). 0/0 등은 0으로 처리
    if a <= 0 and b <= 0:
        return 0.0
    if a <= 0:
        a = 1.0
    if b <= 0:
        b = 1.0
    return abs(math.log(max(a, 1e-9) / max(b, 1e-9)))

def _distance(sig: Dict[str, Any], r_sig: Dict[str, Any]) -> float:
    method = sig.get("method")
    if method == "rl":
        # 학습/평가 work의 log-비율 거리 합
        d_train = _log_ratio(_rl_train_work(sig), _rl_train_work(r_sig))
        d_eval  = _log_ratio(_rl_eval_work(sig),  _rl_eval_work(r_sig))
        # 학습 비중↑
        return 0.8 * d_train + 0.2 * d_eval
    # hjb/rule
    d_grid = _log_ratio(_hjb_grid_work(sig), _hjb_grid_work(r_sig))
    d_eval = _log_ratio(_hjb_eval_work(sig), _hjb_eval_work(r_sig))
    # 계산(그리드) 0.6 + 평가(경로) 0.4
    return 0.6 * d_grid + 0.4 * d_eval

# ===== 내부: 시간 스케일 =====
def _scale_time(sig: Dict[str, Any], r_sig: Dict[str, Any], base_time: float) -> float:
    method = sig.get("method")
    if method == "rl":
        train = _rl_train_work(sig);     train_ref = max(1.0, _rl_train_work(r_sig))
        evalw = _rl_eval_work(sig);      eval_ref  = max(1.0, _rl_eval_work(r_sig))
        scale_train = train / train_ref
        scale_eval  = evalw / eval_ref
        scale = 0.8 * scale_train + 0.2 * scale_eval
    else:
        grid  = _hjb_grid_work(sig);     grid_ref  = max(1.0, _hjb_grid_work(r_sig))
        evalw = _hjb_eval_work(sig);     eval_ref  = max(1.0, _hjb_eval_work(r_sig))
        scale_grid = grid / grid_ref
        scale_eval = evalw / eval_ref
        scale = 0.6 * scale_grid + 0.4 * scale_eval
    # 과도 스케일 클램프(극단값 방지)
    scale = max(0.1, min(10.0, float(scale)))
    return max(1.0, float(base_time)) * scale

# ===== 예측 본체(KNN 가중 평균) =====
def predict_eta_from_history(args, db: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    sig = eta_signature(args)

    # 하드 필터(method 동일) + 최근 역순 스캔
    raw = [row for row in reversed(db) if _hard_filter(sig, row.get("sig", {}))]

    if not raw:
        return None, "no_history"

    # 1차: 동일 data_profile 선호, 없으면 전체에서 진행
    same_profile = [r for r in raw if r.get("sig", {}).get("data_profile") == sig.get("data_profile")]
    pool = same_profile if same_profile else raw

    # 점수(일치) & 거리(규모 유사) 동시 고려 → 상위 후보 추림
    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for r in pool:
        r_sig = r.get("sig", {})
        base_time = float(r.get("time_total_s", 0.0) or 0.0)
        if base_time <= 0:
            continue
        score = _match_score(sig, r_sig)  # 클수록 좋음
        dist = _distance(sig, r_sig)      # 작을수록 좋음
        scored.append((score, dist, r))

    if not scored:
        return None, "bad_history_base"

    # 정렬: 점수 내림차순 → 거리 오름차순 → 최신 우선은 pool이 이미 반영
    scored.sort(key=lambda x: (-x[0], x[1]))

    # KNN 가중 평균
    K = 5
    picked = scored[:K]
    weights = []
    etas = []
    for score, dist, r in picked:
        r_sig = r.get("sig", {})
        base_time = float(r.get("time_total_s", 0.0))
        # 유사도 가중치: 1/(1+dist), 점수 보정(1+score/4)
        w = (1.0 / (1.0 + float(dist))) * (1.0 + float(score) / 4.0)
        eta_scaled = _scale_time(sig, r_sig, base_time)
        weights.append(w)
        etas.append(eta_scaled)

    if not weights or sum(weights) <= 0:
        return None, "scale_error"

    eta = sum(w * e for w, e in zip(weights, etas)) / sum(weights)
    src = "history_rl_knn" if sig.get("method") == "rl" else "history_hjb_knn"
    return max(1.0, float(eta)), src

# ===== 기록 =====
def eta_record(args, elapsed_s: float) -> None:
    try:
        path = eta_db_path(args)
        db = eta_load_db(path)
        db.append({
            "ts": time.time(),
            "time_total_s": float(elapsed_s),
            "sig": eta_signature(args),
        })
        eta_save_db(path, db)
    except Exception:
        pass
