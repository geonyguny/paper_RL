# project/data/loader.py
from __future__ import annotations
import os, time, json, hashlib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

REQUIRED_MIN = ["date", "ret_kr_eq", "cpi_kr", "rf_kr_nom"]

def _hash_key(path: str, asset: str, use_real_rf: str, window: Optional[str]) -> str:
    st = os.stat(path)
    key = {
        "path": os.path.abspath(path),
        "mtime": int(st.st_mtime),
        "size": st.st_size,
        "asset": asset,
        "use_real_rf": use_real_rf,
        "window": window or "",
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()

def _slice_window(df: pd.DataFrame, window: Optional[str]) -> pd.DataFrame:
    if not window:
        return df
    try:
        a, b = window.split(":")
        a = a.strip() or None
        b = b.strip() or None
        if a: df = df[df["date"] >= a]
        if b: df = df[df["date"] <= b]
        return df
    except Exception as e:
        raise ValueError(f"--data_window 형식 오류: '{window}' (예: 1999-01:2024-12)") from e

def load_market_csv(
    path: str,
    asset: str,
    use_real_rf: str = "on",
    data_window: Optional[str] = None,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    CSV 스키마 v1:
      필수: date, ret_kr_eq, cpi_kr, rf_kr_nom
      선택: ret_us_eq_usd, ret_gold_usd, usdkrw, ret_us_eq_krw, ret_gold_krw, rf_kr_real
    - 수익률은 월간 단순수익률(0.01=+1%)
    - CPI는 지수, 인플레율 = cpi_t/cpi_{t-1}-1
    - KRW 환산: (1+r_usd)*(usdkrw_t/usdkrw_{t-1}) - 1
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"market_csv not found: {path}")

    cache_dir = os.path.join(os.path.dirname(path), "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _hash_key(path, asset, use_real_rf, data_window)
    cache_npz = os.path.join(cache_dir, f"{cache_key}.npz")

    if cache and os.path.exists(cache_npz):
        z = np.load(cache_npz, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        # np.load가 object 배열을 넣을 수 있어 dates 처리
        if "dates" in out and out["dates"].dtype.kind == "O":
            out["dates"] = out["dates"].astype(str)
        return out  # type: ignore

    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # 필수 헤더 확인
    for c in REQUIRED_MIN:
        if c not in df.columns:
            raise ValueError(f"CSV 누락컬럼: '{c}' (필수: {REQUIRED_MIN})")

    # 날짜 정렬/표준화
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)

    # 기간 슬라이스
    df = _slice_window(df, data_window)
    if len(df) < 24:
        raise ValueError(f"데이터 구간이 짧습니다(>=24 필요). window={data_window}, len={len(df)}")

    # USD→KRW 환산 수익률 생성(없는 경우)
    if "usdkrw" in df.columns:
        usdkrw = df["usdkrw"].values.astype(float)
        fx_ret = np.empty_like(usdkrw, dtype=float); fx_ret[:] = np.nan
        fx_ret[1:] = (usdkrw[1:] / usdkrw[:-1]) - 1.0
    else:
        usdkrw = None
        fx_ret = None

    def _to_krw(ret_usd_col: str, ret_krw_col: str) -> np.ndarray:
        if ret_krw_col in df.columns:
            return df[ret_krw_col].values.astype(float)
        if ret_usd_col in df.columns and fx_ret is not None:
            r_usd = df[ret_usd_col].values.astype(float)
            out = np.empty_like(r_usd); out[:] = np.nan
            out[1:] = (1.0 + r_usd[1:]) * (1.0 + fx_ret[1:]) - 1.0
            return out
        return np.full(len(df), np.nan, dtype=float)

    ret_kr_eq    = df["ret_kr_eq"].values.astype(float)
    ret_us_eq_krw  = _to_krw("ret_us_eq_usd", "ret_us_eq_krw")
    ret_gold_krw   = _to_krw("ret_gold_usd", "ret_gold_krw")

    # RF 실질/명목
    rf_nom = df["rf_kr_nom"].values.astype(float)
    if "rf_kr_real" in df.columns:
        rf_real = df["rf_kr_real"].values.astype(float)
    else:
        # CPI 기반 근사
        cpi = df["cpi_kr"].values.astype(float)
        inf = np.empty_like(cpi); inf[:] = np.nan
        inf[1:] = (cpi[1:] / cpi[:-1]) - 1.0
        rf_real = rf_nom - np.nan_to_num(inf, nan=0.0)

    # 자산 선택
    asset = asset.upper().strip()
    if asset == "KR":
        ret_asset = ret_kr_eq
    elif asset == "US":
        if np.all(np.isnan(ret_us_eq_krw)):
            raise ValueError("US 수익률 산출 불가: ret_us_eq_usd/ret_us_eq_krw/usdkrw 중 최소 조합 필요")
        ret_asset = ret_us_eq_krw
    elif asset in ("GOLD", "XAU"):
        if np.all(np.isnan(ret_gold_krw)):
            raise ValueError("Gold 수익률 산출 불가: ret_gold_usd/ret_gold_krw/usdkrw 중 최소 조합 필요")
        ret_asset = ret_gold_krw
    else:
        raise ValueError(f"알 수 없는 asset: {asset} (KR|US|GOLD)")

    out = {
        "dates": df["date"].values.astype(str),
        "ret_asset": ret_asset,
        "ret_kr_eq": ret_kr_eq,
        "ret_us_eq_krw": ret_us_eq_krw,
        "ret_gold_krw": ret_gold_krw,
        "rf_nom": rf_nom,
        "rf_real": rf_real,
        "cpi": df["cpi_kr"].values.astype(float),
    }
    if cache:
        np.savez_compressed(cache_npz, **out)
    return out
