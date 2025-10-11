# project/data/loader.py
from __future__ import annotations
import os, json, hashlib
from typing import Dict, Optional
import numpy as np
import pandas as pd

REQUIRED_MIN = ["date", "ret_kr_eq", "cpi_kr", "rf_kr_nom"]

# --------------------------
# Helpers
# --------------------------
def _hash_key(path: str, asset: str, use_real_rf: str, window: Optional[str]) -> str:
    st = os.stat(path)
    key = {
        "path": os.path.abspath(path),
        "mtime": int(st.st_mtime),
        "size": st.st_size,
        "asset": str(asset),
        "use_real_rf": str(use_real_rf),
        "window": window or "",
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()

def _slice_window(df: pd.DataFrame, window: Optional[str]) -> pd.DataFrame:
    if not window:
        return df
    try:
        a, b = window.split(":")
        a = a.strip() or None
        b = b.strip() or None
        if a:
            df = df[df["date"] >= a]
        if b:
            df = df[df["date"] <= b]
        return df
    except Exception as e:
        raise ValueError(f"--data_window 형식 오류: '{window}' (예: 1999-01:2024-12)") from e

def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """지수→월간률 변환, 이미 월간률이면 그대로."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    is_index_like = (np.nanmax(x) > 5.0) or (np.nanmedian(np.abs(x)) > 0.2)
    if is_index_like and x.size >= 2:
        r = np.empty_like(x, dtype=float)
        r[1:] = x[1:] / x[:-1] - 1.0
        r[0] = r[1] if x.size > 1 and np.isfinite(x[1]) else 0.0
        return r
    return x

# --------------------------
# Loader
# --------------------------
def load_market_csv(
    path: str,
    asset: str,
    use_real_rf: str = "on",
    data_window: Optional[str] = None,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    CSV 스키마 v1 (월간):
      필수: date, ret_kr_eq, cpi_kr, rf_kr_nom
      선택: ret_us_eq_usd, ret_gold_usd, usdkrw, ret_us_eq_krw, ret_gold_krw, rf_kr_real

    반환 키:
      dates(str[]), ret_asset, ret_kr_eq, ret_us_eq_krw, ret_gold_krw,
      rf_nom, rf_real, cpi,  (추가) ret_fx, ret_fx_usdkrw

    주의:
      - 수익률은 0.01 = +1%
      - CPI는 지수/률 모두 허용(자동 판별)
      - KRW 환산: (1+r_usd)*(1+fx) - 1
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
        # object 배열 방지
        if "dates" in out and out["dates"].dtype.kind == "O":
            out["dates"] = out["dates"].astype(str)
        return out  # type: ignore

    # --- read & normalize columns ---
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # 필수 헤더 확인
    for c in REQUIRED_MIN:
        if c not in df.columns:
            raise ValueError(f"CSV 누락컬럼: '{c}' (필수: {REQUIRED_MIN})")

    # 날짜 표준화/정렬
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)

    # 기간 슬라이스
    df = _slice_window(df, data_window)
    if len(df) < 24:
        raise ValueError(f"데이터 구간이 짧습니다(>=24 필요). window={data_window}, len={len(df)}")

    # --- FX 월수익률 (usdkrw) ---
    if "usdkrw" in df.columns:
        usdkrw = df["usdkrw"].values.astype(float)
        fx_ret = np.empty_like(usdkrw, dtype=float); fx_ret[:] = np.nan
        fx_ret[1:] = (usdkrw[1:] / usdkrw[:-1]) - 1.0
    else:
        usdkrw = None
        fx_ret = None

    # --- helper: USD → KRW 수익률 열 만들기/가져오기 ---
    def _to_krw(ret_usd_col: str, ret_krw_col: str) -> np.ndarray:
        if ret_krw_col in df.columns:
            return df[ret_krw_col].values.astype(float)
        if ret_usd_col in df.columns and fx_ret is not None:
            r_usd = df[ret_usd_col].values.astype(float)
            out = np.empty_like(r_usd); out[:] = np.nan
            out[1:] = (1.0 + r_usd[1:]) * (1.0 + fx_ret[1:]) - 1.0
            return out
        return np.full(len(df), np.nan, dtype=float)

    # --- risky legs ---
    ret_kr_eq      = df["ret_kr_eq"].values.astype(float)
    ret_us_eq_krw  = _to_krw("ret_us_eq_usd", "ret_us_eq_krw")
    ret_gold_krw   = _to_krw("ret_gold_usd", "ret_gold_krw")

    # --- CPI & RF ---
    cpi_col = df["cpi_kr"].values.astype(float)
    cpi_rate = _to_monthly_rate_like(cpi_col)            # CPI 월간률
    rf_nom = df["rf_kr_nom"].values.astype(float)
    if "rf_kr_real" in df.columns:
        rf_real = df["rf_kr_real"].values.astype(float)
    else:
        rf_real = rf_nom - np.nan_to_num(cpi_rate, nan=0.0)  # 실질 근사

    # --- 자산 선택 (레거시 호환용 ret_asset) ---
    asset_u = asset.upper().strip()
    if asset_u == "KR":
        ret_asset = ret_kr_eq
    elif asset_u == "US":
        if np.all(np.isnan(ret_us_eq_krw)):
            raise ValueError("US 수익률 산출 불가: ret_us_eq_usd/ret_us_eq_krw/usdkrw 중 최소 조합 필요")
        ret_asset = ret_us_eq_krw
    elif asset_u in ("GOLD", "XAU"):
        if np.all(np.isnan(ret_gold_krw)):
            raise ValueError("Gold 수익률 산출 불가: ret_gold_usd/ret_gold_krw/usdkrw 중 최소 조합 필요")
        ret_asset = ret_gold_krw
    else:
        raise ValueError(f"알 수 없는 asset: {asset} (KR|US|GOLD)")

    # --- 출력 사전(+ FX 리턴 포함) ---
    out = {
        "dates": df["date"].values.astype(str),
        "ret_asset": ret_asset.astype(float),
        "ret_kr_eq": ret_kr_eq.astype(float),
        "ret_us_eq_krw": ret_us_eq_krw.astype(float),
        "ret_gold_krw": ret_gold_krw.astype(float),
        "rf_nom": rf_nom.astype(float),
        "rf_real": rf_real.astype(float),
        "cpi": cpi_col.astype(float),          # 원본 CPI 지수 (필요 시 참조)
        "ret_fx": (fx_ret.astype(float) if fx_ret is not None else np.full(len(df), np.nan, dtype=float)),
        "ret_fx_usdkrw": (fx_ret.astype(float) if fx_ret is not None else np.full(len(df), np.nan, dtype=float)),
    }

    if cache:
        np.savez_compressed(cache_npz, **out)
    return out
