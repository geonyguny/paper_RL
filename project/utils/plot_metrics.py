# project/utils/plot_metrics.py
# usage (PowerShell 예시):
#   ".venv\Scripts\python.exe" "project\utils\plot_metrics.py" --in "outputs\paper_main\_logs\metrics.csv" --outdir "outputs\paper_main\reports" --alpha 0.95 --topn 20 --es_cap 0.35

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def safe_float(x):
    if pd.isna(x): return np.nan
    s = str(x)
    if s in ("False","True","sigma","null","NaN",""): return np.nan
    try:
        return float(s)
    except:
        return np.nan

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="metrics.csv 경로")
    ap.add_argument("--outdir", default="outputs/paper_main/reports", help="리포트 출력 폴더")
    ap.add_argument("--alpha", type=float, default=0.95, help="ES 신뢰수준 표기(기록용)")
    ap.add_argument("--topn", type=int, default=20, help="Top N 하이라이트")
    ap.add_argument("--es_cap", type=float, default=None, help="ES95 상한(필터)")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    df = pd.read_csv(args.inp)

    # 정리
    num_cols = ["EW","ES95","Ruin","mean_WT","cap","epochs","entropy","alpha","w_max","q_floor","phi_adval"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    # datetime 정렬
    if "datetime" in df.columns:
        df["datetime_parsed"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime_parsed")
    else:
        df["datetime_parsed"] = pd.NaT

    # ES 캡 필터(선택)
    if args.es_cap is not None and "ES95" in df.columns:
        df = df[df["ES95"].apply(lambda x: (not pd.isna(x)) and x <= args.es_cap)].copy()

    # Top by EW
    df["EW_rank"] = df["EW"].rank(ascending=False, method="first")
    top = df.nsmallest(args.topn, "EW_rank").copy()

    # 저장용 CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(outdir, f"metrics_top_{args.topn}_{ts}.csv")
    top_cols = [c for c in ["datetime","tag","method","EW","ES95","Ruin","mean_WT","cap","epochs","entropy","alpha","w_max","q_floor","phi_adval","seed","notes"] if c in df.columns]
    top[top_cols].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[report] Top{args.topn} 저장: {out_csv}")

    # 1) EW vs ES95 산포도
    if set(["EW","ES95"]).issubset(df.columns):
        plt.figure()
        plt.scatter(df["ES95"], df["EW"], s=14, alpha=0.6)
        plt.title("EW vs ES95 (lower left is safer)")
        plt.xlabel("ES95")
        plt.ylabel("EW")
        # Top N 하이라이트 (큰 마커)
        plt.scatter(top["ES95"], top["EW"], s=48, alpha=0.9)
        out_png = os.path.join(outdir, f"scatter_EW_ES95_{ts}.png")
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[report] 산포도 저장: {out_png}")

    # 2) 태그별 박스플롯 (EW)
    if {"tag","EW"}.issubset(df.columns):
        # 최근 실험 상위 K개 태그만
        recent_tags = df.dropna(subset=["tag"]).tail(5000)["tag"].value_counts().head(10).index.tolist()
        sub = df[df["tag"].isin(recent_tags)].copy()
        if not sub.empty:
            plt.figure()
            sub.boxplot(column="EW", by="tag", grid=False, rot=45)
            plt.suptitle("")
            plt.title("EW by Tag (10 recent tags)")
            plt.xlabel("tag")
            plt.ylabel("EW")
            out_png = os.path.join(outdir, f"box_EW_by_tag_{ts}.png")
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"[report] 박스플롯 저장: {out_png}")

    # 3) cap-entropy 격자 평균 EW 히트맵 유사 테이블(csv)
    have_cols = set(df.columns)
    if {"cap","entropy","EW"}.issubset(have_cols):
        grid = (df
                .dropna(subset=["cap","entropy","EW"])
                .groupby(["cap","entropy"], as_index=False)["EW"].mean()
               )
        grid_csv = os.path.join(outdir, f"grid_cap_entropy_EW_{ts}.csv")
        grid.to_csv(grid_csv, index=False, encoding="utf-8")
        print(f"[report] 격자 평균 EW 저장: {grid_csv}")

    # 4) 방법론(method)별 요약표
    if {"method","EW","ES95","Ruin"}.issubset(have_cols):
        summ = (df.groupby("method", as_index=False)
                  .agg(EW_mean=("EW","mean"),
                       ES95_mean=("ES95","mean"),
                       Ruin_mean=("Ruin","mean"),
                       count=("method","count")))
        summ_csv = os.path.join(outdir, f"summary_by_method_{ts}.csv")
        summ.to_csv(summ_csv, index=False, encoding="utf-8")
        print(f"[report] 방법론 요약 저장: {summ_csv}")

if __name__ == "__main__":
    main()
