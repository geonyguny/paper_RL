# project/data/market/make_market_csv.py
import csv, argparse, math, random
from datetime import date

def add_months(d, m):
    y = d.year + (d.month - 1 + m) // 12
    mo = (d.month - 1 + m) % 12 + 1
    return date(y, mo, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mu_eq_yr", type=float, default=0.06)   # 주식 연평균
    ap.add_argument("--sigma_eq_yr", type=float, default=0.18) # 주식 연변동성
    ap.add_argument("--rf_yr", type=float, default=0.02)       # 무위험 연이자
    ap.add_argument("--infl_yr", type=float, default=0.02)     # 연 인플레이션
    args = ap.parse_args()

    random.seed(args.seed)
    mu_m    = (1.0 + args.mu_eq_yr)**(1.0/12.0) - 1.0
    rf_m    = (1.0 + args.rf_yr)**(1.0/12.0) - 1.0
    infl_m  = (1.0 + args.infl_yr)**(1.0/12.0) - 1.0
    sigma_m = args.sigma_eq_yr / math.sqrt(12.0)

    t0 = date(2010, 1, 1)
    cpi = 100.0  # 수준지수 시작값

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 핵심 컬럼: risky_nom, tbill_nom, infl, cpi
        # 일부 코드와 호환 위해 alias(safe_nom/ret_risky/ret_safe)도 같이 씁니다.
        w.writerow(["date","risky_nom","tbill_nom","infl","cpi","safe_nom","ret_risky","ret_safe"])
        for i in range(args.months):
            d = add_months(t0, i)
            r_risky = random.gauss(mu_m, sigma_m)
            r_safe  = rf_m
            r_infl  = infl_m  # 단순 상수; 필요하면 잡음 추가 가능
            cpi *= (1.0 + r_infl)

            w.writerow([f"{d.year:04d}-{d.month:02d}",
                        f"{r_risky:.6f}",
                        f"{r_safe:.6f}",   # tbill_nom
                        f"{r_infl:.6f}",   # infl (월 물가상승률)
                        f"{cpi:.6f}",      # cpi (수준지수)
                        f"{r_safe:.6f}",   # safe_nom alias
                        f"{r_risky:.6f}",  # ret_risky alias
                        f"{r_safe:.6f}"])  # ret_safe alias

if __name__ == "__main__":
    main()
