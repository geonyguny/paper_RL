import numpy as np
import csv, os

def monthly_prob(qx_annual: float) -> float:
    qx_annual = float(max(min(qx_annual, 1.0), 0.0))
    # 월 환산: 1 - (1 - qx)^(1/12)
    return 1.0 - (1.0 - qx_annual) ** (1.0 / 12.0)

def load_table(path_or_preset: str):
    """
    CSV 스키마: age,qx_annual  (age는 정수, 0~120)
    preset 문자열은 나중에 확장 가능. 현재는 파일 경로만 처리.
    """
    if path_or_preset is None:
        raise ValueError("mortality table path required when mortality_on=on")
    path = path_or_preset
    if not os.path.exists(path):
        raise FileNotFoundError(f"mortality table not found: {path}")
    ages = []
    qxa = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            age = int(row["age"])
            qx = float(row["qx_annual"])
            ages.append(age); qxa.append(qx)
    ages = np.asarray(ages, dtype=int)
    qxa  = np.asarray(qxa, dtype=float)
    # 0~120 범위로 정렬/보간
    order = np.argsort(ages)
    ages = ages[order]; qxa = qxa[order]
    age_grid = np.arange(0, 121, dtype=int)
    qxa_interp = np.interp(age_grid, ages, qxa)
    qxm = np.clip([monthly_prob(q) for q in qxa_interp], 0.0, 1.0)
    return age_grid, np.asarray(qxm, dtype=np.float64)

class MortalitySampler:
    """
    나이(정수 age)에 대한 월별 사망확률 qxm[age] 사용.
    """
    def __init__(self, table_path: str):
        self.ages, self.qxm = load_table(table_path)

    def p_death_month(self, age_years: float) -> float:
        a = int(np.clip(int(age_years), 0, len(self.qxm)-1))
        return float(self.qxm[a])
