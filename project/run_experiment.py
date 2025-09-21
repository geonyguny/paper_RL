# run_experiment.py 내 교체
import datetime
import os  # <- 추가
import csv

def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def append_metrics_csv(path: str, payload: dict):
    # metrics & args 뽑기
    m   = payload.get('metrics') or {}
    arg = payload.get('args') or {}

    # 한 행(row) 구성: 기존 + 소비/연금 컬럼
    row = {
        'ts': now_iso(),
        'asset': payload.get('asset'),
        'method': payload.get('method'),
        'lambda': payload.get('lambda_term'),
        'F_target': payload.get('F_target'),
        'alpha': payload.get('alpha'),

        'ES95': m.get('ES95'),
        'EW': m.get('EW'),
        'EL': m.get('EL'),
        'Ruin': m.get('Ruin'),
        'mean_WT': m.get('mean_WT'),

        'HedgeHit': m.get('HedgeHit'),
        'HedgeKMean': m.get('HedgeKMean'),
        'HedgeActiveW': m.get('HedgeActiveW'),

        'fee_annual': payload.get('fee_annual'),
        'w_max': payload.get('w_max'),
        'horizon_years': payload.get('horizon_years'),
        'seeds': arg.get('seeds'),
        'n_paths': arg.get('n_paths'),
        'mortality_on': (arg.get('mortality') == 'on'),
        'market_mode': arg.get('market_mode'),

        # --- 소비 지표(신규) ---
        'p10_c_last': m.get('p10_c_last'),
        'p50_c_last': m.get('p50_c_last'),
        'p90_c_last': m.get('p90_c_last'),
        'C_ES95_avg': m.get('C_ES95_avg'),

        # --- 연금 오버레이(신규) ---
        'ann_on': arg.get('ann_on'),
        'ann_alpha': arg.get('ann_alpha'),
        'ann_L': arg.get('ann_L'),
        'ann_d': arg.get('ann_d'),
        'ann_index': arg.get('ann_index'),
        'y_ann': m.get('y_ann'),
        'a_factor': m.get('a_factor'),
        'P': m.get('P'),
    }

    # 헤더 결정: 파일이 있으면 기존 헤더 재사용(형식 충돌 방지), 없으면 새 헤더로 생성
    fieldnames = list(row.keys())
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        try:
            # 기존 헤더 읽기
            with open(path, 'r', encoding='utf-8') as rf:
                import csv as _csv
                r = _csv.reader(rf)
                old_header = next(r)
                if old_header:  # 기존 헤더가 있으면 그대로 사용
                    fieldnames = old_header
        except Exception:
            # 문제가 있으면 이번 행의 키를 헤더로 사용 (다음 줄이 흐트러질 수 있으므로 새 폴더 권장)
            fieldnames = list(row.keys())

    # 기존 헤더에 없는 키는 쓰지 않도록 필터(기존 파일에 안전하게 append)
    safe_row = {k: row.get(k) for k in fieldnames}

    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(safe_row)

def main():
    # 위임 엔트리: 기존 스크립트 호출 호환 유지
    from .runner.cli import main as _main
    _main()

if __name__ == "__main__":
    main()