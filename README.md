# KR Decumulation Simulation (v2 — HJB 2D q,w)
- HJB now optimizes both q and w over small grids.

# Project Runner CLI — 사용법 & 전체 명령 예시

이 문서는 `project.runner.cli`의 실행 방법, 주요 옵션, 그리고 **실제로 복사해 실행할 수 있는 전체 명령어**를 정리합니다. Windows PowerShell과 Linux/macOS(Bash) 모두 예시를 제공합니다.

> 참고: `bootstrap` 시장 모드를 사용할 때는 `--market_csv` 또는 `--data_profile {dev,full}` 중 하나를 **반드시** 지정해야 합니다.

---

## 1) 환경 준비

### 1.1 가상환경 생성 및 의존성 설치

**Windows PowerShell**

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS (Bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1.2 의존성 고정(선택)

```powershell
python -m pip freeze > requirements.txt
```

---

## 2) 빠른 시작: RL 실행 (Bootstrap, CVaR-Loss 모드)

### 2.1 요약 출력(summary)

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode summary `
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" `
  --no_paths `
  --tag quick_summary
```

**Linux / macOS (Bash)**

```bash
python -m project.runner.cli \
  --method rl \
  --market_mode bootstrap \
  --data_profile full \
  --es_mode loss \
  --F_target 0.60 \
  --rl_q_cap 0.0042 \
  --rl_epochs 1 \
  --rl_steps_per_epoch 512 \
  --rl_n_paths_eval 200 \
  --seeds 0 \
  --alpha_mix equal \
  --h_FX 1 \
  --return_actor on \
  --print_mode summary \
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" \
  --no_paths \
  --tag quick_summary
```

### 2.2 메트릭스 전용 출력(metrics)

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode metrics `
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" `
  --tag quick_metrics
```

**Linux / macOS (Bash)**

```bash
python -m project.runner.cli \
  --method rl \
  --market_mode bootstrap \
  --data_profile full \
  --es_mode loss \
  --F_target 0.60 \
  --seeds 0 \
  --alpha_mix equal \
  --h_FX 1 \
  --return_actor on \
  --print_mode metrics \
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" \
  --tag quick_metrics
```

### 2.3 전체 출력(full) 및 JSON 저장

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode full `
  --tag full_output_example `
  > .\outputs\full_output_example.json
```

**Linux / macOS (Bash)**

```bash
python -m project.runner.cli \
  --method rl \
  --market_mode bootstrap \
  --data_profile full \
  --es_mode loss \
  --F_target 0.60 \
  --rl_q_cap 0.0042 \
  --rl_epochs 1 \
  --rl_steps_per_epoch 512 \
  --rl_n_paths_eval 200 \
  --seeds 0 \
  --alpha_mix equal \
  --h_FX 1 \
  --return_actor on \
  --print_mode full \
  --tag full_output_example \
  > ./outputs/full_output_example.json
```

---

## 3) 경로 점검 및 메트릭 추출

### 3.1 터미널 부(wealth) 경로 샘플 확인

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode full `
  --tag inspect_paths `
| ConvertFrom-Json `
| Select-Object -ExpandProperty extra `
| Select-Object -ExpandProperty eval_WT `
| Select-Object -First 10
```

### 3.2 경로로부터 요약 통계 구하기

**Windows PowerShell**

```powershell
$resp = python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode full `
  --tag inspect_stats `
| ConvertFrom-Json

$wt = $resp.extra.eval_WT
"n=$($wt.Count)"
$wt | Measure-Object -Minimum -Maximum -Average
```

### 3.3 메트릭을 CSV로 내보내기

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode metrics `
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" `
  --tag export_metrics `
| ConvertFrom-Json `
| Select-Object tag,asset,method,n_paths,EW,ES95,Ruin,mean_WT,es95_source `
| Export-Csv -NoTypeInformation -Encoding UTF8 -Path .\outputs\metrics_export.csv
```

**Linux / macOS (Bash, jq 필요)**

```bash
python -m project.runner.cli \
  --method rl \
  --market_mode bootstrap \
  --data_profile full \
  --es_mode loss \
  --F_target 0.60 \
  --rl_q_cap 0.0042 \
  --rl_epochs 1 \
  --rl_steps_per_epoch 512 \
  --rl_n_paths_eval 200 \
  --seeds 0 \
  --alpha_mix equal \
  --h_FX 1 \
  --return_actor on \
  --print_mode metrics \
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" \
  --tag export_metrics \
| jq -r '[.tag,.asset,.method,.n_paths,.EW,.ES95,.Ruin,.mean_WT,.es95_source] | @csv' \
> ./outputs/metrics_export.csv
```

---

## 4) CVaR(ES95) 계산 검증 예시

CLI는 경로 기반 손실 (L = \max(F - W_T, 0))에 대해 **보간 포함** 표본식(Acerbi–Tasche)을 적용합니다. 아래 명령으로 수동 검증이 가능합니다.

**Windows PowerShell**

```powershell
$resp = python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode full `
  --tag cvar_check_after_patch `
| ConvertFrom-Json

$wt    = $resp.extra.eval_WT
$F     = 0.60
$alpha = 0.95
$L     = $wt | ForEach-Object { [math]::Max($F - $_, 0) }

$Lasc  = $L | Sort-Object
$n     = $Lasc.Count
$j     = [math]::Floor($n * $alpha)
$theta = $n * $alpha - $j
$Lj1   = [double]$Lasc[$j]
$tail_sum = 0.0; for ($i = $j + 1; $i -lt $n; $i++) { $tail_sum += [double]$Lasc[$i] }
$ES95_interp = ((1.0 - $theta) * $Lj1 + $tail_sum) / ($n * (1.0 - $alpha))

"ES95(interp) = $ES95_interp"
"ES95(CLI)    = $($resp.metrics.ES95)"
```

**도우미 스크립트로 검증(선택)**

```powershell
python .\scripts\es95_check.py --mode run_cli --tag cvar_interp_check --F_target 0.60 --alpha 0.95
```

---

## 5) 여러 시드 합산 실행

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --rl_q_cap 0.0042 `
  --rl_epochs 1 `
  --rl_steps_per_epoch 512 `
  --rl_n_paths_eval 200 `
  --seeds 0 1 2 3 4 `
  --alpha_mix equal `
  --h_FX 1 `
  --return_actor on `
  --print_mode summary `
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" `
  --tag seeds_0to4
```

**Linux / macOS (Bash)**

```bash
python -m project.runner.cli \
  --method rl \
  --market_mode bootstrap \
  --data_profile full \
  --es_mode loss \
  --F_target 0.60 \
  --rl_q_cap 0.0042 \
  --rl_epochs 1 \
  --rl_steps_per_epoch 512 \
  --rl_n_paths_eval 200 \
  --seeds 0 1 2 3 4 \
  --alpha_mix equal \
  --h_FX 1 \
  --return_actor on \
  --print_mode summary \
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" \
  --tag seeds_0to4
```

---

## 6) HJB: CVaR 타깃으로 람다 캘리브레이션 예시

**Windows PowerShell**

```powershell
python -m project.runner.cli `
  --method hjb `
  --market_mode bootstrap `
  --data_profile dev `
  --es_mode loss `
  --alpha 0.95 `
  --cvar_target 0.40 `
  --cvar_tol 0.01 `
  --lambda_min 0.0 `
  --lambda_max 2.0 `
  --calib_fast on `
  --calib_max_iter 8 `
  --tag hjb_calibration_example
```

**Linux / macOS (Bash)**

```bash
python -m project.runner.cli \
  --method hjb \
  --market_mode bootstrap \
  --data_profile dev \
  --es_mode loss \
  --alpha 0.95 \
  --cvar_target 0.40 \
  --cvar_tol 0.01 \
  --lambda_min 0.0 \
  --lambda_max 2.0 \
  --calib_fast on \
  --calib_max_iter 8 \
  --tag hjb_calibration_example
```

---

## 7) 데이터 관련 옵션 예시

### 7.1 data_profile에 따른 기본 CSV 자동 설정

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile dev `
  --es_mode wealth `
  --tag use_profile_dev
```

### 7.2 특정 데이터 윈도우 지정

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --data_window 2005-01:2020-12 `
  --es_mode wealth `
  --tag with_data_window
```

---

## 8) 자산배분 및 환헤지 옵션 예시

### 8.1 균등 배분 + 100% 환헤지

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --alpha_mix equal `
  --h_FX 1 `
  --es_mode wealth `
  --tag alloc_fx_equal_fullhedge
```

### 8.2 개별 가중치 지정 + 부분 환헤지

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --alpha_kr 0.50 `
  --alpha_us 0.30 `
  --alpha_au 0.20 `
  --h_FX 0.5 `
  --fx_hedge_cost 0.002 `
  --es_mode wealth `
  --tag alloc_fx_custom
```

---

## 9) 로그 메시지 확인

`--quiet off`를 사용하면 데이터 요약 로그가 함께 출력됩니다.

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile full `
  --es_mode loss `
  --F_target 0.60 `
  --return_actor on `
  --quiet off `
  --print_mode summary `
  --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" `
  --no_paths `
  --tag demo_with_data_log
```

---

## 10) 테스트 실행

### 10.1 PyTest 설치

```powershell
python -m pip install pytest
```

### 10.2 단일 테스트 파일 실행

```powershell
python -m pytest -q tests/test_cli_regression.py
```

### 10.3 전체 테스트 실행

```powershell
python -m pytest -q
```

---

## 11) 새 옵션 요약

* `--print_mode {full,metrics,summary}`: 표준 출력 형식 선택

  * `full`: 원본 전체 구조 출력
  * `metrics`: 선택 키만 추출하여 납작한 형태로 출력
  * `summary`: 주요 메타(`tag, asset, method, age0, sex, n_paths, T`)와 선택 메트릭 묶음 출력
* `--metrics_keys "EW,ES95,Ruin,mean_WT,es95_source"`: `metrics` 또는 `summary` 모드에서 표출할 키
* `--no_paths`: `extra.eval_WT`, `extra.ruin_flags` 등의 대용량 배열을 출력에서 제거(길이 정보만 유지)
* `--return_actor on`: `(cfg, actor)` 반환 → CLI가 내부적으로 `evaluate(..., return_paths=True)`를 호출하여 경로 기반 CVaR(보간식) 재계산

---

## 12) 트러블슈팅

* 오류: `market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full) 필요.`

  * 해결: `--market_csv` 또는 `--data_profile dev` 혹은 `--data_profile full`을 지정하세요.
* PowerShell 파이프라인에서 배열 원소 접근 에러(`ExpandProperty .`)

  * 해결: `| Select-Object -ExpandProperty eval_WT | Select-Object -First 10`처럼 **속성명**을 정확히 지정하세요.



