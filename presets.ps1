# =========================
# presets.ps1 (refactored, no alias conflicts)
# =========================
# 로드:   . .\presets.ps1
# 예시:
#   rs -Tag "test_run"
#   scp -Caps 0.005,0.0065,0.007 -Epochs 100
#   lm -Top 10
#   wt
#   fbt -Like 'sweep_cap*'

# --- 공통 프리셋 (CLI에 그대로 전달되는 인자 배열) ---
$COMMON = @(
  '--asset','US','--market_mode','bootstrap',
  '--market_csv','D:\01_simul\project\data\market\kr_us_gold_bootstrap_full.csv',
  '--data_window','2005-01:2020-12','--use_real_rf','on',
  '--horizon_years','35','--w_max','0.70','--w_fixed','0.60','--fee_annual','0.004',
  # seeds는 공백 구분 다중 인자로! (쉼표 X)
  '--seeds', 0, 1, 2, 3, 4,
  '--n_paths','120',
  '--autosave','on','--quiet','off','--bands','on',
  '--outputs','.\\outputs\\paper_main'
)

# --- RL 기본 하이퍼 ---
$RLBASE = @(
  '--method','rl',
  '--es_mode','wealth',              # 평가 저장에 반영됨
  '--rl_steps_per_epoch','4096',
  '--lr','1e-4',
  '--gae_lambda','0.95',
  '--value_coef','0.5',
  '--max_grad_norm','0.5',
  '--hedge','off',
  '--u_scale','0.0',
  '--survive_bonus','0.003',
  '--teacher_eps0','1.0','--teacher_decay','0.99985'
)

# --- 경로/실행기 ---
$PY  = ".\.venv\Scripts\python.exe"
$CLI = "project.runner.cli"
$METRICS = ".\outputs\paper_main\_logs\metrics.csv"

function RunSim {
  <#
    .SYNOPSIS
      단일 러닝 실행 (프리셋 + 덮어쓰기)
    .EXAMPLE
      RunSim -Tag "cap005_e80" -Epochs 80 -Entropy 0.0012 -Cap 0.005
  #>
  [CmdletBinding()]
  param(
    [string] $Tag = "ad-hoc",
    [int]    $Epochs = 60,
    [double] $Entropy = 0.0012,
    [double] $Cap = 0.005,
    [string] $EsMode = "wealth",
    [string] $DataWindow
  )
  $extra = @(
    '--rl_epochs', $Epochs,
    '--entropy_coef', $Entropy,
    '--rl_q_cap', $Cap,
    '--es_mode', $EsMode,
    '--tag', $Tag
  )
  if ($PSBoundParameters.ContainsKey('DataWindow') -and $DataWindow) {
    $extra += @('--data_window', $DataWindow)
  }

  & $PY -m $CLI @COMMON @RLBASE @extra
}

function SweepCap {
  <#
    .SYNOPSIS
      rl_q_cap 스윕 실행
    .EXAMPLE
      SweepCap -Caps 0.005,0.0075,0.01 -Epochs 80 -Entropy 0.0015
  #>
  [CmdletBinding()]
  param(
    [double[]] $Caps = @(0.005, 0.0075, 0.01),
    [int]      $Epochs = 80,
    [double]   $Entropy = 0.0015,
    [string]   $TagPrefix = "sweep_cap",
    [string]   $EsMode = "wealth",
    [string]   $DataWindow
  )
  foreach ($c in $Caps) {
    $capStr = ("{0:N4}" -f $c).Replace('.', '')
    $tag = "{0}_{1}" -f $TagPrefix, $capStr
    RunSim -Tag $tag -Epochs $Epochs -Entropy $Entropy -Cap $c -EsMode $EsMode -DataWindow $DataWindow
  }
}

function LatestMetrics {
  <#
    .SYNOPSIS
      최근 저장된 metrics.csv 표시
    .EXAMPLE
      LatestMetrics -Top 10
  #>
  [CmdletBinding()]
  param(
    [int] $Top = 10
  )
  if (Test-Path $METRICS) {
    Import-Csv $METRICS |
      Sort-Object ts -Descending |
      Select-Object -First $Top |
      Format-Table -AutoSize ts,method,es_mode,tag,EW,ES95,Ruin
  } else {
    Write-Warning "metrics.csv not found at $METRICS"
  }
}

function Warmup {
  <#
    .SYNOPSIS
      1 에폭 워밍업 실행(로그/저장 확인용)
  #>
  RunSim -Tag "warmup_autosave" -Epochs 1 -Entropy 0.001 -Cap 0.005 -EsMode "wealth"
}

function FindByTag {
  <#
    .SYNOPSIS
      metrics.csv에서 tag로 필터링
    .EXAMPLE
      FindByTag -Like 'sweep_cap*'
  #>
  [CmdletBinding()]
  param([string] $Like = '*')
  if (Test-Path $METRICS) {
    Import-Csv $METRICS |
      Where-Object { $_.tag -like $Like } |
      Sort-Object ts -Descending |
      Format-Table -AutoSize ts,method,es_mode,tag,EW,ES95,Ruin
  } else {
    Write-Warning "metrics.csv not found at $METRICS"
  }
}

# === 편의 별칭 (충돌 회피 버전) ===
# 'sc' = Set-Content, 'ft' = Format-Table 이라 읽기 전용 별칭과 충돌하므로 사용하지 않음
Set-Alias rs  RunSim        -Scope Local -ErrorAction SilentlyContinue
Set-Alias scp Sweep-Cap     -Scope Local -ErrorAction SilentlyContinue
Set-Alias lm  Latest-Metrics -Scope Local -ErrorAction SilentlyContinue
Set-Alias wt  Warmup        -Scope Local -ErrorAction SilentlyContinue
Set-Alias fbt Find-ByTag    -Scope Local -ErrorAction SilentlyContinue
