# =========================
# presets.ps1 (refactored, safe metrics loading & no alias conflicts)
# =========================
# 로드:   . .\presets.ps1
# 예시:
#   rs  -Tag "test_run"
#   scp -Caps 0.005,0.0065,0.007 -Epochs 100
#   lm  -Top 10
#   wt
#   fbt -Like 'sweep_cap*'
#   bp  -Prefix 'cap_sweep_e80_' -Show
#   pbs -Prefix 'cap_sweep_e80_' -Epochs 400 -Entropy 0.0012 -Tag 'final_cap_from_sweep'

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
$PY      = ".\.venv\Scripts\python.exe"
$CLI     = "project.runner.cli"
$METRICS = ".\outputs\paper_main\_logs\metrics.csv"

# -----------------------------
# helpers: safe number parsing & metrics loading
# -----------------------------
function _ToDouble([object]$v) {
  $n = 0.0
  if ([double]::TryParse("$v", [ref]$n)) { return $n }
  return $null
}

function Get-MetricsRows {
  if (-not (Test-Path $METRICS)) {
    Write-Warning "metrics.csv not found at $METRICS"
    return @()
  }
  Import-Csv $METRICS |
    # 최소 유효성: RL 행 & es_mode 존재 & EW 숫자 가능
    Where-Object {
      $_.method -eq 'rl' -and $_.es_mode -in @('wealth','loss') -and (_ToDouble $_.EW) -ne $null
    } |
    # 파생 컬럼(정렬/계산용) 추가
    Select-Object *, @{
      n='EW_d';   e={ _ToDouble $_.EW }
    }, @{
      n='ES95_d'; e={ _ToDouble $_.ES95 }
    }, @{
      n='Ruin_d'; e={ _ToDouble $_.Ruin }
    }
}

# -----------------------------
# 실행기
# -----------------------------
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

# -----------------------------
# 조회/표시
# -----------------------------
function LatestMetrics {
  <#
    .SYNOPSIS
      최근 저장된 metrics.csv 표시
    .EXAMPLE
      LatestMetrics -Top 10
  #>
  [CmdletBinding()]
  param([int] $Top = 10)

  $rows = Get-MetricsRows
  if (-not $rows) { return }

  $rows |
    Sort-Object ts -Descending |
    Select-Object -First $Top |
    # 표시용: 계산된 *_d가 있으면 그 값을 보여주기
    Select-Object ts, method, es_mode, tag,
      @{n='EW';   e={ if ($_.EW_d -ne $null) { $_.EW_d } else { $_.EW } }},
      @{n='ES95'; e={ if ($_.ES95_d -ne $null) { $_.ES95_d } else { $_.ES95 } }},
      @{n='Ruin'; e={ if ($_.Ruin_d -ne $null) { $_.Ruin_d } else { $_.Ruin } }} |
    Format-Table -AutoSize
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

  $rows = Get-MetricsRows
  if (-not $rows) { return }

  $rows |
    Where-Object { $_.tag -like $Like } |
    Sort-Object ts -Descending |
    Select-Object ts, method, es_mode, tag,
      @{n='EW';   e={ if ($_.EW_d -ne $null) { $_.EW_d } else { $_.EW } }},
      @{n='ES95'; e={ if ($_.ES95_d -ne $null) { $_.ES95_d } else { $_.ES95 } }},
      @{n='Ruin'; e={ if ($_.Ruin_d -ne $null) { $_.Ruin_d } else { $_.Ruin } }} |
    Format-Table -AutoSize
}

function Warmup {
  <#
    .SYNOPSIS
      1 에폭 워밍업 실행(로그/저장 확인용)
  #>
  RunSim -Tag "warmup_autosave" -Epochs 1 -Entropy 0.001 -Cap 0.005 -EsMode "wealth"
}

# -----------------------------
# Best/Promote 유틸
# -----------------------------
function BestByPrefix {
  <#
    .SYNOPSIS
      특정 Prefix로 시작하는 tag 중 EW 상위 선택 (cap 복원 포함)
    .EXAMPLE
      BestByPrefix -Prefix 'cap_sweep_e80_' -Show
  #>
  [CmdletBinding()]
  param(
    [string] $Prefix = 'cap_sweep_e80_',
    [switch] $Show
  )

  $rows = Get-MetricsRows | Where-Object { $_.tag -and ($_.tag -like "$Prefix*") -and $_.es_mode -eq 'wealth' }
  if (-not $rows) { Write-Warning "No rows for prefix '$Prefix'"; return }

  $sorted = $rows | Sort-Object EW_d -Descending

  if ($Show) {
    $sorted |
      Select-Object -First 10 -Property ts, tag, @{
        n='cap'; e={
          if ($_.tag -match '_(\d{5})$') { ([int]$Matches[1]) / 10000.0 } else { $null }
        }
      }, @{
        n='EW'; e={ $_.EW_d }
      }, @{
        n='ES95'; e={ $_.ES95_d }
      }, @{
        n='Ruin'; e={ $_.Ruin_d }
      } | Format-Table -AutoSize
  }

  $best = $sorted | Select-Object -First 1
  $cap  = $null
  if ($best.tag -match '_(\d{5})$') { $cap = ([int]$Matches[1]) / 10000.0 }

  return [PSCustomObject]@{
    ts   = $best.ts
    tag  = $best.tag
    EW   = $best.EW_d
    ES95 = $best.ES95_d
    Ruin = $best.Ruin_d
    cap  = $cap
  }
}

function PromoteBestFromSweep {
  <#
    .SYNOPSIS
      BestByPrefix로 고른 cap으로 롱런 실행
    .EXAMPLE
      PromoteBestFromSweep -Prefix 'cap_sweep_e80_' -Epochs 400 -Entropy 0.0012 -Tag 'final_cap_from_sweep'
  #>
  [CmdletBinding()]
  param(
    [string] $Prefix  = 'cap_sweep_e80_',
    [int]    $Epochs  = 400,
    [double] $Entropy = 0.0012,
    [string] $Tag     = 'final_cap_from_sweep',
    [string] $EsMode  = 'wealth'
  )
  $best = BestByPrefix -Prefix $Prefix -Show
  if (-not $best) { return }
  if ($null -eq $best.cap) {
    Write-Warning "Could not parse cap from tag '$($best.tag)'"
    return
  }
  Write-Host "[promote] best tag: $($best.tag)  cap=$($best.cap)  EW=$('{0:F6}' -f $best.EW)"
  RunSim -Tag $Tag -Epochs $Epochs -Entropy $Entropy -Cap $best.cap -EsMode $EsMode
}

# -----------------------------
# (선택) 깨끗한 행만 남기는 정리 유틸
# -----------------------------
function Clean-MetricsCsv {
  <#
    .SYNOPSIS
      metrics.csv에서 숫자 필드(EW/ES95/Ruin) 깨진 행 제거
    .EXAMPLE
      Clean-MetricsCsv -InPlace
  #>
  [CmdletBinding()] param([switch]$InPlace)

  if (-not (Test-Path $METRICS)) { Write-Warning "metrics.csv not found"; return }
  $all  = Import-Csv $METRICS
  $good = $all | Where-Object { (_ToDouble $_.EW) -ne $null -and (_ToDouble $_.ES95) -ne $null -and (_ToDouble $_.Ruin) -ne $null }

  $out = "$($METRICS).clean"
  $good | Export-Csv $out -NoTypeInformation -Encoding UTF8
  Write-Host "[clean] wrote $out  (kept $($good.Count) / total $($all.Count))"

  if ($InPlace) {
    $bak = "$($METRICS).bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
    Copy-Item $METRICS $bak
    Move-Item $out $METRICS -Force
    Write-Host "[clean] replaced metrics.csv (backup: $bak)"
  }
}

# ===== Utils: metrics merge/clean =====
function cmcsv {
  param(
    [switch]$InPlace
  )
  $root = "D:\01_simul"
  $script = Join-Path $root "scripts\merge_metrics.ps1"
  if ($InPlace) {
    pwsh -File $script -Root $root -InPlace
  } else {
    pwsh -File $script -Root $root
  }
}

# ===== Report: plot metrics & export (alias 충돌 방지용 이름: rpt) =====
function rpt {
  param(
    [string]$In = "D:\01_simul\outputs\paper_main\_logs\metrics.csv",
    [string]$OutDir = "D:\01_simul\outputs\paper_main\reports",
    [double]$Alpha = 0.95,
    [int]$TopN = 20,
    [double]$ESCap
  )
  $py = "D:\01_simul\.venv\Scripts\python.exe"
  $script = "D:\01_simul\project\utils\plot_metrics.py"
  $args = @("--in", $In, "--outdir", $OutDir, "--alpha", $Alpha, "--topn", $TopN)
  if ($PSBoundParameters.ContainsKey("ESCap")) {
    $args += @("--es_cap", $ESCap)
  }
  & $py $script @args
}

# (선택) 기존 Windows 기본 alias 'rp'를 그대로 두고 싶다면 아무 것도 안 하셔도 됩니다.
# (대신 앞으로는 rpt 를 쓰세요)
# (고급) 정말 'rp'로 쓰고 싶다면 아래 2줄을 presets.ps1 상단에 추가하면 alias를 지워 재정의할 수 있습니다:
#   if (Get-Alias rp -ErrorAction SilentlyContinue) { Remove-Item alias:rp -Force }
#   function rp { param([Parameter(ValueFromRemainingArguments=$true)]$args) rpt @args }


# === 편의 별칭 (충돌 회피 버전) ===
# 'sc' = Set-Content, 'ft' = Format-Table 이라 읽기 전용 별칭과 충돌하므로 사용하지 않음
Set-Alias rs    RunSim               -Scope Global -ErrorAction SilentlyContinue
Set-Alias scp   SweepCap             -Scope Global -ErrorAction SilentlyContinue
Set-Alias lm    LatestMetrics        -Scope Global -ErrorAction SilentlyContinue
Set-Alias wt    Warmup               -Scope Global -ErrorAction SilentlyContinue
Set-Alias fbt   FindByTag            -Scope Global -ErrorAction SilentlyContinue
Set-Alias bp    BestByPrefix         -Scope Global -ErrorAction SilentlyContinue
Set-Alias pbs   PromoteBestFromSweep -Scope Global -ErrorAction SilentlyContinue
Set-Alias cmcsv Clean-MetricsCsv     -Scope Global -ErrorAction SilentlyContinue
