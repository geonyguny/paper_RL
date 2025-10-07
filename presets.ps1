# =========================
# presets.ps1 (single-file: capmicro/sweepw/esgrid built-in; no alias conflicts)
# =========================
# 로드:   . .\presets.ps1
# 예시:
#   rs   -Tag "test_run"
#   scp  -Caps 0.005,0.0065,0.007 -Epochs 100
#   lm   -Top 10
#   wt
#   fbt  -Like 'sweep_cap*'
#   bp   -Prefix 'cap_sweep_e80_' -Show
#   pbs  -Prefix 'cap_sweep_e80_' -Epochs 400 -Entropy 0.0012 -Tag 'final_cap_from_sweep'
#   rpt  -TopN 20 -ESCap 0.35
#   capmicro -Start 0.0042 -End 0.0048 -Step 0.0001 -Epochs 120 -Entropy 0.0012 -TagPrefix "cap_micro_e120"
#   sweepw   -Windows "2000-01:2020-12","2005-01:2024-12" -Cap 0.0045 -Epochs 100 -Entropy 0.0012 -TagPrefix "cap005_window"
#   esgrid   -Cap 0.0045 -Lambdas 0.2,0.4,0.8,1.2 -Epochs 120 -Entropy 0.0012 -TagPrefix "esgrid_term"

# (선택) Windows 기본 alias 'rp' 제거하고 싶다면 상단에서 해제 가능 (우리는 rpt를 사용)
# if (Get-Alias rp -ErrorAction SilentlyContinue) { Remove-Item alias:rp -Force }

# --- 공통 프리셋 (CLI에 그대로 전달되는 인자 배열) ---
$COMMON = @(
  '--asset','US',
  '--market_mode','bootstrap',
  '--market_csv','D:\01_simul\project\data\market\kr_us_gold_bootstrap_full.csv',
  '--bootstrap_block','24',
  '--data_window','2005-01:2020-12',
  '--use_real_rf','on',
  '--horizon_years','35',
  '--w_max','0.70',
  '--w_fixed','0.60',
  '--fee_annual','0.004',
  '--alpha','0.95',
  '--lambda_term','0.0',
  # seeds는 공백 구분 다중 인자로! (쉼표 X)
  '--seeds', 0, 1, 2, 3, 4,
  '--n_paths','120',
  '--autosave','on',
  '--quiet','off',
  '--bands','on',
  '--outputs','.\\outputs\\paper_main'
)

# --- RL 기본 하이퍼 ---
$RLBASE = @(
  '--method','rl',
  '--es_mode','wealth',
  '--xai_on','on',
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
  $rows = Import-Csv $METRICS

  # ts 컬럼 없을 시 datetime으로 보완
  if ($rows.Count -gt 0 -and -not ($rows[0].PSObject.Properties.Name -contains 'ts') -and ($rows[0].PSObject.Properties.Name -contains 'datetime')) {
    $rows | ForEach-Object { $_ | Add-Member -NotePropertyName ts -NotePropertyValue $_.datetime -Force }
  }

  $rows |
    Where-Object {
      $_.method -eq 'rl' -and $_.es_mode -in @('wealth','loss','term') -and (_ToDouble $_.EW) -ne $null
    } |
    Select-Object *, @{n='EW_d';e={ _ToDouble $_.EW }},
                     @{n='ES95_d';e={ _ToDouble $_.ES95 }},
                     @{n='Ruin_d';e={ _ToDouble $_.Ruin }}
}

# -----------------------------
# 실행기
# -----------------------------
function RunSim {
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
  [CmdletBinding()]
  param([int] $Top = 10)

  $rows = Get-MetricsRows
  if (-not $rows) { return }

  $rows |
    Sort-Object ts -Descending |
    Select-Object -First $Top |
    Select-Object ts, method, es_mode, tag,
      @{n='EW';   e={ if ($_.EW_d   -ne $null) { $_.EW_d }   else { $_.EW } }},
      @{n='ES95'; e={ if ($_.ES95_d -ne $null) { $_.ES95_d } else { $_.ES95 } }},
      @{n='Ruin'; e={ if ($_.Ruin_d -ne $null) { $_.Ruin_d } else { $_.Ruin } }} |
    Format-Table -AutoSize
}

function FindByTag {
  [CmdletBinding()]
  param([string] $Like = '*')

  $rows = Get-MetricsRows
  if (-not $rows) { return }

  $rows |
    Where-Object { $_.tag -like $Like } |
    Sort-Object ts -Descending |
    Select-Object ts, method, es_mode, tag,
      @{n='EW';   e={ if ($_.EW_d   -ne $null) { $_.EW_d }   else { $_.EW } }},
      @{n='ES95'; e={ if ($_.ES95_d -ne $null) { $_.ES95_d } else { $_.ES95 } }},
      @{n='Ruin'; e={ if ($_.Ruin_d -ne $null) { $_.Ruin_d } else { $_.Ruin } }} |
    Format-Table -AutoSize
}

function Warmup {
  RunSim -Tag "warmup_autosave" -Epochs 1 -Entropy 0.001 -Cap 0.005 -EsMode "wealth"
}

# -----------------------------
# Best/Promote 유틸
# -----------------------------
function BestByPrefix {
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
        n='cap'; e={ if ($_.tag -match '_(\d{5})$') { ([int]$Matches[1]) / 10000.0 } else { $null } }
      }, @{ n='EW'; e={ $_.EW_d } }, @{ n='ES95'; e={ $_.ES95_d } }, @{ n='Ruin'; e={ $_.Ruin_d } } |
      Format-Table -AutoSize
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
# metrics 정리/병합 유틸 (이름 충돌 방지: cmmerge / cmclean)
# -----------------------------
function Clean-MetricsCsv {
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

function CmMerge {
  param([switch]$InPlace)
  $root = "D:\01_simul"
  $script = Join-Path $root "scripts\merge_metrics.ps1"
  if ($InPlace) { pwsh -File $script -Root $root -InPlace } else { pwsh -File $script -Root $root }
}

# ===== Report: plot metrics & export (alias 충돌 방지: rpt) =====
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

# -----------------------------
# 헬퍼 3종 (외부 scripts\*.ps1 의존 없이 즉시 실행)
# -----------------------------
function capmicro {
  param(
    [double]$Start = 0.0042,
    [double]$End   = 0.0048,
    [double]$Step  = 0.0001,
    [int]   $Epochs = 120,
    [double]$Entropy = 0.0012,
    [string]$TagPrefix = "cap_micro_e120",
    [string]$DataWindow = $null   # null이면 COMMON 기본 적용
  )
  $caps = @()
  for ($c=$Start; $c -le $End + 1e-12; $c += $Step) { $caps += [math]::Round($c,5) }
  foreach ($c in $caps) {
    $slug = "{0:D5}" -f [int]([math]::Round($c*10000,0))   # 0.0045 -> 00045
    $tag  = "{0}_{1}" -f $TagPrefix, $slug
    if ($null -ne $DataWindow -and $DataWindow) {
      RunSim -Tag $tag -Epochs $Epochs -Entropy $Entropy -Cap $c -EsMode "wealth" -DataWindow $DataWindow
    } else {
      RunSim -Tag $tag -Epochs $Epochs -Entropy $Entropy -Cap $c -EsMode "wealth"
    }
  }
}

function sweepw {
  param(
    [string[]]$Windows = @("2000-01:2020-12","2005-01:2024-12"),
    [double]$Cap = 0.0045,
    [int]   $Epochs = 100,
    [double]$Entropy = 0.0012,
    [string]$TagPrefix = "cap005_window"
  )
  foreach ($win in $Windows) {
    $slug = ($win -replace "[:\-]","_") -replace "__","_"
    $capSlug = "{0:D5}" -f [int]([math]::Round($Cap*10000,0))
    $tag  = "{0}_{1}_{2}" -f $TagPrefix, $slug, $capSlug
    RunSim -Tag $tag -Epochs $Epochs -Entropy $Entropy -Cap $Cap -EsMode "wealth" -DataWindow $win
  }
}

function esgrid {
  param(
    [double[]]$Lambdas = @(0.2,0.4,0.8,1.2),
    [double]$Cap = 0.0045,
    [int]   $Epochs = 120,
    [double]$Entropy = 0.0012,
    [double]$Alpha = 0.95,
    [string]$TagPrefix = "esgrid_term",
    [string]$DataWindow = "2005-01:2020-12"
  )
  foreach ($lam in $Lambdas) {
    $lamStr = ("{0:0.00}" -f $lam).Replace(",",".")
    $capSlug = "{0:D5}" -f [int]([math]::Round($Cap*10000,0))
    $tag  = "{0}_l{1}_cap{2}" -f $TagPrefix, $lamStr, $capSlug

    $extra = @(
      '--rl_epochs', $Epochs,
      '--entropy_coef', $Entropy,
      '--rl_q_cap', $Cap,
      '--es_mode', 'term',
      '--cvar_stage','on','--alpha_stage', $Alpha, '--lambda_stage', $lamStr,
      '--data_window', $DataWindow,
      '--tag', $tag
    )
    & $PY -m $CLI @COMMON @RLBASE @extra
  }
}

# === 편의 별칭 (충돌 회피 버전) ===
Set-Alias rs    RunSim               -Scope Global -ErrorAction SilentlyContinue
Set-Alias scp   SweepCap             -Scope Global -ErrorAction SilentlyContinue
Set-Alias lm    LatestMetrics        -Scope Global -ErrorAction SilentlyContinue
Set-Alias wt    Warmup               -Scope Global -ErrorAction SilentlyContinue
Set-Alias fbt   FindByTag            -Scope Global -ErrorAction SilentlyContinue
Set-Alias bp    BestByPrefix         -Scope Global -ErrorAction SilentlyContinue
Set-Alias pbs   PromoteBestFromSweep -Scope Global -ErrorAction SilentlyContinue
# 과거 호환: cmcsv 호출 습관을 유지하고 싶은 경우 → clean에 매핑
if (Get-Alias cmcsv -ErrorAction SilentlyContinue) { Remove-Item alias:cmcsv -Force }
Set-Alias cmclean Clean-MetricsCsv   -Scope Global -ErrorAction SilentlyContinue
Set-Alias cmmerge CmMerge            -Scope Global -ErrorAction SilentlyContinue
Set-Alias cmcsv  Clean-MetricsCsv    -Scope Global -ErrorAction SilentlyContinue
