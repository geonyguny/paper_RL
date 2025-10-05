# scripts/merge_metrics.ps1
# usage:
#   pwsh -File "D:\01_simul\scripts\merge_metrics.ps1" -Root "D:\01_simul" -InPlace
#   pwsh -File "D:\01_simul\scripts\merge_metrics.ps1" -Root "D:\01_simul" -Out "D:\01_simul\outputs\paper_main\_logs\metrics_merged.csv"

param(
  [string]$Root = "D:\01_simul",
  [string]$Out = "",
  [switch]$InPlace
)

$ErrorActionPreference = "Stop"

# 0) 경로 세팅
$logsDir = Join-Path $Root "outputs\paper_main\_logs"
$mainCsv = Join-Path $logsDir "metrics.csv"
$tempMerged = Join-Path $logsDir "_metrics_merged_tmp.csv"
$backup = Join-Path $logsDir ("metrics_backup_{0:yyyyMMdd_HHmmss}.csv" -f (Get-Date))

if (!(Test-Path $logsDir)) {
  throw "[merge] logs 경로가 없습니다: $logsDir"
}
if (!(Test-Path $mainCsv)) {
  # 없으면 빈 헤더 생성(최소 필드 셋)
  @"
run_id,datetime,tag,method,EW,ES95,Ruin,mean_WT,cap,epochs,entropy,alpha,w_max,q_floor,phi_adval,seed,notes
"@ | Set-Content -Encoding UTF8 $mainCsv
  Write-Host "[merge] 신규 metrics.csv 생성" -ForegroundColor Yellow
}

# 1) 통합 대상 모으기 (현재는 단일 파일 정책, 확장 여지 남김)
$csvFiles = @($mainCsv)

# 2) 로드 & 정리
$rows = @()
foreach ($f in $csvFiles) {
  try {
    $data = Import-Csv -Path $f
    $rows += $data
  } catch {
    Write-Host "[merge] CSV 로드 실패: $f ($($_.Exception.Message))" -ForegroundColor Red
  }
}

if ($rows.Count -eq 0) {
  Write-Host "[merge] 병합할 데이터가 없습니다." -ForegroundColor Yellow
  exit 0
}

# 3) 최소 컬럼 보장
$required = @("run_id","datetime","tag","method","EW","ES95","Ruin","mean_WT","cap","epochs","entropy","alpha","w_max","q_floor","phi_adval","seed","notes")
foreach ($r in $rows) {
  foreach ($col in $required) {
    if (-not ($r.PSObject.Properties.Name -contains $col)) {
      $r | Add-Member -NotePropertyName $col -NotePropertyValue ""
    }
  }
}

# 4) 타입/클린: 숫자 필드 변환 & 이상값 제거(문자 'False','sigma' 등)
function To-Num($v) {
  if ($null -eq $v) { return $null }
  $s = [string]$v
  if ($s -match '^(False|True|sigma|null|NaN)$') { return $null }
  $n = $null
  [double]::TryParse($s, [ref]$n) | Out-Null
  return $n
}

$numericCols = @("EW","ES95","Ruin","mean_WT","cap","epochs","entropy","alpha","w_max","q_floor","phi_adval")
foreach ($r in $rows) {
  foreach ($c in $numericCols) {
    $r.$c = To-Num $r.$c
  }
}

# 5) 태그로부터 하이퍼 파싱(부족 시 보강)
#    예: tag: cap_sweep_e80_cap0.0075_ent0.0012 → cap, epochs, entropy 추론
foreach ($r in $rows) {
  $tag = [string]$r.tag
  if ($tag) {
    if (-not $r.epochs) {
      if ($tag -match 'e(\d+)') { $r.epochs = [double]$Matches[1] }
    }
    if (-not $r.cap) {
      if ($tag -match 'cap([0-9]*\.[0-9]+|[0-9]+)') { $r.cap = [double]$Matches[1] }
    }
    if (-not $r.entropy) {
      if ($tag -match 'ent([0-9]*\.[0-9]+|[0-9]+)') { $r.entropy = [double]$Matches[1] }
    }
  }
}

# 6) run_id 부여(없으면)
foreach ($r in $rows) {
  if (-not $r.run_id -or [string]::IsNullOrWhiteSpace($r.run_id)) {
    $r.run_id = [guid]::NewGuid().ToString()
  }
}

# 7) 중복 제거 키: (datetime, tag, seed, method) → 최신만 보존
$rows = $rows | Sort-Object datetime -Descending
$seen = @{}
$dedup = @()
foreach ($r in $rows) {
  $key = "{0}|{1}|{2}|{3}" -f $r.datetime,$r.tag,$r.seed,$r.method
  if (-not $seen.ContainsKey($key)) {
    $seen[$key] = $true
    $dedup += $r
  }
}
$dedup = $dedup | Sort-Object datetime

# 8) 백업 및 저장
Copy-Item -Path $mainCsv -Destination $backup -Force
$dedup | Export-Csv -Path $tempMerged -NoTypeInformation -Encoding UTF8

if ($InPlace) {
  Move-Item -Path $tempMerged -Destination $mainCsv -Force
  Write-Host "[merge] InPlace 완료. 백업: $backup" -ForegroundColor Green
} else {
  if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path $logsDir "metrics_merged.csv"
  }
  Move-Item -Path $tempMerged -Destination $Out -Force
  Write-Host "[merge] 병합 저장: $Out (백업: $backup)" -ForegroundColor Green
}
