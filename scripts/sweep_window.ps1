# scripts/sweep_window.ps1
param(
  [string[]]$Windows = @("2000-01:2020-12","2005-01:2024-12"),
  [double]$Cap = 0.0045,
  [int]$Epochs = 100,
  [double]$Entropy = 0.0012,
  [string]$TagPrefix = "cap005_window",
  [string]$Root = "D:\01_simul"
)

$ErrorActionPreference = "Stop"
$py  = Join-Path $Root ".venv\Scripts\python.exe"
$cli = Join-Path $Root "project\runner\cli.py"

foreach ($win in $Windows) {
  # 윈도우를 태그용으로 깔끔히
  $slug = $win -replace "[:\-]","_" -replace "__","_"
  $tag  = "{0}_{1}_{2}" -f $TagPrefix, $slug, ("{0:0.00000}" -f $Cap).Replace("0.","000").Replace("0,","000")

  $args = @(
    $cli, "--asset","US","--method","rl",
    "--outputs",".\outputs\paper_main",
    "--market_mode","bootstrap","--market_csv",(Join-Path $Root "project\data\market\kr_us_gold_bootstrap_full.csv"),
    "--bootstrap_block","24","--use_real_rf","on",
    "--horizon_years","35","--w_max","0.70","--fee_annual","0.004",
    "--alpha","0.95","--lambda_term","0.0",
    "--es_mode","wealth","--xai_on","on",
    "--rl_epochs",$Epochs,"--entropy_coef",$Entropy,"--rl_q_cap",$Cap,
    "--data_window",$win,
    "--tag",$tag
  )
  & $py @args
}
