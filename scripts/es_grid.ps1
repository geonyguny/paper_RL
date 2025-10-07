# scripts/es_grid.ps1
param(
  [double[]]$Lambdas = @(0.2,0.4,0.8,1.2),
  [double]$Cap = 0.0045,
  [int]$Epochs = 120,
  [double]$Entropy = 0.0012,
  [double]$Alpha = 0.95,
  [string]$TagPrefix = "esgrid_term",
  [string]$DataWindow = "2005-01:2020-12",
  [string]$Root = "D:\01_simul"
)

$ErrorActionPreference = "Stop"
$py  = Join-Path $Root ".venv\Scripts\python.exe"
$cli = Join-Path $Root "project\runner\cli.py"

foreach ($lam in $Lambdas) {
  $lamStr = ("{0:0.00}" -f $lam).Replace(",",".")
  $capSlug = ("{0:0.00000}" -f $Cap).Replace("0.","000").Replace("0,","000")
  $tag  = "{0}_l{1}_cap{2}" -f $TagPrefix, $lamStr, $capSlug

  $args = @(
    $cli, "--asset","US","--method","rl",
    "--outputs",".\outputs\paper_main",
    "--market_mode","bootstrap","--market_csv",(Join-Path $Root "project\data\market\kr_us_gold_bootstrap_full.csv"),
    "--bootstrap_block","24","--use_real_rf","on",
    "--horizon_years","35","--w_max","0.70","--fee_annual","0.004",
    "--alpha",$Alpha, "--lambda_term","0.0",  # term penalty는 0으로 둠
    "--es_mode","term",                       # ★ terminal ES 관점
    "--cvar_stage","on", "--alpha_stage",$Alpha, "--lambda_stage",$lamStr,  # ★ CVaR 스테이지 온
    "--xai_on","on",
    "--rl_epochs",$Epochs,"--entropy_coef",$Entropy,"--rl_q_cap",$Cap,
    "--data_window",$DataWindow,
    "--tag",$tag
  )
  & $py @args
}
