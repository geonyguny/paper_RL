$COMMON = @(
  '--asset','US','--market_mode','bootstrap',
  '--market_csv','D:\01_simul\project\data\market\kr_us_gold_bootstrap_full.csv',
  '--data_window','2005-01:2020-12','--use_real_rf','on',
  '--horizon_years','35','--w_max','0.70','--w_fixed','0.60','--fee_annual','0.004',
  '--seeds', 0, 1, 2, 3, 4,
  '--n_paths','120',
  '--autosave','on','--quiet','off','--bands','on',
  '--outputs','.\\outputs\\paper_main'
)

$RLBASE = @(
  '--method','rl',
  '--es_mode','wealth',
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
