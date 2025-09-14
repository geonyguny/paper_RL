def test_env_import_and_reset():
    # env 모듈 임포트 + reset만 확인(빠르고 외부입력 불필요)
    from project.env.retirement_env import RetirementEnv
    env = RetirementEnv(horizon_years=1, market_mode="iid", mu_risky=0.0, sigma_risky=0.0, r_safe=0.0)
    s = env.reset()
    assert hasattr(env, "state")
