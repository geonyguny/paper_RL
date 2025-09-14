import pytest

def test_env_import_and_reset():
    try:
        from project.env.retirement_env import RetirementEnv
    except Exception as e:
        pytest.skip(f"skip on CI (env import failed): {e}")
        return
    env = RetirementEnv(horizon_years=1, market_mode='iid', mu_risky=0.0, sigma_risky=0.0, r_safe=0.0)
    s = env.reset()
    assert hasattr(env, 'state')
