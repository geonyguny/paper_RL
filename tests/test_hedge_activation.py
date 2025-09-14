import numpy as np

def test_hedge_cost_applies_only_when_active(monkeypatch):
    from project.env.retirement_env import RetirementEnv
    env = RetirementEnv(
        horizon_years=1,
        hedge="on", hedge_mode="downside",
        hedge_sigma_k=0.50, hedge_cost=0.005,
        fee_annual=0.0,
        market_mode="iid", mu_risky=0.0, sigma_risky=0.0, r_safe=0.0
    )
    env.reset()
    q, w = 0.02, 0.60

    # 상승장(비발동)
    def up_stub(): return +0.05, 0.0
    monkeypatch.setattr(env, "_draw_returns", up_stub, raising=False)
    _, _, _, info_up = env.step(np.array([q, w], dtype=float))
    assert info_up.get("hedge_active", 0) == 0

    # 하락장(발동)
    env.reset()
    def down_stub(): return -0.10, 0.0
    monkeypatch.setattr(env, "_draw_returns", down_stub, raising=False)
    _, _, _, info_dn = env.step(np.array([q, w], dtype=float))
    assert info_dn.get("hedge_active", 0) == 1
    assert "r_risky_eff" in info_dn
