import numpy as np, pytest

def _mk_env():
    from project.env.retirement_env import RetirementEnv
    return RetirementEnv(horizon_years=1, hedge="on", hedge_mode="downside",
                         hedge_sigma_k=0.75, hedge_cost=0.005,
                         fee_annual=0.0, market_mode="iid",
                         mu_risky=0.0, sigma_risky=0.0, r_safe=0.0)

def test_downside_no_gain_on_up(monkeypatch):
    env = _mk_env(); env.reset()
    def up(): return +0.05, 0.0
    monkeypatch.setattr(env, "_draw_returns", up, raising=False)
    s0 = env.state.W
    _,_,_,info = env.step(np.array([0.02, 0.6]))
    # 상승시 비활성이어야 하고 r_eff == r
    assert info.get("hedge_active", 0) == 0
    assert pytest.approx(info["r_risky_eff"], 1e-12) == 0.05

def test_downside_softens_down_not_flip(monkeypatch):
    env = _mk_env(); env.reset()
    def down(): return -0.10, 0.0
    monkeypatch.setattr(env, "_draw_returns", down, raising=False)
    _,_,_,info = env.step(np.array([0.02, 0.6]))
    # 하락시 활성, r_eff는 여전히 음수이되 |r_eff| <= |r|
    assert info.get("hedge_active", 0) == 1
    r, r_eff = -0.10, info["r_risky_eff"]
    assert r_eff <= 0.0 and abs(r_eff) <= abs(r) + 1e-12
