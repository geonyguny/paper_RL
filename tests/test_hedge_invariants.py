import numpy as np, pytest

def _mk_env():
    from project.env.retirement_env import RetirementEnv
    return RetirementEnv(horizon_years=1, hedge='on', hedge_mode='downside',
                         hedge_sigma_k=0.75, hedge_cost=0.005,
                         fee_annual=0.0, market_mode='iid',
                         mu_risky=0.0, sigma_risky=0.0, r_safe=0.0)

def test_downside_no_gain_on_up(monkeypatch):
    env = _mk_env(); env.reset()
    def up(): return +0.05, 0.0
    monkeypatch.setattr(env, "_draw_returns", up, raising=False)
    _,_,_,info = env.step(np.array([0.02, 0.60]))
    assert info.get("hedge_active", 0) == 0
    assert "r_risky_eff" in info
    # 상승에서는 r_eff == r (여기선 0.05)
    assert abs(info["r_risky_eff"] - 0.05) < 1e-12

def test_downside_softens_down_not_flip(monkeypatch):
    env = _mk_env(); env.reset()
    def down(): return -0.10, 0.0
    monkeypatch.setattr(env, "_draw_returns", down, raising=False)
    _,_,_,info = env.step(np.array([0.02, 0.60]))
    assert info.get("hedge_active", 0) == 1
    r_eff = info["r_risky_eff"]
    # 여전히 음수이되 |r_eff| <= |r|
    assert r_eff <= 0.0 and abs(r_eff) <= 0.10 + 1e-12
