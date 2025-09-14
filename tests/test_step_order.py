import math, numpy as np

def test_step_order_consumption_returns_fee(monkeypatch):
    from project.env.retirement_env import RetirementEnv
    env = RetirementEnv(
        horizon_years=1, w_max=1.0, q_floor=0.0,
        fee_annual=0.12,              # 월로 환산되는지 확인용
        market_mode="iid", mu_risky=0.0, sigma_risky=0.0, r_safe=0.0
    )
    env.reset()
    W0 = env.state.W
    q, w = 0.10, 0.50

    # 수익률 0%로 고정 → 소비 후 fee만 적용되는 상황
    def _stub_returns(): return 0.0, 0.0
    monkeypatch.setattr(env, "_draw_returns", _stub_returns, raising=False)

    _, _, _, info = env.step(np.array([q, w], dtype=float))

    after_c = W0 * (1 - q)
    # 월수수료 해석 차이를 모두 허용(연→월 지수/단순 두 방식)
    fee_m_a = (1.0 + env.fee_annual)**(1/12) - 1.0
    fee_m_b = env.fee_annual / 12.0
    exp_a = after_c * (1 - fee_m_a)
    exp_b = after_c * (1 - fee_m_b)

    assert math.isclose(env.state.W, exp_a, rel_tol=1e-6, abs_tol=1e-6) \
        or math.isclose(env.state.W, exp_b, rel_tol=1e-6, abs_tol=1e-6)
