import argparse, json, os, hashlib, sys, csv, datetime
import numpy as _np
from .config import (
    SimConfig, ASSET_PRESETS,
    CVAR_TARGET_DEFAULT, CVAR_TOL_DEFAULT, LAMBDA_MIN_DEFAULT, LAMBDA_MAX_DEFAULT, LAMBDA_MAX_ITER
)
from .env import RetirementEnv
from .hjb import HJBSolver
from .eval import evaluate

# (optional) autosave (eval.py에 구현되어 있으면 사용)
try:
    from .eval import save_metrics_autocsv, plot_frontier_from_csv  # noqa
    _HAS_AUTOSAVE = True
except Exception:
    _HAS_AUTOSAVE = False

# --- stdout UTF-8 ---
try:
    sys.stdout.reconfigure(encoding='utf-8')  # py3.7+
except Exception:
    pass


def _arrhash(a):
    if a is None:
        return "none"
    x = _np.asarray(a, dtype=_np.float32).tobytes()
    return hashlib.md5(x).hexdigest()


def _auto_eta_grid(cfg: SimConfig, requested_n: int = None):
    """
    F_target 기반의 eta-grid 자동 생성:
      - lambda_term <= 0: eta-grid = (0.0,)
      - lambda_term > 0: eta-grid = linspace(0, F(>0이면 F, 아니면 1.0), n)
    """
    cur = getattr(cfg, "hjb_eta_grid", ())
    if float(getattr(cfg, "lambda_term", 0.0) or 0.0) <= 0.0:
        if not cur or len(cur) <= 1:
            cfg.hjb_eta_grid = (0.0,)
        return

    n = int(requested_n or getattr(cfg, "hjb_eta_n", 41) or 41)
    F = float(getattr(cfg, "F_target", 0.0) or 0.0)
    F = F if F > 0.0 else 1.0
    try:
        cfg.hjb_eta_grid = tuple(_np.linspace(0.0, F, n))
    except Exception:
        step = F / max(n - 1, 1)
        cfg.hjb_eta_grid = tuple(0.0 + i * step for i in range(n))


def make_cfg(args) -> SimConfig:
    cfg = SimConfig()
    if args.asset in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[args.asset].items():  # type: ignore
            setattr(cfg, k, v)
    cfg.asset = args.asset

    # overrides (CLI 인자 반영)
    for k, v in dict(
        w_max=args.w_max,
        # 수수료는 두 키 모두 세팅(모듈 간 호환)
        fee_annual=args.fee_annual,
        phi_adval=args.fee_annual,
        horizon_years=args.horizon_years,
        lambda_term=args.lambda_term,
        alpha=args.alpha,
        baseline=args.baseline,
        p_annual=args.p_annual,
        g_real_annual=args.g_real_annual,
        w_fixed=args.w_fixed,
        floor_on=args.floor_on,
        f_min_real=args.f_min_real,
        F_target=args.F_target,
        hjb_W_grid=args.hjb_W_grid,
        hjb_Nshock=args.hjb_Nshock,
        # hedge
        hedge=args.hedge,                      # "on"/"off" (RetirementEnv가 기대)
        hedge_on=(args.hedge == "on"),         # bool도 함께 유지(로깅 등)
        hedge_mode=args.hedge_mode,
        hedge_cost=args.hedge_cost,
        hedge_sigma_k=args.hedge_sigma_k,
        hedge_tx=args.hedge_tx,  # ★ 추가
        # market realism
        market_mode=args.market_mode,
        market_csv=args.market_csv,
        bootstrap_block=args.bootstrap_block,
        use_real_rf=args.use_real_rf,          # "on"/"off" 문자열 그대로
        # mortality
        mortality=args.mortality,              # "on"/"off" 문자열
        mortality_on=(args.mortality == "on"), # bool도 함께
        mort_table=args.mort_table,
        age0=args.age0,
        sex=args.sex,
        bequest_kappa=args.bequest_kappa,
        bequest_gamma=args.bequest_gamma,
        # RL shaping / constraints
        rl_q_cap=args.rl_q_cap,
        teacher_eps0=args.teacher_eps0,
        teacher_decay=args.teacher_decay,
        lw_scale=args.lw_scale,
        survive_bonus=args.survive_bonus,
        crra_gamma=args.crra_gamma,
        u_scale=args.u_scale,
        # stage-wise CVaR
        cvar_stage_on=(args.cvar_stage == "on"),
        alpha_stage=args.alpha_stage,
        lambda_stage=args.lambda_stage,
        cstar_mode=args.cstar_mode,
        cstar_m=args.cstar_m,
        # XAI toggle
        xai_on=(args.xai_on == "on"),
    ).items():
        if v is not None:
            setattr(cfg, k, v)

    cfg.seeds = tuple(args.seeds)
    cfg.n_paths_eval = int(args.n_paths)
    cfg.outputs = args.outputs
    cfg.method = args.method
    cfg.es_mode = args.es_mode

    # ==== 안정화 및 기본값 보정 ====
    # 1) eta-grid 자동 생성
    _auto_eta_grid(cfg, requested_n=args.hjb_eta_n)

    # 2) w-action 격자: config.py 자동설정 보완(없으면 8점 균등)
    if hasattr(cfg, "hjb_w_grid") and (getattr(cfg, "hjb_w_grid", None) in (None, ())):
        n_w = 8
        cfg.hjb_w_grid = tuple(_np.round(_np.linspace(0.0, cfg.w_max, n_w), 2))

    # 3) 충격 수 늘려 plateau 위험 감소
    setattr(cfg, "dev_split_w_grid", False)
    setattr(cfg, "hjb_Nshock", max(int(getattr(cfg, "hjb_Nshock", 32) or 32), 256))

    return cfg


def _monthly_from_cfg(cfg):
    """cfg.monthly()가 없을 때를 위한 월간 파라미터 폴백."""
    if hasattr(cfg, "monthly"):
        try:
            m = cfg.monthly()
            g_m = float(m.get("g_m", 0.0))
            p_m = float(m.get("p_m", 0.0))
            return g_m, p_m
        except Exception:
            pass
    steps = int(getattr(cfg, "steps_per_year", 12))
    g_ann = float(getattr(cfg, "g_real_annual", 0.0))
    p_ann = float(getattr(cfg, "p_annual", 0.0))
    g_m = (1.0 + g_ann) ** (1.0 / max(steps, 1)) - 1.0
    p_m = (1.0 + p_ann) ** (1.0 / max(steps, 1)) - 1.0
    return g_m, p_m


def build_actor(cfg: SimConfig, args):
    env = RetirementEnv(cfg)
    if args.method == "rule":
        if cfg.baseline == "4pct":
            def actor(_obs):
                q_m = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
                w = cfg.w_fixed if cfg.w_fixed is not None else cfg.w_max
                return q_m, w
            return actor

        elif cfg.baseline == "cpb":
            def actor(_obs):
                g_m, p_m = _monthly_from_cfg(cfg)
                q_m = p_m
                w = cfg.w_fixed if cfg.w_fixed is not None else cfg.w_max
                return q_m, w
            return actor

        elif cfg.baseline == "vpw":
            # VPW: 관측(obs)에 의존하지 않고 남은 기간만으로 q 결정
            # - env.t / env.T 사용
            # - g_m 계산은 cfg.monthly() 우선, 없으면 연간 성장률→월간 변환
            def _get_g_m_from_cfg(_cfg):
                # 1) cfg.monthly()가 있으면 우선 활용
                try:
                    if hasattr(_cfg, "monthly") and callable(_cfg.monthly):
                        m = _cfg.monthly()
                        if isinstance(m, dict) and "g_m" in m:
                            gm = float(m["g_m"])
                            if _np.isfinite(gm):
                                return gm
                except Exception:
                    pass
                # 2) fallback: 연간 실질 성장률 -> 월간 환산
                g_ann = float(getattr(_cfg, "g_real_annual", 0.0) or 0.0)
                spm = int(getattr(_cfg, "steps_per_year", 12) or 12)
                return (1.0 + g_ann)**(1.0 / spm) - 1.0

            def actor(_obs):
                # 남은 월수
                t = int(getattr(env, "t", 0))
                T = int(getattr(env, "T", 1))
                Nm = max(T - t, 1)

                # 월간 성장률
                g_m = _get_g_m_from_cfg(cfg)

                # 등비수열 합 기반 소진률
                if abs(g_m) < 1e-12:
                    q_m = 1.0 / Nm
                else:
                    a = (1.0 - (1.0 + g_m) ** (-Nm)) / g_m
                    q_m = 1.0 / max(a, 1e-12)  # 분모 안전판

                # 안전한 클리핑
                q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
                q_m = float(_np.clip(q_m, q_floor, 1.0))

                # 주식비중
                w = float(cfg.w_fixed if getattr(cfg, "w_fixed", None) is not None else cfg.w_max)

                # (옵션) t=0에서 q0와 파라미터 로그 찍기
                # if t == 0:
                #     print(f"[vpw] q0={q_m:.6f}, g_m={g_m:.8f}, T={T}, Nm={Nm}")

                return q_m, w

            return actor

        else:
            raise SystemExit("--baseline required for method=rule (4pct|cpb|vpw)")

    elif args.method == "hjb":
        # ==== HJB 해 정책 생성 ====
        sol = HJBSolver(cfg).solve(seed=cfg.seeds[0])
        Pi_w = sol.get('Pi_w', None)
        Pi_q = sol.get('Pi_q', None)

        # 정책 요약(진단용) + eta 출력
        print("policy_hash_q=", _arrhash(Pi_q))
        print("policy_hash_w=", _arrhash(Pi_w))
        if 'eta' in sol:
            try:
                print("eta_selected=", float(sol['eta']))
            except Exception:
                pass
            # 디버그 통계
            try:
                if Pi_w is not None and Pi_q is not None and getattr(Pi_w, "size", 0) > 0 and getattr(Pi_q, "size", 0) > 0:
                    _w = _np.asarray(Pi_w); _q = _np.asarray(Pi_q)
                    print("w_stats[min,mean,max]=", [float(_w.min()), float(_w.mean()), float(_w.max())])
                    print("q_stats[min,mean,max]=", [float(_q.min()), float(_q.mean()), float(_q.max())])
                    w_t0 = _w[0] if _w.ndim >= 1 else _w
                    q_t0 = _q[0] if _q.ndim >= 1 else _q
                    print("unique_w_t0 (rounded):", _np.unique(_np.round(w_t0, 3))[:10])
                    print("unique_q_t0 (rounded):", _np.unique(_np.round(q_t0, 3))[:10])
            except Exception as _e:
                print(f"[dbg] policy stats skipped: {_e}")

        # W-grid (solver 결과 없으면 기본 생성)
        if 'W_grid' in sol and sol['W_grid'] is not None:
            Wg = _np.asarray(sol['W_grid'], dtype=float)
        else:
            Wg = _np.linspace(cfg.hjb_W_min, cfg.hjb_W_max, cfg.hjb_W_grid)

        if Pi_w is None or getattr(Pi_w, 'size', 0) == 0 or Pi_q is None or getattr(Pi_q, 'size', 0) == 0:
            # fallback: 상수 정책
            const_w = float(min(max(cfg.hjb_w_grid[-1], 0.0), cfg.w_max))
            q4 = 1.0 - (1.0 - 0.04) ** (1.0 / cfg.steps_per_year)
            const_q = float(q4)
            def actor(_obs): return const_q, const_w
            return actor

        T_pol = int(Pi_w.shape[0])

        # obs: np.array([t_normalized, W])
        def actor(obs):
            # t 인덱스는 env.t가 신뢰도 높음
            t_idx = int(_np.clip(getattr(env, "t", 0), 0, T_pol - 1))
            W = float(obs[1])
            # 구간 인덱스 탐색
            i = int(_np.clip(_np.searchsorted(Wg, W) - 1, 0, Wg.size - 2))
            w = float(Pi_w[t_idx, i]); q = float(Pi_q[t_idx, i])
            return q, w

        return actor

    else:
        # legacy RL path (구버전 호환; 일반적으로 사용되지 않음)
        try:
            from .rl import A2CTrainer  # noqa
            pol = A2CTrainer(cfg).train(seed=cfg.seeds[0])
            def actor(obs): return pol.act(obs)
            return actor
        except Exception:
            raise SystemExit("RL route requires --method rl in main() (trainer moved to project.trainer.rl_a2c).")


# --- 공통 유틸 ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def slim_args(args) -> dict:
    keys = [
        "asset", "method", "baseline", "w_max", "fee_annual", "horizon_years",
        "alpha", "lambda_term", "F_target", "p_annual", "g_real_annual",
        "w_fixed", "floor_on", "f_min_real", "es_mode", "outputs",
        "hjb_W_grid", "hjb_Nshock", "hjb_eta_n",
        # hedge
        "hedge", "hedge_mode", "hedge_cost", "hedge_sigma_k", "hedge_tx",
        # market
        "market_mode", "market_csv", "bootstrap_block", "use_real_rf",
        # mortality
        "mortality", "mort_table", "age0", "sex", "bequest_kappa", "bequest_gamma",
        # stage-wise CVaR + RL shaping
        "cvar_stage", "alpha_stage", "lambda_stage", "cstar_mode", "cstar_m",
        "rl_q_cap", "teacher_eps0", "teacher_decay", "lw_scale", "survive_bonus",
        "crra_gamma", "u_scale",
        # XAI
        "xai_on",
        # eval
        "seeds", "n_paths",
        # RL
        "rl_epochs", "rl_steps_per_epoch", "rl_n_paths_eval", "gae_lambda",
        "entropy_coef", "value_coef", "lr", "max_grad_norm",
    ]
    d = {}
    for k in keys:
        v = getattr(args, k, None)
        d[k] = v
    return d


def append_metrics_csv(path: str, payload: dict):
    """fallback: autosave가 없을 때 최소 CSV 기록."""
    row = {
        'ts': now_iso(),
        'asset': payload.get('asset'),
        'method': payload.get('method'),
        'lambda': payload.get('lambda_term'),
        'F_target': payload.get('F_target'),
        'alpha': payload.get('alpha'),
        'ES95': (payload.get('metrics') or {}).get('ES95'),
        'EW': (payload.get('metrics') or {}).get('EW'),
        'Ruin': (payload.get('metrics') or {}).get('Ruin'),
        'mean_WT': (payload.get('metrics') or {}).get('mean_WT'),
        'hedge_on': (payload.get('args') or {}).get('hedge') == 'on',
        'hedge_mode': (payload.get('args') or {}).get('hedge_mode'),
        'fee_annual': payload.get('fee_annual'),
        'w_max': payload.get('w_max'),
        'horizon_years': payload.get('horizon_years'),
        'seeds': (payload.get('args') or {}).get('seeds'),
        'n_paths': (payload.get('args') or {}).get('n_paths'),
        # extras for realism
        'mortality_on': (payload.get('args') or {}).get('mortality') == 'on',
        'market_mode': (payload.get('args') or {}).get('market_mode'),
    }
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# --- 단일 실행 (HJB/Rule 공통) ---
def run_once(args):
    cfg = make_cfg(args)
    ensure_dir(args.outputs)
    actor = build_actor(cfg, args)
    m = evaluate(cfg, actor, es_mode=args.es_mode)
    out = dict(
        asset=cfg.asset, method=args.method, baseline=args.baseline, metrics=m,
        w_max=cfg.w_max, fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term,
        alpha=cfg.alpha, F_target=cfg.F_target, es_mode=args.es_mode,
        n_paths=cfg.n_paths_eval * len(cfg.seeds),
        args=slim_args(args),
    )
    # autosave
    if args.autosave == "on":
        try:
            if _HAS_AUTOSAVE:
                csv_path = save_metrics_autocsv(m, cfg, outputs=cfg.outputs)
                print(f"[autosave] metrics -> {csv_path}")
            else:
                ensure_dir(os.path.join(cfg.outputs, "_logs"))
                csv_path = os.path.join(cfg.outputs, "_logs", "metrics.csv")
                append_metrics_csv(csv_path, out)
                print(f"[autosave:fallback] metrics -> {csv_path}")
        except Exception as e:
            print(f"[autosave] skipped: {e}")
    return out


# --- RL 실행 ---
def run_rl(args):
    cfg = make_cfg(args)
    ensure_dir(args.outputs)
    try:
        from .trainer.rl_a2c import train_rl  # new trainer route
    except Exception as e:
        raise SystemExit(f"RL trainer import failed: {e}")
    fields = train_rl(
        cfg,
        seed_list=args.seeds,
        outputs=args.outputs,
        n_paths_eval=args.rl_n_paths_eval,
        rl_epochs=args.rl_epochs,
        steps_per_epoch=args.rl_steps_per_epoch,
        lr=args.lr,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    out = dict(
        asset=cfg.asset, method="rl", baseline="",
        metrics={"EW": fields.get("EW"), "ES95": fields.get("ES95"),
                 "Ruin": fields.get("Ruin"), "mean_WT": fields.get("mean_WT")},
        w_max=cfg.w_max,
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=cfg.lambda_term,
        alpha=cfg.alpha, F_target=cfg.F_target, es_mode="loss",
        n_paths=args.rl_n_paths_eval * len(args.seeds),
        args=slim_args(args),
    )
    return out


# --- CVaR λ calibration (fast mode + early stop + final high-res) ---
def copy_args(args, **overrides):
    from argparse import Namespace
    d = vars(args).copy()
    d.update(overrides)
    return Namespace(**d)


def calibrate_lambda(args):
    """Binary search on lambda_term to hit ES95(loss) <= cvar_target within cvar_tol."""
    lo, hi = float(args.lambda_min), float(args.lambda_max)
    target = float(args.cvar_target)
    tol = float(args.cvar_tol)
    max_iter = int(getattr(args, "calib_max_iter", 8))
    use_fast = (getattr(args, "calib_fast", "on") == "on")

    history = []
    cache = {}

    def eval_at(lmbd, fast=True):
        if (lmbd, fast) in cache:
            return cache[(lmbd, fast)]
        overrides = dict(lambda_term=float(lmbd), es_mode="loss")
        if fast and use_fast:
            # 저해상도 + 적은 paths + 단일 seed (속도 ↑)
            overrides.update(dict(
                hjb_W_grid=81, hjb_Nshock=128, hjb_eta_n=41,
                n_paths=150, seeds=[args.seeds[0]]
            ))
        local = copy_args(args, **overrides)
        res = run_once(local)
        es = (res.get('metrics') or {}).get('ES95')
        cache[(lmbd, fast)] = (res, es)
        return cache[(lmbd, fast)]

    # 초기 양 끝 평가(FAST)
    res_lo, es_lo = eval_at(lo, fast=True)
    res_hi, es_hi = eval_at(hi, fast=True)

    # 양 끝이 이미 target 이하라면 lo를 채택
    if (es_lo is not None) and (es_hi is not None) and (es_lo <= target) and (es_hi <= target):
        final_res, final_es = eval_at(lo, fast=False)
        final_res['cvar_calibration'] = {
            'selected_lambda': float(lo),
            'selected_ES95': float(final_es) if final_es is not None else None,
            'cvar_target': target,
            'cvar_tol': tol,
            'lambda_min': float(args.lambda_min),
            'lambda_max': float(args.lambda_max),
            'iterations': 1,
            'status': 'already_below_target',
            'history_tail': [{'lambda': float(lo), 'ES95': float(es_lo)}],
        }
        return final_res

    # 필요한 경우 hi 확장(FAST)
    expand = 0
    while (es_lo is not None) and (es_hi is not None) and (es_lo > target) and (es_hi > target) and expand < 3:
        hi *= 2.0
        res_hi, es_hi = eval_at(hi, fast=True)
        expand += 1

    best = (lo, res_lo, es_lo)
    prev_es = None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        res_mid, es_mid = eval_at(mid, fast=True)
        history.append({'lambda': float(mid), 'ES95': float(es_mid) if es_mid is not None else None})

        # plateau 감지
        if prev_es is not None and es_mid is not None and abs(es_mid - prev_es) < 1e-4:
            best = (mid, res_mid, es_mid)
            status = 'plateau'
            break
        prev_es = es_mid

        if es_mid is None:
            lo = mid
            best = (mid, res_mid, es_mid)
            status = 'incomplete'
            continue

        if abs(es_mid - target) <= tol:
            best = (mid, res_mid, es_mid)
            status = 'ok'
            break

        if es_mid > target:
            lo = mid
        else:
            hi = mid
        best = (mid, res_mid, es_mid)
        status = 'ok'

    chosen_lambda, _, _ = best

    # 최종 고해상도 평가
    final_res, final_es = eval_at(chosen_lambda, fast=False)
    final_res['cvar_calibration'] = {
        'selected_lambda': float(chosen_lambda),
        'selected_ES95': float(final_es) if final_es is not None else None,
        'cvar_target': target,
        'cvar_tol': tol,
        'lambda_min': float(args.lambda_min),
        'lambda_max': float(args.lambda_max),
        'iterations': len(history),
        'status': status if 'status' in locals() else 'ok',
        'history_tail': history[-5:],
    }
    return final_res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", type=str, default="KR")
    p.add_argument("--method", type=str, default="hjb", choices=["hjb", "rl", "rule"])
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--w_max", type=float, default=0.70)
    p.add_argument("--fee_annual", type=float, default=0.004)
    p.add_argument("--horizon_years", type=int, default=35)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--lambda_term", type=float, default=0.0)
    p.add_argument("--F_target", type=float, default=0.0)
    p.add_argument("--p_annual", type=float, default=0.04)
    p.add_argument("--g_real_annual", type=float, default=0.02)
    p.add_argument("--w_fixed", type=float, default=0.60)
    p.add_argument("--floor_on", action="store_true")
    p.add_argument("--f_min_real", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--es_mode", type=str, default="wealth", choices=["wealth", "loss"])
    p.add_argument("--outputs", type=str, default="./outputs")

    # === HJB 옵션 ===
    p.add_argument("--hjb_W_grid", type=int, default=None)
    p.add_argument("--hjb_Nshock", type=int, default=None)
    p.add_argument("--hjb_eta_n", type=int, default=None)

    # === Hedge ===
    p.add_argument("--hedge", choices=["on", "off"], default="off")
    p.add_argument("--hedge_mode", choices=["mu", "sigma", "downside"], default="sigma")
    p.add_argument("--hedge_cost", type=float, default=0.005)
    p.add_argument("--hedge_sigma_k", type=float, default=0.20)
    p.add_argument("--hedge_tx", type=float, default=0.0)  # ★ 추가: 발동시 추가비용(연)

    # === Market realism ===
    p.add_argument("--market_mode", choices=["iid", "bootstrap"], default="iid")
    p.add_argument("--market_csv", type=str, default=None)
    p.add_argument("--bootstrap_block", type=int, default=24)
    p.add_argument("--use_real_rf", choices=["on", "off"], default="on")

    # === Mortality ===
    p.add_argument("--mortality", choices=["on", "off"], default="off")
    p.add_argument("--mort_table", type=str, default=None)  # CSV or preset name
    p.add_argument("--age0", type=int, default=65)
    p.add_argument("--sex", choices=["M", "F"], default="M")
    p.add_argument("--bequest_kappa", type=float, default=0.0)
    p.add_argument("--bequest_gamma", type=float, default=1.0)

    # === CVaR calibration ===
    p.add_argument("--cvar_target", type=float, default=CVAR_TARGET_DEFAULT)
    p.add_argument("--cvar_tol", type=float, default=CVAR_TOL_DEFAULT)
    p.add_argument("--lambda_min", type=float, default=LAMBDA_MIN_DEFAULT)
    p.add_argument("--lambda_max", type=float, default=LAMBDA_MAX_DEFAULT)
    p.add_argument("--calib_fast", choices=["on", "off"], default="on")
    p.add_argument("--calib_max_iter", type=int, default=8)

    # === autosave ===
    p.add_argument("--autosave", choices=["on", "off"], default="off")

    # === RL 학습 설정 ===
    p.add_argument("--rl_epochs", type=int, default=60)
    p.add_argument("--rl_steps_per_epoch", type=int, default=2048)
    p.add_argument("--rl_n_paths_eval", type=int, default=300)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    # shaping / constraints
    p.add_argument("--rl_q_cap", type=float, default=0.0)
    p.add_argument("--teacher_eps0", type=float, default=0.0)
    p.add_argument("--teacher_decay", type=float, default=1.0)
    p.add_argument("--lw_scale", type=float, default=0.0)
    p.add_argument("--survive_bonus", type=float, default=0.0)
    p.add_argument("--crra_gamma", type=float, default=3.0)
    p.add_argument("--u_scale", type=float, default=0.0)

    # === Stage-wise CVaR ===
    p.add_argument("--cvar_stage", choices=["on", "off"], default="off")
    p.add_argument("--alpha_stage", type=float, default=0.95)
    p.add_argument("--lambda_stage", type=float, default=0.0)
    p.add_argument("--cstar_mode", choices=["fixed", "annuity", "vpw"], default="annuity")
    p.add_argument("--cstar_m", type=float, default=0.04/12)  # fixed일 때 사용

    # === XAI ===
    p.add_argument("--xai_on", choices=["on", "off"], default="on")

    args = p.parse_args()

    if args.method == "rl":
        out = run_rl(args)
    elif args.method == "hjb" and (args.cvar_target is not None):
        out = calibrate_lambda(args)
    else:
        out = run_once(args)

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
