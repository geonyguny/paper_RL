from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, List

from .cvar_utils import maybe_extract_WT

# ---- evaluate 동적 임포트 ----
def _import_evaluate():
    candidates = [
        "project.runner.evaluate",
        "project.evaluate",
        "project.runner.eval",
        "project.eval",
    ]
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["evaluate"])
            return getattr(mod, "evaluate")
        except Exception:
            continue
    return None

_evaluate = _import_evaluate()  # type: ignore

# ---- n_paths 추정 ----
def _estimate_n_paths(args, out: Dict[str, Any]) -> Optional[int]:
    try:
        if isinstance(out, dict):
            np_exist = out.get("n_paths")
            if isinstance(np_exist, (int, float)) and int(np_exist) > 0:
                return int(np_exist)
        seeds = getattr(args, "seeds", [])
        if isinstance(seeds, int):
            n_seeds = 1
        elif isinstance(seeds, (list, tuple)):
            n_seeds = max(1, len(seeds))
        else:
            n_seeds = 1
        if str(getattr(args, "method", "hjb")).lower() == "rl":
            n_eval = int(getattr(args, "rl_n_paths_eval", 0) or 0)
            if n_eval > 0:
                return n_eval * n_seeds
        n_base = int(getattr(args, "n_paths", 0) or 0)
        if n_base > 0:
            return n_base * n_seeds
    except Exception:
        pass
    return None

# ---- stdout 축소/요약 ----
def prune_for_stdout(args, out: Dict[str, Any]) -> Any:
    def _sel_metrics(md: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
        return {k: md[k] for k in md if k in keys}

    if getattr(args, "no_paths", False) and isinstance(out, dict):
        out = dict(out)
        extra = out.get("extra")
        if isinstance(extra, dict):
            for k in ("eval_WT", "ruin_flags"):
                if k in extra and isinstance(extra[k], (list, tuple)):
                    try:
                        extra[k + "_n"] = len(extra[k])
                    except Exception:
                        pass
                    del extra[k]
            out["extra"] = extra

    mode = str(getattr(args, "print_mode", "full")).lower()
    if mode == "full":
        return out

    metrics = out["metrics"] if isinstance(out, dict) and isinstance(out.get("metrics"), dict) else out
    keys = [s.strip() for s in str(getattr(args, "metrics_keys", "")).split(",") if s.strip()]

    n_paths_guess = _estimate_n_paths(args, out)
    age0_guess, sex_guess = None, None
    try:
        if isinstance(out, dict):
            age0_guess = out.get("age0")
            sex_guess = out.get("sex")
        if age0_guess is None:
            age0_guess = getattr(args, "age0", None)
        if sex_guess is None:
            sex_guess = getattr(args, "sex", None)
    except Exception:
        pass

    if mode == "metrics":
        mini = _sel_metrics(metrics, keys)
        if isinstance(out, dict):
            mini["time_total_s"] = out.get("time_total_s")
            mini["time_total_hms"] = out.get("time_total_hms")
        mini.update({
            "tag": (out.get("tag") if isinstance(out, dict) else None) or getattr(args, "tag", None),
            "asset": (out.get("asset") if isinstance(out, dict) else None) or getattr(args, "asset", None),
            "method": (out.get("method") if isinstance(out, dict) else None) or getattr(args, "method", None),
            "n_paths": n_paths_guess,
        })
        return mini

    if mode == "summary":
        args_dict = out.get("args", {}) if isinstance(out, dict) else {}
        top_tag    = (out.get("tag") if isinstance(out, dict) else None) or getattr(args, "tag", None)
        top_method = (out.get("method") if isinstance(out, dict) else None) or getattr(args, "method", None)
        top_asset  = (out.get("asset") if isinstance(out, dict) else None) or getattr(args, "asset", None)
        summary_obj = {
            "tag": top_tag,
            "asset": top_asset,
            "method": top_method,
            "age0": age0_guess if age0_guess is not None else (args_dict or {}).get("age0"),
            "sex": sex_guess if sex_guess is not None else (args_dict or {}).get("sex"),
            "metrics": _sel_metrics(metrics, keys),
            "n_paths": n_paths_guess,
            "T": (out.get("extra") or {}).get("T") if isinstance(out, dict) else None,
            "time_total_s": out.get("time_total_s") if isinstance(out, dict) else None,
            "time_total_hms": out.get("time_total_hms") if isinstance(out, dict) else None,
        }
        if isinstance(out, dict) and isinstance(out.get("cvar_calibration"), dict):
            summary_obj["cvar_calibration"] = out["cvar_calibration"]
        return summary_obj

    return out

# ---- cfg/actor 추출 ----
def try_extract_cfg_actor(res: Any) -> Tuple[Optional[Any], Optional[Any]]:
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    cfg = getattr(res, "cfg", None)
    actor = getattr(res, "actor", None) or getattr(res, "policy", None)
    if cfg is not None or actor is not None:
        return cfg, actor
    if isinstance(res, dict):
        cfg = res.get("args") or res.get("cfg")
        actor = res.get("actor") or res.get("policy") or res.get("pi")
        if cfg is not None or actor is not None:
            return cfg, actor
    return None, None

# ---- 평가 결과 패킹(+필요시 경로 재평가) ----
def maybe_evaluate_with_es_mode(res: Any, es_mode: str, want_paths: bool = False) -> Dict[str, Any]:
    if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], dict):
        pack: Dict[str, Any] = {"metrics": dict(res[0])}
        if len(res) >= 2 and isinstance(res[1], dict):
            pack["extra"] = dict(res[1])
        if "es_mode" not in pack["metrics"]:
            pack["metrics"]["es_mode"] = str(es_mode).lower()
    elif isinstance(res, dict):
        if "metrics" in res and isinstance(res["metrics"], dict):
            pack = {"metrics": dict(res["metrics"])}
            if isinstance(res.get("extra"), dict):
                pack["extra"] = dict(res["extra"])
            for k in ("asset","method","w_max","fee_annual","lambda_term","alpha","F_target","outputs","tag","n_paths","args"):
                if k in res:
                    pack[k] = res[k]
            if "es_mode" not in pack["metrics"]:
                pack["metrics"]["es_mode"] = str(es_mode).lower()
        else:
            pack = {"metrics": dict(res)}
            if "es_mode" not in pack["metrics"]:
                pack["metrics"]["es_mode"] = str(es_mode).lower()
    else:
        pack = {
            "result": "ok",
            "note": "evaluate not executed in cli (no evaluate import or unexpected return type).",
            "es_mode": str(es_mode).lower(),
        }

    have_paths = False
    try:
        wt0 = maybe_extract_WT({"metrics": pack.get("metrics", {}), "extra": pack.get("extra", {})})
        have_paths = wt0 is not None and len(list(wt0)) > 0
    except Exception:
        have_paths = False

    if want_paths and not have_paths and _evaluate is not None:
        cfg, actor = try_extract_cfg_actor(res)
        if cfg is None:
            cfg, actor = try_extract_cfg_actor(pack)
        if cfg is not None:
            try:
                m = None
                try:
                    m = _evaluate(cfg, actor, es_mode=str(es_mode).lower(), return_paths=True)  # type: ignore
                except TypeError:
                    m = _evaluate(cfg, actor, es_mode=str(es_mode).lower())  # type: ignore

                if isinstance(m, tuple) and len(m) >= 1 and isinstance(m[0], dict):
                    pack["metrics"] = m[0]
                    if len(m) >= 2 and isinstance(m[1], dict):
                        pack["extra"] = m[1]
                elif isinstance(m, dict):
                    pack["metrics"] = m

                if "es_mode" not in pack.get("metrics", {}):
                    pack["metrics"]["es_mode"] = str(es_mode).lower()

                wt_paths = maybe_extract_WT({"metrics": pack.get("metrics", {}), "extra": pack.get("extra", {})})
                if wt_paths is not None:
                    if "extra" not in pack or not isinstance(pack["extra"], dict):
                        pack["extra"] = {}
                    pack["extra"]["eval_WT"] = list(wt_paths)
            except Exception:
                pass

    return pack
