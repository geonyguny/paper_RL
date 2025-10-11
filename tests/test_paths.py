def test_bootstrap_paths_diverge(eval_entrypoint):
    out = eval_entrypoint(method="hjb", market_mode="bootstrap", n_paths=5, print_mode="full")
    xs = out["extra"]["eval_WT"]
    assert isinstance(xs, list) and len(xs) == 5
    assert len(set([round(x, 8) for x in xs])) >= 2
