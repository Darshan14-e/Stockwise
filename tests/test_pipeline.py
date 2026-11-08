"""Sanity checks. Run with:  python -m pytest -q   (or just: python tests/test_pipeline.py)"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pipeline import equal_fill_table
from src.data import generate_synthetic
from src.forecast import FEATURES, Q_ALPHAS, TARGET_CLIP, LeadTimeForecaster, make_features, training_ratio
from src.inventory import simulate
from src.report import make_figures


def _sim(demand, S, lead, n=1):
    cost, price = np.full(n, 100.0), np.full(n, 150.0)
    return simulate(demand, S, np.full(n, lead), cost, price, warmup=0)


def test_cover_exactly_lead_time_demand_means_no_stockouts():
    T = 60
    res = _sim(np.ones((T, 1)), np.full((T, 1), 3.0), lead=3)   # 1 unit/day, 3-day lead time
    assert res["fill_rate"] == 1.0


def test_zero_target_sells_nothing():
    T = 40
    res = _sim(np.ones((T, 1)), np.zeros((T, 1)), lead=2)
    assert res["fill_rate"] == 0.0 and res["n_orders"] == 0


def test_huge_target_never_stocks_out():
    rng = np.random.default_rng(0)
    d = rng.poisson(5, size=(100, 3)).astype(float)
    res = simulate(d, np.full((100, 3), 1e6), np.array([2, 5, 9]), np.ones(3), np.ones(3) * 2, warmup=0)
    assert res["fill_rate"] == 1.0


def test_features_do_not_peek_at_the_future():
    D, P, info = generate_synthetic(n_skus=5, n_days=300, seed=3)
    t_idx = np.arange(60, 200)
    a = make_features(D, P, info["lead_time"], t_idx)

    D2 = D.copy()
    D2.iloc[200:] = 999                       # scramble everything after the last decision day
    b = make_features(D2, P, info["lead_time"], t_idx)

    # Targets for late rows reach into the scrambled period, but features must not move.
    pd.testing.assert_frame_equal(a[FEATURES + ["base", "mean28", "std28"]], b[FEATURES + ["base", "mean28", "std28"]])


def test_extreme_training_targets_are_capped():
    # after a stretch of zero sales the recent run-rate is tiny, so demand / run-rate can be enormous
    df = pd.DataFrame({"y": [10.0, 5000.0, 0.0], "base": [10.0, 1.0, 5.0]})
    r = training_ratio(df)
    assert r.max() == TARGET_CLIP and r.iloc[0] == 1.0 and r.iloc[2] == 0.0


def test_quantile_predictions_never_cross():
    D, P, info = generate_synthetic(n_skus=8, n_days=400, seed=5)
    feats = make_features(D, P, info["lead_time"], np.arange(56, 400))
    train = feats[feats["y"].notna()]
    pred = LeadTimeForecaster().fit(train).predict(train.head(500))
    stacked = np.column_stack([pred[a] for a in Q_ALPHAS])
    assert (np.diff(stacked, axis=1) >= -1e-9).all()      # higher service level never means a lower target


def _toy_curves(spans):
    rows = []
    for policy, (lo, hi) in spans.items():
        for f in np.linspace(lo, hi, 6):
            rows.append({"policy": policy, "alpha": f, "fill_rate": f, "avg_inventory_value": 1000 * f})
    return pd.DataFrame(rows)


def test_equal_fill_table_says_so_when_policies_never_overlap():
    curves = _toy_curves({"baseline": (0.90, 0.95), "ml_error": (0.97, 0.99), "ml_quantile": (0.95, 0.98)})
    table, _ = equal_fill_table(curves)
    assert table.empty                                    # no like-for-like comparison, and no crash


def test_equal_fill_table_uses_practical_service_levels_when_they_overlap():
    curves = _toy_curves({"baseline": (0.85, 0.99), "ml_error": (0.90, 0.995), "ml_quantile": (0.88, 0.99)})
    table, _ = equal_fill_table(curves)
    assert list(table["fill_rate_target"]) == [0.93, 0.95, 0.97, 0.98]
    assert table.notna().all().all()


def test_charts_are_still_drawn_when_there_is_no_headline_comparison():
    curves = _toy_curves({"baseline": (0.90, 0.95), "ml_error": (0.97, 0.99), "ml_quantile": (0.95, 0.98)})
    acc = pd.DataFrame([{"metric": "coverage (target = alpha)", "alpha": a, "baseline": a - 0.05,
                         "ml_error": a, "ml_quantile": a - 0.01} for a in (0.8, 0.9, 0.95)])
    sens = pd.DataFrame([{"scenario": s, "policy": p, "total_cost": 1000.0 + i}
                         for s in ("As planned", "Slower suppliers") for i, p in enumerate(("baseline", "ml_error", "ml_quantile"))])
    empty = pd.DataFrame(columns=["fill_rate_target"])
    with tempfile.TemporaryDirectory() as tmp:
        make_figures(Path(tmp), acc, curves, sens, empty, unit=("US$ thousand", 1e3))
        assert (Path(tmp) / "fig_tradeoff.png").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
