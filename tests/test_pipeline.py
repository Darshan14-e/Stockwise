"""Sanity checks. Run with:  python -m pytest -q   (or just: python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import generate_synthetic, load_m5
from src.forecast import FEATURES, make_features
from src.inventory import common_fill_targets, simulate


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


def test_m5_loader_selects_items_and_reference_prices_from_training_period(tmp_path):
    sales = pd.DataFrame(
        [
            {"item_id": "early", "store_id": "CA_1", "d_1": 5, "d_2": 5, "d_3": 5, "d_4": 0, "d_5": 0, "d_6": 0},
            {"item_id": "late", "store_id": "CA_1", "d_1": 1, "d_2": 1, "d_3": 1, "d_4": 100, "d_5": 100, "d_6": 100},
        ]
    )
    sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, 7)],
            "date": pd.date_range("2020-01-01", periods=6).astype(str),
            "wm_yr_wk": range(1, 7),
        }
    ).to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame(
        {
            "store_id": ["CA_1"] * 6,
            "item_id": ["early"] * 6,
            "wm_yr_wk": range(1, 7),
            "sell_price": [10, 20, 30, 1000, 1000, 1000],
        }
    ).to_csv(tmp_path / "sell_prices.csv", index=False)

    demand, promotions, info = load_m5(tmp_path, n_skus=1, n_days=6)

    assert list(demand.columns) == ["early"]
    assert info.loc["early", "price"] == 20
    assert info.loc["early", "unit_cost"] == 14
    assert not promotions.to_numpy().any()


def test_common_fill_targets_are_empty_when_policy_ranges_do_not_overlap():
    curves = pd.DataFrame(
        {
            "policy": ["baseline", "baseline", "ml_error", "ml_error"],
            "fill_rate": [0.90, 0.97, 0.975, 0.99],
        }
    )
    assert common_fill_targets(curves, ["baseline", "ml_error"]) == []


def test_common_fill_targets_stay_within_all_policy_ranges():
    curves = pd.DataFrame(
        {
            "policy": ["baseline", "baseline", "ml_error", "ml_error"],
            "fill_rate": [0.90, 0.96, 0.93, 0.98],
        }
    )
    assert np.allclose(common_fill_targets(curves, ["baseline", "ml_error"]), [0.934, 0.945, 0.956])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
