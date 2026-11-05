"""Demand data: a synthetic retail generator and an M5 (Walmart) loader.

Both return the same three objects so the rest of the pipeline does not care
where the data came from:

    D     DataFrame  (days x SKUs)  daily units sold
    P     DataFrame  (days x SKUs)  True on promotion days (assumed to be a known, planned calendar)
    info  DataFrame  one row per SKU: unit_cost, price, lead_time (days)
"""
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FRACTION = 0.62


# --------------------------------------------------------------------------
# Synthetic retail demand
# --------------------------------------------------------------------------
def generate_synthetic(n_skus=60, n_days=900, seed=7, start="2023-01-01"):
    """Daily demand for a small store assortment.

    Each SKU gets its own level, weekly pattern, yearly seasonality, trend,
    promotions (with an uplift) and over-dispersed (negative binomial) noise.
    About 20% of SKUs are slow, intermittent sellers. Costs, prices and
    supplier lead times are drawn per SKU so the inventory maths has
    something realistic to chew on.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    demand, promo_cols, info_rows = {}, {}, []
    for i in range(n_skus):
        sku = f"SKU{i + 1:03d}"
        slow = rng.random() < 0.2
        base = rng.uniform(0.6, 2.0) if slow else rng.lognormal(2.0, 0.8)

        weekly = rng.dirichlet(np.ones(7) * 8) * 7           # multipliers, mean 1
        amp, phase = rng.uniform(0.0, 0.4), rng.uniform(0, 2 * np.pi)
        season = 1 + amp * np.sin(2 * np.pi * doy / 365.25 + phase)
        trend = 1 + rng.uniform(-0.2, 0.3) * t / n_days

        promo = np.zeros(n_days, dtype=bool)
        n_promo = rng.integers(int(n_days / 365 * 2), int(n_days / 365 * 4) + 1)
        for s in rng.choice(n_days - 12, n_promo, replace=False):
            promo[s : s + rng.integers(5, 11)] = True
        uplift = np.where(promo, rng.uniform(1.4, 2.2), 1.0)

        mu = base * weekly[dow] * season * trend * uplift
        k = rng.uniform(5, 15)                                # smaller k = noisier
        demand[sku] = rng.negative_binomial(k, k / (k + mu))
        promo_cols[sku] = promo

        unit_cost = float(np.round(rng.uniform(80, 900), 0))
        info_rows.append(
            dict(
                sku=sku,
                unit_cost=unit_cost,
                price=float(np.round(unit_cost * rng.uniform(1.25, 1.8), 0)),
                lead_time=int(rng.integers(2, 11)),
            )
        )

    D = pd.DataFrame(demand, index=dates)
    P = pd.DataFrame(promo_cols, index=dates)
    info = pd.DataFrame(info_rows).set_index("sku")
    return D, P, info


# --------------------------------------------------------------------------
# M5 (Walmart) loader
# --------------------------------------------------------------------------
def load_m5(m5_dir, store_id="CA_1", n_skus=60, n_days=900, seed=7):
    """Load the Kaggle M5 Forecasting dataset for one store.

    Expects these files in `m5_dir` (download from Kaggle, "M5 Forecasting -
    Accuracy"): calendar.csv, sell_prices.csv and sales_train_evaluation.csv
    (or sales_train_validation.csv).

    Item selection and the assumed unit cost/shelf price use only the first
    TRAIN_FRACTION of the requested date window. M5 contains realized prices,
    not a planned promotion calendar, so no promotion feature is supplied.
    Lead times are drawn reproducibly from 2-10 days using `seed`.
    """
    m5_dir = Path(m5_dir)
    sales_file = next(
        (m5_dir / f for f in ("sales_train_evaluation.csv", "sales_train_validation.csv")
         if (m5_dir / f).exists()),
        None,
    )
    if sales_file is None:
        raise FileNotFoundError(f"No sales_train_*.csv found in {m5_dir}")

    sales = pd.read_csv(sales_file)
    sales = sales[sales["store_id"] == store_id]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    modeled_day_cols = day_cols[-n_days:]
    train_days = modeled_day_cols[:int(TRAIN_FRACTION * len(modeled_day_cols))]
    if not train_days:
        raise ValueError("The requested M5 window is too short for training-period item selection")
    totals = sales[train_days].sum(axis=1)
    top = sales.loc[totals.nlargest(n_skus).index].set_index("item_id")

    cal = pd.read_csv(m5_dir / "calendar.csv", usecols=["d", "date", "wm_yr_wk"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.set_index("d").loc[modeled_day_cols]

    D = top[modeled_day_cols].T
    D.index = pd.DatetimeIndex(cal["date"].to_numpy())
    D.columns.name = None

    prices = pd.read_csv(m5_dir / "sell_prices.csv")
    prices = prices[(prices["store_id"] == store_id) & (prices["item_id"].isin(top.index))]
    week_price = prices.pivot(index="wm_yr_wk", columns="item_id", values="sell_price")
    wk = pd.Series(cal["wm_yr_wk"].to_numpy(), index=D.index)
    price_daily = week_price.reindex(wk.to_numpy())[D.columns]
    price_daily.index = D.index
    price_daily = price_daily.ffill()

    train_prices = price_daily.iloc[:len(train_days)].median()
    if train_prices.isna().any():
        missing = train_prices.index[train_prices.isna()].tolist()
        raise ValueError(f"No training-period prices available for selected M5 items: {missing}")

    rng = np.random.default_rng(seed)
    price = train_prices
    P = pd.DataFrame(False, index=D.index, columns=D.columns)
    info = pd.DataFrame(
        {
            "unit_cost": (price * 0.7).round(2),
            "price": price.round(2),
            "lead_time": rng.integers(2, 11, size=len(price)),
        },
        index=D.columns,
    )
    info.index.name = "sku"
    return D, P, info


def load_data(source="synthetic", m5_dir=None, **kwargs):
    if source == "synthetic":
        return generate_synthetic(**kwargs)
    if source == "m5":
        if m5_dir is None:
            raise ValueError("--m5-dir is required when --data m5")
        return load_m5(m5_dir, **kwargs)
    raise ValueError(f"Unknown data source: {source}")
