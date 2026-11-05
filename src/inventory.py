"""Daily inventory simulation under an order-up-to policy.

Timeline for each SKU on each day d (a day is a full cycle):
  1. Orders that were due today arrive and join on-hand stock.
  2. Customers buy. Anything we cannot supply is a lost sale (no backorders).
  3. We look at inventory position = on-hand + already-ordered-but-not-arrived,
     and order enough to bring it up to today's target level S[d].
  4. The order arrives `lead_time` days later, so it can serve demand on day d+lead_time.
Target S[d] therefore has to cover demand on days d+1 ... d+lead_time, which is
exactly what the forecasting module predicts.
"""
import numpy as np
import pandas as pd


def simulate(D_true, S, lead_actual, unit_cost, price, hold_rate=0.25,
             order_cost=0.0, warmup=14):
    """Run the simulation for all SKUs at once.

    D_true      (T, n) units demanded each day
    S           (T, n) order-up-to level chosen at the end of each day
    lead_actual (n,)   days an order really takes to arrive
    unit_cost, price (n,) in rupees; hold_rate = yearly holding cost as a share of unit cost;
    order_cost = fixed cost per order placed (0 by default: with daily review every policy orders
    almost daily, so it would not separate them; orders are still counted). Metrics skip the first `warmup` days.
    """
    D_true = np.asarray(D_true, dtype=float)
    T, n = D_true.shape
    lead = np.asarray(lead_actual, dtype=int)
    cols = np.arange(n)

    arrivals = np.zeros((T + lead.max() + 1, n))
    on_hand = np.asarray(S[0], dtype=float).copy()      # start fully stocked
    pipeline = np.zeros(n)

    oh_hist = np.zeros((T, n))
    lost_hist = np.zeros((T, n))
    orders_hist = np.zeros((T, n))

    for d in range(T):
        arr = arrivals[d]
        on_hand += arr
        pipeline -= arr

        sold = np.minimum(on_hand, D_true[d])
        lost_hist[d] = D_true[d] - sold
        on_hand -= sold

        qty = np.ceil(np.maximum(0.0, S[d] - (on_hand + pipeline)))
        placed = qty > 0
        arrivals[d + lead[placed], cols[placed]] += qty[placed]
        pipeline += qty
        orders_hist[d] = placed
        oh_hist[d] = on_hand

    w = slice(warmup, None)
    demand, lost = D_true[w], lost_hist[w]
    inv_value = (oh_hist[w] * unit_cost).sum(axis=1)     # rupees tied up each day
    holding = float((oh_hist[w] * unit_cost * hold_rate / 365).sum())
    lost_margin = float((lost * (price - unit_cost)).sum())
    n_orders = float(orders_hist[w].sum())
    return {
        "fill_rate": float(1 - lost.sum() / demand.sum()),
        "stockout_rate": float((lost > 0).mean()),
        "avg_inventory_value": float(inv_value.mean()),
        "holding_cost": holding,
        "lost_margin": lost_margin,
        "ordering_cost": n_orders * order_cost,
        "total_cost": holding + lost_margin + n_orders * order_cost,
        "n_orders": n_orders,
    }


def pivot_targets(df, values, n_days, skus, t0):
    """Turn a long (sku, t) table of targets into a (days from t0, SKUs) matrix."""
    s = pd.Series(np.asarray(values), index=pd.MultiIndex.from_arrays([df["t"], df["sku"]]))
    mat = s.unstack("sku").reindex(index=range(t0, n_days), columns=skus)
    return mat.to_numpy()


def interp_at_fill(curve, target_fill):
    """Average inventory value needed to reach `target_fill`, by interpolating a policy's curve."""
    c = curve.sort_values("fill_rate")
    if not (c["fill_rate"].min() <= target_fill <= c["fill_rate"].max()):
        return float("nan")
    return float(np.interp(target_fill, c["fill_rate"], c["avg_inventory_value"]))


def common_fill_targets(curves, policies):
    """Choose comparison targets only within every policy's observed fill-rate range."""
    ranges = curves.groupby("policy")["fill_rate"].agg(["min", "max"])
    if any(policy not in ranges.index for policy in policies):
        return []
    lo = ranges.loc[list(policies), "min"].max()
    hi = ranges.loc[list(policies), "max"].min()
    if lo > hi:
        return []
    return sorted({
        round(lo + 0.15 * (hi - lo), 3),
        round((lo + hi) / 2, 3),
        round(hi - 0.15 * (hi - lo), 3),
    })
