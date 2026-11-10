import numpy as np
import pandas as pd


def simulate(D_true, S, lead_actual, unit_cost, price, hold_rate=0.25,
             order_cost=0.0, warmup=14, detail=False):
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
    if detail:   # per-day, per-SKU arrays (after warm-up), for diagnostics
        return {"on_hand": oh_hist[w], "lost": lost[:], "demand": demand}
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
    s = pd.Series(np.asarray(values), index=pd.MultiIndex.from_arrays([df["t"], df["sku"]]))
    mat = s.unstack("sku").reindex(index=range(t0, n_days), columns=skus)
    return mat.to_numpy()


def interp_at_fill(curve, target_fill):
    c = curve.sort_values("fill_rate")
    if not (c["fill_rate"].min() <= target_fill <= c["fill_rate"].max()):
        return float("nan")
    return float(np.interp(target_fill, c["fill_rate"], c["avg_inventory_value"]))
