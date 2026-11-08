"""Forecast-to-inventory decision engine: run the whole pipeline.

    python run_pipeline.py                       # synthetic data, runs in under a minute
    python run_pipeline.py --data m5 --m5-dir path/to/m5-forecasting-accuracy --out results_m5
    python summarize.py results_m5               # prints the tables for your README

Steps: load demand -> build features -> train lead-time demand models ->
check forecast quality -> turn forecasts into order-up-to levels under three
policies -> simulate inventory on unseen days -> compare stock needed at equal service.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TRAIN_FRACTION, load_data
from src.forecast import (
    MIN_HISTORY, Q_ALPHAS, REPORT_ALPHAS, TARGET_CLIP, Z_ALPHAS, LeadTimeForecaster,
    baseline_quantile, coverage, error_based_quantile, make_features, pinball, training_ratio, wape,
)
from src.inventory import interp_at_fill, pivot_targets, simulate

POLICIES = {
    "baseline": "Moving average + normal safety stock",
    "ml_error": "ML mean forecast + safety stock from its own errors",
    "ml_quantile": "ML quantile forecast (direct order-up-to level)",
}
# Each policy is swept over its own service-level settings. They are wide enough that the three
# fill-rate ranges overlap, which the equal-fill comparison needs.
POLICY_ALPHAS = {"baseline": Z_ALPHAS, "ml_error": Z_ALPHAS, "ml_quantile": Q_ALPHAS}
UNITS = {"synthetic": ("₹ lakh", 1e5), "m5": ("US$ thousand", 1e3)}   # label, divisor for charts and tables
SENS_ALPHA = 0.95


def policy_targets(df, preds, sigma, alpha, policy):
    """Order-up-to level for every row of df under one policy."""
    if policy == "baseline":
        return baseline_quantile(df, alpha)
    if policy == "ml_error":
        return error_based_quantile(df, preds["mean"], sigma, alpha)
    return preds[alpha]


def build_S(df, preds, sigma, alpha, policy, n_days, skus, t0):
    return pivot_targets(df, policy_targets(df, preds, sigma, alpha, policy), n_days, skus, t0)


def equal_fill_table(curves):
    """Stock each policy needs to reach the same fill rate, over the range all three can reach."""
    cols = ["fill_rate_target", *[f"{p}_inventory_value" for p in POLICIES],
            "ml_quantile_saving_vs_baseline_pct", "ml_error_saving_vs_baseline_pct"]
    spans = {p: (g["fill_rate"].min(), g["fill_rate"].max()) for p, g in curves.groupby("policy")}
    lo, hi = max(s[0] for s in spans.values()), min(s[1] for s in spans.values())
    if hi - lo < 0.005:     # the policies never reach the same service level: say so, don't guess
        return pd.DataFrame(columns=cols), (lo, hi)
    # practical service levels first; if none is reachable by all three, fall back to the middle of the shared range
    targets = [t for t in (0.93, 0.95, 0.97, 0.98) if lo <= t <= hi]
    if not targets:
        targets = sorted({round(lo + f * (hi - lo), 3) for f in (0.15, 0.5, 0.85)})
    rows = []
    for target in targets:
        inv = {p: interp_at_fill(curves[curves.policy == p], target) for p in POLICIES}
        rows.append({
            "fill_rate_target": target,
            **{f"{p}_inventory_value": v for p, v in inv.items()},
            "ml_quantile_saving_vs_baseline_pct": 100 * (1 - inv["ml_quantile"] / inv["baseline"]),
            "ml_error_saving_vs_baseline_pct": 100 * (1 - inv["ml_error"] / inv["baseline"]),
        })
    return pd.DataFrame(rows, columns=cols), (lo, hi)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unit_label, unit_scale = UNITS[args.data]

    # ---------------------------------------------------------------- data --
    extra = {"store_id": args.store} if args.data == "m5" else {}
    D, P, info = load_data(args.data, m5_dir=args.m5_dir, n_days=args.days, n_skus=args.skus, seed=args.seed, **extra)
    n_days, skus = len(D), list(D.columns)
    lead = info.loc[skus, "lead_time"]
    t1, t2 = int(TRAIN_FRACTION * n_days), int(0.78 * n_days)   # train | validation | test
    print(f"{len(skus)} SKUs x {n_days} days  |  train < day {t1}, validation < day {t2}, test = last {n_days - t2} days")

    # ------------------------------------------------------------ forecasts --
    feats = make_features(D, P, lead, np.arange(MIN_HISTORY, n_days))
    known = feats["y"].notna()
    train = feats[known & (feats["t"] + feats["H"] < t1)]
    val = feats[known & (feats["t"] >= t1) & (feats["t"] + feats["H"] < t2)]
    test = feats[known & (feats["t"] >= t2)]
    print(f"rows: train {len(train):,}  validation {len(val):,}  test {len(test):,}")

    fc = LeadTimeForecaster(target_cap=args.target_cap).fit(train)
    val_pred = fc.predict(val)
    sigma = (val["y"] - val_pred["mean"]).groupby(val["sku"]).std().fillna(0.0)

    # forecast quality on the unseen test period
    te_pred, y = fc.predict(test), test["y"].to_numpy()
    rows = [
        {"metric": "WAPE (mean forecast)", "alpha": "-",
         "baseline": wape(y, test["base"].to_numpy()), "ml_error": wape(y, te_pred["mean"]),
         "ml_quantile": np.nan}
    ]
    for a in REPORT_ALPHAS:
        rows.append({
            "metric": "pinball loss", "alpha": a,
            "baseline": pinball(y, baseline_quantile(test, a), a),
            "ml_error": pinball(y, error_based_quantile(test, te_pred["mean"], sigma, a), a),
            "ml_quantile": pinball(y, te_pred[a], a),
        })
        rows.append({
            "metric": "coverage (target = alpha)", "alpha": a,
            "baseline": coverage(y, baseline_quantile(test, a)),
            "ml_error": coverage(y, error_based_quantile(test, te_pred["mean"], sigma, a)),
            "ml_quantile": coverage(y, te_pred[a]),
        })
    accuracy = pd.DataFrame(rows)
    accuracy.to_csv(out / "forecast_accuracy.csv", index=False)
    print("\nForecast quality on unseen test days")
    print(accuracy.round(3).to_string(index=False))

    # data health: the numbers that explain odd results (extreme targets, zero-sales stretches, bias)
    ratio = (train["y"] / train["base"])
    diagnostics = {
        "train_target_ratio_median": float(ratio.median()),
        "train_target_ratio_p99": float(ratio.quantile(0.99)),
        "train_target_ratio_max": float(ratio.max()),
        "train_rows_ratio_above_cap_pct": float(100 * (ratio > args.target_cap).mean()) if args.target_cap else 0.0,
        "train_rows_with_zero_28d_sales_pct": float(100 * (train["mean28"] == 0).mean()),
        "test_ml_mean_bias_pct": float(100 * (te_pred["mean"].sum() / y.sum() - 1)),
        "test_baseline_bias_pct": float(100 * (test["base"].sum() / y.sum() - 1)),
        "target_cap_used": args.target_cap,
    }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print("\nData health")
    print(f"  training target ratio (lead-time demand / recent run-rate): median {diagnostics['train_target_ratio_median']:.2f}, "
          f"99th pct {diagnostics['train_target_ratio_p99']:.1f}, max {diagnostics['train_target_ratio_max']:.0f}; "
          + (f"{diagnostics['train_rows_ratio_above_cap_pct']:.2f}% of rows above the cap of {args.target_cap:g}"
             if args.target_cap else "cap is OFF"))
    print(f"  training rows with no sales in the previous 28 days: {diagnostics['train_rows_with_zero_28d_sales_pct']:.2f}%")
    print(f"  test bias (forecast total vs actual total): ML mean {diagnostics['test_ml_mean_bias_pct']:+.1f}%, "
          f"moving average {diagnostics['test_baseline_bias_pct']:+.1f}%")

    # ----------------------------------------------------------- simulation --
    unit_cost, price = info.loc[skus, "unit_cost"].to_numpy(), info.loc[skus, "price"].to_numpy()
    sim_feats = make_features(D, P, lead, np.arange(t2, n_days))
    sim_preds = fc.predict(sim_feats)
    D_test = D.iloc[t2:].to_numpy()

    curve_rows = []
    for policy, alphas in POLICY_ALPHAS.items():
        for a in alphas:
            S = build_S(sim_feats, sim_preds, sigma, a, policy, n_days, skus, t2)
            res = simulate(D_test, S, lead.to_numpy(), unit_cost, price)
            curve_rows.append({"policy": policy, "alpha": a, **res})
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(out / "policy_curves.csv", index=False)

    # --------------------------------------------------------- sensitivity --
    scenarios = {}
    scenarios["As planned"] = (D, lead.to_numpy())
    scenarios["Suppliers 2 days slower than planned"] = (D, (lead + 2).to_numpy())
    shocked = D.copy()
    shocked.iloc[t2 + 30:] = np.round(shocked.iloc[t2 + 30:] * 1.25)
    scenarios["Demand jumps 25% after test day 30"] = (shocked, lead.to_numpy())

    sens_rows = []
    for name, (D_s, lead_actual) in scenarios.items():
        sf = make_features(D_s, P, lead, np.arange(t2, n_days))   # forecasts only see the shocked history
        sp = fc.predict(sf)
        for policy in POLICIES:
            S = build_S(sf, sp, sigma, SENS_ALPHA, policy, n_days, skus, t2)
            res = simulate(D_s.iloc[t2:].to_numpy(), S, lead_actual, unit_cost, price)
            sens_rows.append({"scenario": name, "policy": policy, **res})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out / "sensitivity.csv", index=False)

    # ------------------------------------------------------------- summary --
    headline, (lo, hi) = equal_fill_table(curves)
    headline.to_csv(out / "headline.csv", index=False)

    print("\nPolicy curves (test period)")
    show = ["policy", "alpha", "fill_rate", "stockout_rate", "avg_inventory_value", "holding_cost", "lost_margin", "total_cost"]
    print(curves[show].round(3).to_string(index=False))
    print(f"\nSensitivity at alpha = {SENS_ALPHA}")
    print(sens[["scenario", "policy", "fill_rate", "avg_inventory_value", "total_cost"]].round(3).to_string(index=False))
    print("\nStock needed to hit the same fill rate (interpolated; positive saving = less stock than baseline)")
    if headline.empty:
        print(f"  The policies never reach a common fill rate (ranges overlap {lo:.3f}..{hi:.3f}), so no like-for-like comparison is possible.")
    else:
        print(headline.round(3).to_string(index=False))

    (out / "run_info.json").write_text(json.dumps(
        {"data": args.data, "n_skus": len(skus), "n_days": n_days, "train_end": t1, "test_start": t2,
         "n_test_days": n_days - t2, "unit_label": unit_label, "unit_scale": unit_scale,
         "target_cap": args.target_cap, "common_fill_range": [round(float(lo), 4), round(float(hi), 4)]}, indent=2))

    if not args.no_charts:
        from src.report import make_figures
        make_figures(out, accuracy, curves, sens, headline, unit=(unit_label, unit_scale))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "m5"], default="synthetic")
    ap.add_argument("--m5-dir", default=None)
    ap.add_argument("--store", default="CA_1", help="M5 store id, e.g. CA_1, TX_2, WI_3 (only used with --data m5)")
    ap.add_argument("--skus", type=int, default=60)
    ap.add_argument("--days", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-cap", type=float, default=TARGET_CLIP,
                    help="cap on the training target (lead-time demand / recent run-rate); 0 switches the cap off")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-charts", action="store_true")
    main(ap.parse_args())
