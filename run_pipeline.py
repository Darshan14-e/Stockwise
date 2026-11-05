"""Forecast-to-inventory decision engine.

Run with synthetic data:

    python run_pipeline.py

Run with M5 data:

    python run_pipeline.py --data m5 --m5-dir path/to/m5-data

Pipeline:

1. Load demand data
2. Build time-safe forecasting features
3. Train mean and quantile demand models
4. Evaluate forecast quality on unseen test data
5. Convert forecasts into inventory targets
6. Simulate inventory policies
7. Run sensitivity scenarios
8. Generate charts and result files
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TRAIN_FRACTION, load_data

from src.forecast import (
    ALPHAS,
    MIN_HISTORY,
    LeadTimeForecaster,
    baseline_quantile,
    coverage,
    error_based_quantile,
    make_features,
    pinball,
    wape,
)

from src.inventory import (
    common_fill_targets,
    interp_at_fill,
    pivot_targets,
    simulate,
)


POLICIES = {
    "baseline": "Moving average + normal safety stock",
    "ml_error": "ML mean forecast + safety stock from its own errors",
    "ml_quantile": "ML quantile forecast (direct order-up-to level)",
}

SENS_ALPHA = 0.95


# ---------------------------------------------------------------------
# Inventory target policies
# ---------------------------------------------------------------------

def policy_targets(df, forecaster, sigma, alpha, policy):
    """Calculate an order-up-to target for each SKU/day row."""

    if policy == "baseline":
        return baseline_quantile(df, alpha)

    preds = forecaster.predict(df)

    if policy == "ml_error":
        return error_based_quantile(
            df,
            preds["mean"],
            sigma,
            alpha
        )

    # Direct ML quantile forecast.
    return preds[alpha]


def build_S(
    df,
    forecaster,
    sigma,
    alpha,
    policy,
    n_days,
    skus,
    t0,
):
    """Convert long-format policy targets into a day x SKU matrix."""

    targets = policy_targets(
        df,
        forecaster,
        sigma,
        alpha,
        policy,
    )

    return pivot_targets(
        df,
        targets,
        n_days,
        skus,
        t0,
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main(args):

    # Create the output directory.
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------

    D, P, info = load_data(
        args.data,
        m5_dir=args.m5_dir,
        n_days=args.days,
        n_skus=args.skus,
        seed=args.seed,
    )

    n_days = len(D)
    skus = list(D.columns)

    lead = info.loc[skus, "lead_time"]

    # Data split:
    # Training -> validation -> unseen test period.
    t1 = int(TRAIN_FRACTION * n_days)
    t2 = int(0.78 * n_days)

    print(
        f"{len(skus)} SKUs x {n_days} days"
        f"  |  train < day {t1},"
        f" validation < day {t2},"
        f" test = last {n_days - t2} days"
    )

    # -------------------------------------------------------------
    # 2. Build forecasting features
    # -------------------------------------------------------------

    feats = make_features(
        D,
        P,
        lead,
        np.arange(MIN_HISTORY, n_days),
    )

    known = feats["y"].notna()

    # Training rows must have their complete lead-time target
    # before the end of the training period.
    train = feats[
        known
        & (feats["t"] + feats["H"] < t1)
    ]

    val = feats[
        known
        & (feats["t"] >= t1)
        & (feats["t"] + feats["H"] < t2)
    ]

    test = feats[
        known
        & (feats["t"] >= t2)
    ]

    print(
        f"rows: train {len(train):,}"
        f"  validation {len(val):,}"
        f"  test {len(test):,}"
    )

    # -------------------------------------------------------------
    # 3. Train forecasting models
    # -------------------------------------------------------------

    fc = LeadTimeForecaster().fit(train)

    # Validation forecast errors are used to estimate
    # safety stock for the ML-error policy.
    val_pred = fc.predict(val)

    sigma = (
        val["y"] - val_pred["mean"]
    ).groupby(
        val["sku"]
    ).std().fillna(0.0)

    # -------------------------------------------------------------
    # 4. Forecast quality on unseen test data
    # -------------------------------------------------------------

    te_pred = fc.predict(test)
    y = test["y"].to_numpy()

    rows = [
        {
            "metric": "WAPE (mean forecast)",
            "alpha": "-",

            "baseline": wape(
                y,
                test["base"].to_numpy()
            ),

            "ml_error": wape(
                y,
                te_pred["mean"]
            ),

            "ml_quantile": np.nan,
        }
    ]

    for alpha in ALPHAS:

        # Quantile forecast quality.
        rows.append(
            {
                "metric": "pinball loss",
                "alpha": alpha,

                "baseline": pinball(
                    y,
                    baseline_quantile(test, alpha),
                    alpha,
                ),

                "ml_error": pinball(
                    y,
                    error_based_quantile(
                        test,
                        te_pred["mean"],
                        sigma,
                        alpha,
                    ),
                    alpha,
                ),

                "ml_quantile": pinball(
                    y,
                    te_pred[alpha],
                    alpha,
                ),
            }
        )

        # Coverage should ideally be close to alpha.
        rows.append(
            {
                "metric": "coverage (target = alpha)",
                "alpha": alpha,

                "baseline": coverage(
                    y,
                    baseline_quantile(test, alpha),
                ),

                "ml_error": coverage(
                    y,
                    error_based_quantile(
                        test,
                        te_pred["mean"],
                        sigma,
                        alpha,
                    ),
                ),

                "ml_quantile": coverage(
                    y,
                    te_pred[alpha],
                ),
            }
        )

    accuracy = pd.DataFrame(rows)

    accuracy.to_csv(
        out / "forecast_accuracy.csv",
        index=False,
    )

    print("\nForecast quality on unseen test days")
    print(
        accuracy.round(3).to_string(index=False)
    )

    # -------------------------------------------------------------
    # 5. Inventory policy simulation
    # -------------------------------------------------------------

    unit_cost = (
        info.loc[skus, "unit_cost"]
        .to_numpy()
    )

    price = (
        info.loc[skus, "price"]
        .to_numpy()
    )

    # Build decision features for every test day.
    sim_feats = make_features(
        D,
        P,
        lead,
        np.arange(t2, n_days),
    )

    D_test = D.iloc[t2:].to_numpy()

    curve_rows = []

    for policy in POLICIES:

        for alpha in ALPHAS:

            S = build_S(
                sim_feats,
                fc,
                sigma,
                alpha,
                policy,
                n_days,
                skus,
                t2,
            )

            res = simulate(
                D_test,
                S,
                lead.to_numpy(),
                unit_cost,
                price,
            )

            curve_rows.append(
                {
                    "policy": policy,
                    "alpha": alpha,
                    **res,
                }
            )

    curves = pd.DataFrame(curve_rows)

    curves.to_csv(
        out / "policy_curves.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 6. Sensitivity analysis
    # -------------------------------------------------------------

    scenarios = {}

    # Normal operating conditions.
    scenarios["As planned"] = (
        D,
        lead.to_numpy(),
    )

    # Suppliers take two additional days.
    scenarios["Suppliers 2 days slower than planned"] = (
        D,
        (lead + 2).to_numpy(),
    )

    # Demand increases by 25% after 30 days of testing.
    shocked = D.copy()

    shocked.iloc[t2 + 30:] = np.round(
        shocked.iloc[t2 + 30:] * 1.25
    )

    scenarios["Demand jumps 25% after test day 30"] = (
        shocked,
        lead.to_numpy(),
    )

    sens_rows = []

    for name, (D_s, lead_actual) in scenarios.items():

        # IMPORTANT:
        # Features are rebuilt using the scenario demand.
        # At each decision day, make_features only uses historical
        # demand available up to that day, so it does not peek into
        # future demand. This allows the inventory policy to adapt
        # after a shock becomes observable.
        sf = make_features(
            D_s,
            P,
            lead,
            np.arange(t2, n_days),
        )

        for policy in POLICIES:

            S = build_S(
                sf,
                fc,
                sigma,
                SENS_ALPHA,
                policy,
                n_days,
                skus,
                t2,
            )

            res = simulate(
                D_s.iloc[t2:].to_numpy(),
                S,
                lead_actual,
                unit_cost,
                price,
            )

            sens_rows.append(
                {
                    "scenario": name,
                    "policy": policy,
                    **res,
                }
            )

    sens = pd.DataFrame(sens_rows)

    sens.to_csv(
        out / "sensitivity.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 7. Same-service inventory comparison
    # -------------------------------------------------------------

    headline = []

    targets = common_fill_targets(
        curves,
        POLICIES,
    )

    if not targets:

        print(
            "\nPolicies have no shared test fill-rate range; "
            "same-service inventory comparison is unavailable."
        )

    for target in targets:

        inv = {
            policy: interp_at_fill(
                curves[
                    curves.policy == policy
                ],
                target,
            )
            for policy in POLICIES
        }

        headline.append(
            {
                "fill_rate_target": target,

                **{
                    f"{policy}_inventory_value": value
                    for policy, value in inv.items()
                },

                "ml_quantile_saving_vs_baseline_pct":
                    100 * (
                        1
                        - inv["ml_quantile"]
                        / inv["baseline"]
                    ),

                "ml_error_saving_vs_baseline_pct":
                    100 * (
                        1
                        - inv["ml_error"]
                        / inv["baseline"]
                    ),
            }
        )

    headline = pd.DataFrame(
        headline,
        columns=[
            "fill_rate_target",
            "baseline_inventory_value",
            "ml_error_inventory_value",
            "ml_quantile_inventory_value",
            "ml_quantile_saving_vs_baseline_pct",
            "ml_error_saving_vs_baseline_pct",
        ],
    )

    headline.to_csv(
        out / "headline.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 8. Print results
    # -------------------------------------------------------------

    print("\nPolicy curves (test period)")

    show = [
        "policy",
        "alpha",
        "fill_rate",
        "stockout_rate",
        "avg_inventory_value",
        "holding_cost",
        "lost_margin",
        "total_cost",
    ]

    print(
        curves[show]
        .round(3)
        .to_string(index=False)
    )

    print(
        f"\nSensitivity at alpha = {SENS_ALPHA}"
    )

    print(
        sens[
            [
                "scenario",
                "policy",
                "fill_rate",
                "avg_inventory_value",
                "total_cost",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nInventory needed to hit the same fill rate "
        "(interpolated)"
    )

    print(
        headline
        .round(2)
        .to_string(index=False)
    )

    # -------------------------------------------------------------
    # 9. Save run information
    # -------------------------------------------------------------

    run_info = {
        "data": args.data,
        "n_skus": len(skus),
        "n_days": n_days,
        "train_end": t1,
        "test_start": t2,
        "n_test_days": n_days - t2,
    }

    (
        out / "run_info.json"
    ).write_text(
        json.dumps(
            run_info,
            indent=2,
        )
    )

    # -------------------------------------------------------------
    # 10. Generate charts
    # -------------------------------------------------------------

    if not args.no_charts:

        from src.report import make_figures

        make_figures(
            out,
            accuracy,
            curves,
            sens,
            headline,
        )


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        choices=["synthetic", "m5"],
        default="synthetic",
    )

    ap.add_argument(
        "--m5-dir",
        default=None,
    )

    ap.add_argument(
        "--skus",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--days",
        type=int,
        default=900,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    ap.add_argument(
        "--out",
        default="results",
    )

    ap.add_argument(
        "--no-charts",
        action="store_true",
    )

    main(ap.parse_args())