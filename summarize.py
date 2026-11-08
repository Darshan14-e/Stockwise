"""Print the results of a run as Markdown tables, ready to paste into the README.

    python summarize.py results          # synthetic run
    python summarize.py results_m5       # real M5 run

Every number comes straight from the CSV files, so nothing has to be typed by hand.
"""
import json
import sys
from pathlib import Path

import pandas as pd

try:  # Windows consoles can choke on the rupee sign
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES = {"baseline": "Baseline", "ml_error": "ML + error buffer", "ml_quantile": "ML quantile"}


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def main(folder):
    d = Path(folder)
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    label, scale = info["unit_label"], info["unit_scale"]
    curves = pd.read_csv(d / "policy_curves.csv")
    acc = pd.read_csv(d / "forecast_accuracy.csv")
    sens = pd.read_csv(d / "sensitivity.csv")
    head = pd.read_csv(d / "headline.csv")
    diag = json.loads((d / "diagnostics.json").read_text(encoding="utf-8"))
    money = lambda v: f"{v / scale:,.2f}"

    print(f"Data: {info['data']}, {info['n_skus']} products, {info['n_days']} days; "
          f"tested on the last {info['n_test_days']} days. Money in {label}.\n")

    print("### Stock needed for the same fill rate\n")
    if head.empty:
        print("The three policies never reach a common fill rate, so there is no like-for-like comparison.\n")
    else:
        rows = []
        for _, r in head.iterrows():
            rows.append({
                "Fill rate": f"{100 * r.fill_rate_target:.1f}%",
                f"Baseline stock ({label})": money(r.baseline_inventory_value),
                "ML + error buffer": f"{money(r.ml_error_inventory_value)} ({-r.ml_error_saving_vs_baseline_pct:+.0f}%)",
                "ML quantile": f"{money(r.ml_quantile_inventory_value)} ({-r.ml_quantile_saving_vs_baseline_pct:+.0f}%)",
            })
        print(md(pd.DataFrame(rows)))
        print("\nPercentages compare with the baseline: negative means less stock, positive means more.\n")

    print("### All three policies set to a 95% target\n")
    at = curves[curves.alpha == 0.95].set_index("policy")
    rows = [{"Policy": NAMES[p], "Fill rate": f"{100 * at.loc[p, 'fill_rate']:.1f}%",
             f"Avg stock ({label})": money(at.loc[p, "avg_inventory_value"]),
             f"Holding cost ({label})": money(at.loc[p, "holding_cost"]),
             f"Lost margin ({label})": money(at.loc[p, "lost_margin"]),
             f"Total cost ({label})": money(at.loc[p, "total_cost"])} for p in NAMES]
    print(md(pd.DataFrame(rows)))
    print("\nThe policies end up at different fill rates, so compare costs together with the fill rate column.\n")

    print("### Forecast quality on unseen days\n")
    w = acc[acc.metric.str.startswith("WAPE")].iloc[0]
    p95 = acc[(acc.metric == "pinball loss") & (acc.alpha.astype(str) == "0.95")].iloc[0]
    c95 = acc[(acc.metric.str.startswith("coverage")) & (acc.alpha.astype(str) == "0.95")].iloc[0]
    c99 = acc[(acc.metric.str.startswith("coverage")) & (acc.alpha.astype(str) == "0.99")].iloc[0]
    rows = [
        {"": "Error of the average forecast (WAPE, lower is better)", "Baseline": f"{100 * w.baseline:.1f}%",
         "ML + error buffer": f"{100 * w.ml_error:.1f}%", "ML quantile": "n/a"},
        {"": "Pinball loss at the 95% quantile (lower is better)", "Baseline": f"{p95.baseline:.3g}",
         "ML + error buffer": f"{p95.ml_error:.3g}", "ML quantile": f"{p95.ml_quantile:.3g}"},
        {"": "Share of demand covered when aiming for 95%", "Baseline": f"{100 * c95.baseline:.1f}%",
         "ML + error buffer": f"{100 * c95.ml_error:.1f}%", "ML quantile": f"{100 * c95.ml_quantile:.1f}%"},
        {"": "Share of demand covered when aiming for 99%", "Baseline": f"{100 * c99.baseline:.1f}%",
         "ML + error buffer": f"{100 * c99.ml_error:.1f}%", "ML quantile": f"{100 * c99.ml_quantile:.1f}%"},
    ]
    print(md(pd.DataFrame(rows)))

    print("\n### Stress tests (95% target)\n")
    rows = []
    for s in dict.fromkeys(sens.scenario):
        row = {"Scenario": s}
        for p in NAMES:
            r = sens[(sens.scenario == s) & (sens.policy == p)].iloc[0]
            row[NAMES[p]] = f"{money(r.total_cost)} (fill {100 * r.fill_rate:.1f}%)"
        rows.append(row)
    print(md(pd.DataFrame(rows)))
    print(f"\nTotal cost in {label}.\n")

    print("### Data health\n")
    print(f"- Training target ratio: median {diag['train_target_ratio_median']:.2f}, 99th percentile "
          f"{diag['train_target_ratio_p99']:.1f}, maximum {diag['train_target_ratio_max']:.0f}; "
          + (f"{diag['train_rows_ratio_above_cap_pct']:.2f}% of rows are above the cap of {diag['target_cap_used']:g}."
             if diag["target_cap_used"] else "the cap is switched off."))
    print(f"- Training rows with no sales in the previous 28 days: {diag['train_rows_with_zero_28d_sales_pct']:.2f}%.")
    print(f"- Forecast bias on the test days: ML mean {diag['test_ml_mean_bias_pct']:+.1f}%, "
          f"moving average {diag['test_baseline_bias_pct']:+.1f}%.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python summarize.py RESULTS_FOLDER")
    main(sys.argv[1])
