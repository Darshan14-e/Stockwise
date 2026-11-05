
"""Charts for the README.

Creates:
1. Trade-off chart: service level vs inventory
2. Calibration chart: actual coverage vs target coverage
3. Sensitivity chart: total cost under different scenarios
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so charts can be generated from the terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# COLOURS AND STYLES
# ---------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

GRID = "#e1e0d9"
AXIS = "#c3c2b7"


# Fixed style for each inventory policy.
STYLE = {
    "baseline": {
        "color": "#2a78d6",
        "marker": "o",
        "label": "Baseline: moving average + normal safety stock",
    },
    "ml_error": {
        "color": "#eb6834",
        "marker": "s",
        "label": "ML mean forecast + safety stock from its errors",
    },
    "ml_quantile": {
        "color": "#1baf7a",
        "marker": "^",
        "label": "ML quantile forecast (direct)",
    },
}


SHORT = {
    "baseline": "Baseline",
    "ml_error": "ML + error buffer",
    "ml_quantile": "ML quantile",
}


# ---------------------------------------------------------------------
# COMMON CHART FUNCTIONS
# ---------------------------------------------------------------------

def _base_ax(figsize=(8.2, 5.0)):
    """Create a consistently styled figure and axis."""

    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Hide unnecessary borders.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Style remaining borders.
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    ax.tick_params(
        colors=MUTED,
        labelsize=9,
        length=0,
    )

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    return fig, ax


def _title(fig, ax, title, subtitle):
    """Add a chart title and subtitle."""

    ax.set_title(
        title,
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        pad=28,
    )

    ax.text(
        0,
        1.035,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color=INK2,
        va="bottom",
    )


def _legend(ax, **kwargs):
    """Create a consistent legend."""

    return ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
        handlelength=1.8,
        **kwargs,
    )


# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------

def make_figures(out, accuracy, curves, sens, headline):
    """Generate all charts."""

    out = Path(out)

    # Create the output directory if it does not exist.
    out.mkdir(parents=True, exist_ok=True)

    _tradeoff(out, curves, headline)
    _calibration(out, accuracy)
    _sensitivity(out, sens)

    print(f"Charts written to {out}/")


# ---------------------------------------------------------------------
# CHART 1: SERVICE LEVEL VS INVENTORY
# ---------------------------------------------------------------------

def _tradeoff(out, curves, headline):
    """Plot fill rate against average inventory."""

    fig, ax = _base_ax((8.5, 5.8))

    # -------------------------------------------------------------
    # Plot each policy.
    # -------------------------------------------------------------

    for policy, style in STYLE.items():

        data = curves[
            curves["policy"] == policy
        ].sort_values("alpha")

        if data.empty:
            continue

        # Convert fill rate to percentage.
        x = data["fill_rate"] * 100

        # Inventory values are shown in thousands of rupees.
        # Your data ranges from approximately 2,000 to 19,000.
        y = data["avg_inventory_value"] / 1000

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markersize=6.5,
            linewidth=2.2,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=style["label"],
            zorder=3,
        )

    # -------------------------------------------------------------
    # Direct labels for the policies.
    # -------------------------------------------------------------

    offsets = {
        "baseline": (-0.05, 0.20),
        "ml_error": (-0.05, 0.50),
        "ml_quantile": (-0.05, -0.25),
    }

    for policy in STYLE:

        data = curves[
            curves["policy"] == policy
        ].sort_values("alpha")

        if data.empty:
            continue

        # Use the first point.
        point = data.iloc[0]

        x = point["fill_rate"] * 100
        y = point["avg_inventory_value"] / 1000

        dx, dy = offsets.get(policy, (0, 0))

        ax.text(
            x + dx,
            y + dy,
            SHORT[policy],
            ha="right",
            va="center",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # SAME-SERVICE COMPARISON
    #
    # Only draw the arrow if the headline table contains valid
    # baseline and ML quantile values.
    # -------------------------------------------------------------

    required_columns = [
        "fill_rate_target",
        "baseline_inventory_value",
        "ml_quantile_inventory_value",
    ]

    can_compare = all(
        column in headline.columns
        for column in required_columns
    )

    if can_compare:

        valid_headline = headline.dropna(
            subset=[
                "baseline_inventory_value",
                "ml_quantile_inventory_value",
            ]
        )

        if not valid_headline.empty:

            # Select the middle valid comparison row.
            mid = valid_headline.iloc[
                len(valid_headline) // 2
            ]

            xt = mid["fill_rate_target"] * 100

            y_baseline = (
                mid["baseline_inventory_value"] / 1000
            )

            y_quantile = (
                mid["ml_quantile_inventory_value"] / 1000
            )

            # Draw a vertical reference line.
            ax.axvline(
                xt,
                color=MUTED,
                linewidth=1,
                linestyle=(0, (4, 3)),
                zorder=1,
            )

            # Draw the comparison arrow.
            ax.annotate(
                "",
                xy=(xt, y_quantile),
                xytext=(xt, y_baseline),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": INK2,
                    "linewidth": 1.2,
                },
                zorder=4,
            )

            # Display saving percentage if available.
            if (
                "ml_quantile_saving_vs_baseline_pct"
                in mid.index
            ):

                saving = mid[
                    "ml_quantile_saving_vs_baseline_pct"
                ]

                if not np.isnan(saving):

                    middle_y = (
                        y_baseline + y_quantile
                    ) / 2

                    ax.text(
                        xt + 0.15,
                        middle_y,
                        f"{saving:.0f}% less stock",
                        fontsize=8.5,
                        color=INK,
                        va="center",
                    )

    # -------------------------------------------------------------
    # AXIS SETTINGS
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Fill rate: share of demanded units actually sold (%)",
        fontsize=9.5,
        color=INK2,
        labelpad=8,
    )

    ax.set_ylabel(
        "Average stock held (₹ thousand at cost)",
        fontsize=9.5,
        color=INK2,
        labelpad=8,
    )

    # Set sensible x-axis limits.
    min_fill = curves["fill_rate"].min() * 100
    max_fill = curves["fill_rate"].max() * 100

    ax.set_xlim(
        min_fill - 1.0,
        max_fill + 0.7,
    )

    # Add some space above the highest point.
    max_inventory = (
        curves["avg_inventory_value"].max() / 1000
    )

    ax.set_ylim(
        bottom=0,
        top=max_inventory * 1.15,
    )

    _title(
        fig,
        ax,
        "Service level vs inventory",
        (
            "Each point is a service-level setting. "
            "Lower inventory for the same fill rate is better."
        ),
    )

    _legend(
        ax,
        loc="upper left",
    )

    # Use manual spacing rather than tight_layout.
    fig.subplots_adjust(
        top=0.82,
        bottom=0.16,
        left=0.13,
        right=0.96,
    )

    fig.savefig(
        out / "fig_tradeoff.png",
        facecolor=SURFACE,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# CHART 2: CALIBRATION
# ---------------------------------------------------------------------

def _calibration(out, accuracy):
    """Plot actual coverage against the target service level."""

    # Select coverage rows only.
    cov = accuracy[
        accuracy["metric"].str.startswith(
            "coverage"
        )
    ].copy()

    cov["alpha"] = cov["alpha"].astype(float)

    fig, ax = _base_ax((7.8, 5.2))

    target = cov["alpha"].to_numpy()

    x = np.arange(len(target))

    # -------------------------------------------------------------
    # Plot the target line.
    # -------------------------------------------------------------

    ax.plot(
        x,
        target,
        color=MUTED,
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    ax.text(
        x[-1] + 0.12,
        target[-1],
        "Target",
        fontsize=8.8,
        color=INK2,
        va="center",
    )

    # -------------------------------------------------------------
    # Plot each policy.
    # -------------------------------------------------------------

    for policy, style in STYLE.items():

        if policy not in cov.columns:
            continue

        values = cov[policy].to_numpy()

        ax.plot(
            x,
            values,
            color=style["color"],
            marker=style["marker"],
            markersize=6.5,
            linewidth=2.2,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=style["label"],
            zorder=3,
        )

    # Add a label for the baseline line.
    if "baseline" in cov.columns:

        baseline_value = (
            cov["baseline"].iloc[-1]
        )

        ax.text(
            x[-1] + 0.12,
            baseline_value,
            "Baseline",
            fontsize=8.8,
            color=INK,
            va="center",
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # AXIS SETTINGS
    # -------------------------------------------------------------

    ax.set_xlim(
        -0.3,
        x[-1] + 1.0,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{value:.2f}"
            for value in target
        ]
    )

    # Add a little padding to the y-axis.
    all_values = []

    for policy in STYLE:
        if policy in cov.columns:
            all_values.extend(
                cov[policy].dropna().tolist()
            )

    all_values.extend(target.tolist())

    if all_values:

        lower = max(
            0,
            min(all_values) - 0.015,
        )

        upper = min(
            1.02,
            max(all_values) + 0.015,
        )

        ax.set_ylim(
            lower,
            upper,
        )

    ax.set_xlabel(
        "Target service level (what the policy aims for)",
        fontsize=9.5,
        color=INK2,
        labelpad=8,
    )

    ax.set_ylabel(
        "Share of lead-time demand actually covered",
        fontsize=9.5,
        color=INK2,
        labelpad=8,
    )

    _title(
        fig,
        ax,
        "How often the stock target was enough",
        (
            "Closer to the dashed target is better. "
            "Below it means the policy under-protects. "
            "Unseen test days."
        ),
    )

    _legend(
        ax,
        loc="lower right",
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.16,
        left=0.13,
        right=0.94,
    )

    fig.savefig(
        out / "fig_calibration.png",
        facecolor=SURFACE,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# CHART 3: SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------

def _sensitivity(out, sens):
    """Compare total cost across different scenarios."""

    # Preserve the scenario order from the data.
    scenarios = list(
        dict.fromkeys(
            sens["scenario"]
        )
    )

    fig, ax = _base_ax((9.0, 5.4))

    # Width of each bar.
    width = 0.25

    # Center positions for scenarios.
    base_positions = np.arange(
        len(scenarios)
    )

    # -------------------------------------------------------------
    # Draw bars for each policy.
    # -------------------------------------------------------------

    policy_offsets = {
        "baseline": -width,
        "ml_error": 0,
        "ml_quantile": width,
    }

    for policy, style in STYLE.items():

        values = []

        for scenario in scenarios:

            row = sens[
                (sens["scenario"] == scenario)
                & (sens["policy"] == policy)
            ]

            if row.empty:
                values.append(np.nan)
            else:
                # Display cost in thousands of rupees.
                values.append(
                    row["total_cost"].iloc[0] / 1000
                )

        positions = (
            base_positions
            + policy_offsets[policy]
        )

        ax.bar(
            positions,
            values,
            width=width,
            color=style["color"],
            edgecolor=SURFACE,
            linewidth=1.6,
            label=style["label"],
            zorder=3,
        )

    # -------------------------------------------------------------
    # Format scenario labels.
    # -------------------------------------------------------------

    labels = []

    for scenario in scenarios:

        label = (
            scenario
            .replace(
                " after ",
                "\nafter ",
            )
            .replace(
                " than ",
                " than\n",
            )
        )

        labels.append(label)

    ax.set_xticks(
        base_positions
    )

    ax.set_xticklabels(
        labels,
        fontsize=9,
        color=INK2,
    )

    ax.set_ylabel(
        "Total cost over the test period (₹ thousand)",
        fontsize=9.5,
        color=INK2,
        labelpad=8,
    )

    # Give the bars some space at the top.
    current_max = (
        sens["total_cost"].max() / 1000
    )

    ax.set_ylim(
        0,
        current_max * 1.15,
    )

    _title(
        fig,
        ax,
        "What if the plan is wrong?",
        (
            "Total cost = holding cost + margin lost on missed sales, "
            "at a 95% service target. Lower is better."
        ),
    )

    _legend(
        ax,
        loc="upper left",
        ncol=1,
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.20,
        left=0.12,
        right=0.96,
    )

    fig.savefig(
        out / "fig_sensitivity.png",
        facecolor=SURFACE,
        bbox_inches="tight",
    )

    plt.close(fig)