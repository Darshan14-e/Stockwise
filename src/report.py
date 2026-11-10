
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
STYLE = {  # fixed colour per policy, in the same order everywhere
    "baseline":    dict(color="#2a78d6", marker="o", label="Baseline: moving average + normal safety stock"),
    "ml_error":    dict(color="#eb6834", marker="s", label="ML mean forecast + safety stock from its errors"),
    "ml_quantile": dict(color="#1baf7a", marker="^", label="ML quantile forecast (direct)"),
}
SHORT = {"baseline": "Baseline", "ml_error": "ML + error buffer", "ml_quantile": "ML quantile"}


def _base_ax(figsize=(8.2, 5.0)):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _title(fig, ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=30)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _legend(ax):
    """Legend below the plot, so it can never sit on top of the data."""
    return ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.8,
                     loc="upper left", bbox_to_anchor=(0, -0.17), ncol=1)


def make_figures(out, accuracy, curves, sens, headline, unit=("₹ lakh", 1e5)):
    out = Path(out)
    _tradeoff(out, curves, headline, unit)
    _calibration(out, accuracy)
    _sensitivity(out, sens, unit)
    print(f"charts written to {out}/")


def _tradeoff(out, curves, headline, unit):
    label, scale = unit
    fig, ax = _base_ax()
    shown = curves[curves["fill_rate"] >= 0.90]          # the region that matters; low service is not interesting
    ends = {}
    for p, st in STYLE.items():
        c = shown[shown.policy == p].sort_values("fill_rate")
        if c.empty:
            continue
        x, y = c["fill_rate"] * 100, c["avg_inventory_value"] / scale
        ax.plot(x, y, color=st["color"], marker=st["marker"], markersize=5.5, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=st["label"], zorder=3)
        ends[p] = (float(x.iloc[-1]), float(y.iloc[-1]))

    ax.set_ylim(bottom=0)
    x_all = shown["fill_rate"] * 100
    ax.set_xlim(x_all.min() - 0.4, x_all.max() + 0.22 * (x_all.max() - x_all.min()))

    # direct labels at the right-hand end of each line, nudged apart so they never overlap
    ymax = ax.get_ylim()[1]
    last_y = None
    for p, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        ty = y if last_y is None else max(y, last_y + 0.07 * ymax)
        ax.text(x + 0.12, ty, SHORT[p], ha="left", va="center", fontsize=9, color=INK, fontweight="bold")
        last_y = ty

    # like-for-like callout, only when the three policies reach a common fill rate
    if headline is not None and len(headline) > 0:
        mid = headline.iloc[int((headline["fill_rate_target"] - 0.95).abs().to_numpy().argmin())]
        xt = mid["fill_rate_target"] * 100
        y_b = mid["baseline_inventory_value"] / scale
        y_q = mid["ml_quantile_inventory_value"] / scale
        saving = mid["ml_quantile_saving_vs_baseline_pct"]
        if np.isfinite([y_b, y_q, saving]).all():
            ax.axvline(xt, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate("", xy=(xt, y_q), xytext=(xt, y_b),
                        arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.2), zorder=4)
            word = "less" if saving > 0 else "more"
            ax.text(0.985, 0.05,
                    f"At {xt:.1f}% fill rate (arrow), the ML quantile policy\nneeds ~{abs(saving):.0f}% {word} stock than the baseline",
                    transform=ax.transAxes, fontsize=8.8, color=INK, va="bottom", ha="right")

    ax.set_xlabel("Fill rate: share of demanded units actually sold (%)", fontsize=9.5, color=INK2, labelpad=8)
    ax.set_ylabel(f"Average stock held ({label} at cost)", fontsize=9.5, color=INK2, labelpad=8)
    _title(fig, ax, "Stock needed at each service level",
           "Each point is one service-level setting. Lower is better: less stock for the same fill rate. Unseen test days.")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_tradeoff.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _calibration(out, accuracy):
    cov = accuracy[accuracy.metric.str.startswith("coverage")].copy()
    cov["alpha"] = cov["alpha"].astype(float)
    fig, ax = _base_ax((7.4, 4.8))
    a = cov["alpha"].to_numpy()
    x = np.arange(len(a))
    ax.plot(x, a, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x[-1] + 0.12, a[-1], "Target", fontsize=8.8, color=INK2, va="center")
    for p, st in STYLE.items():
        ax.plot(x, cov[p].to_numpy(), color=st["color"], marker=st["marker"], markersize=6.5, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=st["label"], zorder=3)
    ax.text(x[-1] + 0.12, cov["baseline"].iloc[-1], "Baseline", fontsize=8.8, color=INK, va="center", fontweight="bold")
    ax.set_xlim(-0.3, x[-1] + 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in a])
    ax.set_xlabel("Target service level (what the policy aims for)", fontsize=9.5, color=INK2, labelpad=8)
    ax.set_ylabel("Share of lead-time demand actually covered", fontsize=9.5, color=INK2, labelpad=8)
    _title(fig, ax, "How often the stock target was enough",
           "Closer to the dashed target is better. Below it means the policy under-protects. Unseen test days.")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _sensitivity(out, sens, unit):
    label, scale = unit
    scen = list(dict.fromkeys(sens["scenario"]))
    fig, ax = _base_ax((8.2, 4.8))
    w, gap = 0.26, 0.0
    for i, (p, st) in enumerate(STYLE.items()):
        vals = [sens[(sens.scenario == s) & (sens.policy == p)]["total_cost"].iloc[0] / scale for s in scen]
        pos = np.arange(len(scen)) + (i - 1) * (w + gap)
        ax.bar(pos, vals, width=w, color=st["color"], edgecolor=SURFACE, linewidth=1.6,
               label=st["label"], zorder=3)
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels([s.replace(" after ", "\nafter ").replace(" than ", " than\n") for s in scen], fontsize=9, color=INK2)
    ax.set_ylabel(f"Total cost over the test period ({label})", fontsize=9.5, color=INK2, labelpad=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    _title(fig, ax, "What if the plan is wrong?",
           "Cost = holding + margin lost on missed sales, all policies set to a 95% target. Fill rates differ; see the table.")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_sensitivity.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
