"""Lead-time demand forecasting.

The inventory decision needs one number per SKU per day: how much will we sell
over the supplier lead time (the days between ordering and the stock arriving)?
So instead of forecasting daily demand and summing, we forecast that
lead-time total directly, as a mean and as several upper quantiles.

Everything the model sees at decision day t uses only data up to day t, plus
the promotion calendar, which retailers plan in advance.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm


FEATURES = [
    "ratio7",
    "ratio14",
    "ratio56",
    "cv28",
    "log_level",
    "promo_frac",
    "promo_last7",
    "weekend_frac",
    "dow",
    "month",
    "sin_doy",
    "cos_doy",
    "H",
]

ALPHAS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99)

# Days of historical demand required before making the first decision.
MIN_HISTORY = 56


def make_features(D, P, lead_times, t_idx):
    """Build one feature row per (SKU, decision day t).

    Parameters
    ----------
    D : DataFrame
        Daily demand, indexed by date with one column per SKU.

    P : DataFrame
        Promotion calendar with the same shape as D.

    lead_times : Series
        Lead time for each SKU.

    t_idx : array-like
        Decision-day indices.

    Returns
    -------
    DataFrame
        Features, scaling base, SKU, day index and lead-time demand target.
    """

    n_days = len(D)
    max_h = int(lead_times.max())
    dates = D.index

    # Updated Pandas properties.
    # These replace the deprecated .dayofweek and .dayofyear.
    dow_all = dates.day_of_week.to_numpy()
    doy_all = dates.day_of_year.to_numpy()
    month_all = dates.month.to_numpy()

    def roll(window, function):
        return getattr(
            D.rolling(window, min_periods=window),
            function
        )().to_numpy()

    m7 = roll(7, "mean")
    m14 = roll(14, "mean")
    m28 = roll(28, "mean")
    m56 = roll(56, "mean")
    s28 = roll(28, "std")

    # Promotions are assumed to be known in advance.
    # Future days are padded with "no promotion".
    P_pad = np.vstack([
        P.to_numpy().astype(float),
        np.zeros((max_h + 1, P.shape[1]))
    ])

    cum_p = np.vstack([
        np.zeros((1, P_pad.shape[1])),
        np.cumsum(P_pad, axis=0)
    ])

    D_np = D.to_numpy().astype(float)

    cum_d = np.vstack([
        np.zeros((1, D_np.shape[1])),
        np.cumsum(D_np, axis=0)
    ])

    t_idx = np.asarray(t_idx)
    frames = []

    for j, sku in enumerate(D.columns):

        h = int(lead_times[sku])
        t = t_idx

        # Prevent division by very small demand averages.
        mm28 = np.maximum(m28[t, j], 0.1)

        # Planned promotions during the future lead-time window.
        promo_win = (
            cum_p[t + h + 1, j]
            - cum_p[t + 1, j]
        )

        # Promotion activity in the recent seven days.
        promo_last7 = (
            cum_p[t + 1, j]
            - cum_p[t - 6, j]
        ) / 7

        # Fraction of lead-time days that are weekends.
        weekend = np.sum(
            [
                ((dow_all[t] + k) % 7) >= 5
                for k in range(1, h + 1)
            ],
            axis=0
        )

        # Lead-time demand target:
        # demand from t+1 through t+h.
        end = np.minimum(t + h + 1, n_days)

        y = (
            cum_d[end, j]
            - cum_d[t + 1, j]
        )

        # Targets cannot be calculated if the future lead-time
        # demand is outside the available dataset.
        y = np.where(
            t + h <= n_days - 1,
            y,
            np.nan
        )

        frames.append(
            pd.DataFrame(
                {
                    "sku": sku,
                    "t": t,
                    "H": h,

                    "ratio7": m7[t, j] / mm28,
                    "ratio14": m14[t, j] / mm28,
                    "ratio56": m56[t, j] / mm28,

                    "cv28": s28[t, j] / mm28,

                    "log_level": np.log1p(m28[t, j]),

                    "promo_frac": promo_win / h,
                    "promo_last7": promo_last7,

                    "weekend_frac": weekend / h,

                    "dow": dow_all[t],
                    "month": month_all[t],

                    "sin_doy": np.sin(
                        2 * np.pi * doy_all[t] / 365.25
                    ),

                    "cos_doy": np.cos(
                        2 * np.pi * doy_all[t] / 365.25
                    ),

                    # Scaling factor for normalized ML target.
                    "base": mm28 * h,

                    "mean28": m28[t, j],
                    "std28": s28[t, j],

                    "y": y,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


class LeadTimeForecaster:
    """LightGBM models for lead-time demand.

    One model predicts mean demand and separate models predict
    upper demand quantiles.
    """

    def __init__(self, alphas=ALPHAS, seed=0):

        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )

        self.alphas = tuple(alphas)

        self.mean_model = LGBMRegressor(
            objective="regression",
            **params
        )

        self.q_models = {
            alpha: LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                **params
            )
            for alpha in self.alphas
        }

    def fit(self, df):
        """Train mean and quantile models."""

        X = df[FEATURES]

        # Normalize demand by each SKU's recent demand level.
        ratio = df["y"] / df["base"]

        self.mean_model.fit(X, ratio)

        for model in self.q_models.values():
            model.fit(X, ratio)

        return self

    def predict(self, df):
        """Predict mean and quantile lead-time demand."""

        X = df[FEATURES]
        base = df["base"].to_numpy()

        out = {
            "mean": (
                np.clip(
                    self.mean_model.predict(X),
                    0,
                    None
                )
                * base
            )
        }

        for alpha, model in self.q_models.items():

            out[alpha] = (
                np.clip(
                    model.predict(X),
                    0,
                    None
                )
                * base
            )

        return out


def baseline_quantile(df, alpha):
    """Moving average forecast with normal-distribution safety stock."""

    return (
        df["base"].to_numpy()
        + norm.ppf(alpha)
        * df["std28"].to_numpy()
        * np.sqrt(df["H"].to_numpy())
    )


def error_based_quantile(df, mean_pred, sigma_by_sku, alpha):
    """ML mean forecast plus safety stock from validation errors."""

    return (
        mean_pred
        + norm.ppf(alpha)
        * df["sku"].map(sigma_by_sku).to_numpy()
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def wape(y, f):
    """Weighted Absolute Percentage Error."""

    return float(
        np.abs(y - f).sum()
        / np.abs(y).sum()
    )


def pinball(y, q, alpha):
    """Pinball loss for quantile forecasts."""

    d = y - q

    return float(
        np.mean(
            np.maximum(
                alpha * d,
                (alpha - 1) * d
            )
        )
    )


def coverage(y, q):
    """Fraction of actual demand covered by the forecast."""

    return float(
        np.mean(y <= q)
    )