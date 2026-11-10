import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm

FEATURES = [
    "ratio7", "ratio14", "ratio56", "cv28", "log_level",
    "promo_frac", "promo_last7", "weekend_frac",
    "dow", "month", "sin_doy", "cos_doy", "H",
]
Q_ALPHAS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99)   # quantile models we train
Z_ALPHAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)  # normal-theory policies
REPORT_ALPHAS = (0.80, 0.90, 0.95, 0.98, 0.99)                      # levels shown in the accuracy table
MIN_HISTORY = 56  # days of history needed before the first decision day

# Training targets are "lead-time demand / (recent average x lead time)". After a stretch of zero
# sales that denominator is tiny, so the ratio can reach the hundreds and a handful of such rows can
# drag the whole model (we saw ML policies hold 2-4x the stock). Capping the training target at 4x
# the recent run-rate removes that; results were the same for any cap between 3 and 6.
TARGET_CLIP = 4.0


def make_features(D, P, lead_times, t_idx):
    """Build one feature row per (SKU, decision day t) for t in t_idx.

    D, P : DataFrames (days x SKUs). lead_times : Series indexed by SKU.
    Returns a DataFrame with the features, the scaling `base`, the SKU, the day
    index `t` and the target `y` (units sold in days t+1..t+H, NaN if those
    days are not in the data yet).
    """
    n_days = len(D)
    max_h = int(lead_times.max())
    dates = D.index
    dow_all = dates.dayofweek.to_numpy()
    doy_all = dates.dayofyear.to_numpy()
    month_all = dates.month.to_numpy()

    roll = lambda w, f: getattr(D.rolling(w, min_periods=w), f)().to_numpy()
    m7, m14, m28, m56 = roll(7, "mean"), roll(14, "mean"), roll(28, "mean"), roll(56, "mean")
    s28 = roll(28, "std")

    # promotions are a known calendar; pad the future with "no promo"
    P_pad = np.vstack([P.to_numpy().astype(float), np.zeros((max_h + 1, P.shape[1]))])
    cum_p = np.vstack([np.zeros((1, P_pad.shape[1])), np.cumsum(P_pad, axis=0)])
    D_np = D.to_numpy().astype(float)
    cum_d = np.vstack([np.zeros((1, D_np.shape[1])), np.cumsum(D_np, axis=0)])

    t_idx = np.asarray(t_idx)
    frames = []
    for j, sku in enumerate(D.columns):
        h = int(lead_times[sku])
        t = t_idx
        mm28 = np.maximum(m28[t, j], 0.1)
        promo_win = cum_p[t + h + 1, j] - cum_p[t + 1, j]           # days t+1..t+h
        promo_last7 = (cum_p[t + 1, j] - cum_p[t - 6, j]) / 7       # days t-6..t
        weekend = np.sum([((dow_all[t] + k) % 7) >= 5 for k in range(1, h + 1)], axis=0)
        end = np.minimum(t + h + 1, n_days)
        y = cum_d[end, j] - cum_d[t + 1, j]
        y = np.where(t + h <= n_days - 1, y, np.nan)

        frames.append(
            pd.DataFrame(
                {
                    "sku": sku, "t": t, "H": h,
                    "ratio7": m7[t, j] / mm28, "ratio14": m14[t, j] / mm28,
                    "ratio56": m56[t, j] / mm28, "cv28": s28[t, j] / mm28,
                    "log_level": np.log1p(m28[t, j]),
                    "promo_frac": promo_win / h, "promo_last7": promo_last7,
                    "weekend_frac": weekend / h,
                    "dow": dow_all[t], "month": month_all[t],
                    "sin_doy": np.sin(2 * np.pi * doy_all[t] / 365.25),
                    "cos_doy": np.cos(2 * np.pi * doy_all[t] / 365.25),
                    "base": mm28 * h,            # "flat" forecast, also our scaling factor
                    "mean28": m28[t, j], "std28": s28[t, j],
                    "y": y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class LeadTimeForecaster:
    def __init__(self, alphas=Q_ALPHAS, seed=0, target_cap=TARGET_CLIP):
        self.target_cap = target_cap
        params = dict(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, verbose=-1,
        )
        self.alphas = tuple(alphas)
        self.mean_model = LGBMRegressor(objective="regression", **params)
        self.q_models = {a: LGBMRegressor(objective="quantile", alpha=a, **params) for a in self.alphas}

    def fit(self, df):
        X, ratio = df[FEATURES], training_ratio(df, self.target_cap)
        self.mean_model.fit(X, ratio)
        for m in self.q_models.values():
            m.fit(X, ratio)
        return self

    def predict(self, df):
        X, base = df[FEATURES], df["base"].to_numpy()
        out = {"mean": np.clip(self.mean_model.predict(X), 0, None) * base}
        # Each quantile is its own model, so a higher quantile can occasionally come out below a lower
        # one ("quantile crossing"). Sorting along the quantile axis fixes that.
        q = np.column_stack([np.clip(self.q_models[a].predict(X), 0, None) for a in self.alphas])
        q = np.maximum.accumulate(q, axis=1)
        for i, a in enumerate(self.alphas):
            out[a] = q[:, i] * base
        return out


def training_ratio(df, cap=TARGET_CLIP):
    ratio = df["y"] / df["base"]
    return ratio.clip(upper=cap) if cap else ratio


def baseline_quantile(df, alpha):
    return df["base"].to_numpy() + norm.ppf(alpha) * df["std28"].to_numpy() * np.sqrt(df["H"].to_numpy())


def error_based_quantile(df, mean_pred, sigma_by_sku, alpha):
    """ML mean forecast + safety stock from that forecast's own validation errors."""
    return mean_pred + norm.ppf(alpha) * df["sku"].map(sigma_by_sku).to_numpy()


# ---------------------------------------------------------------- metrics --
def wape(y, f):
    return float(np.abs(y - f).sum() / np.abs(y).sum())


def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y, q):
    return float(np.mean(y <= q))
