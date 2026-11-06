"""Dividend features: trailing-12-month yield, yield percentile vs own history, dividend growth,
special-dividend exclusion, and cut / suspension detection. All features are trailing-only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsx_div_ta.config import SignalConfig

TRADING_DAYS_PER_YEAR = 252
PRIOR_PAYMENTS_FOR_MEDIAN = 8
MIN_SUSPENSION_DAYS = 120
ALLOWED_FREQUENCIES: tuple[float, ...] = (1.0, 2.0, 4.0, 12.0)

DIVIDEND_COLUMNS: tuple[str, ...] = (
    "ttm_div",
    "ttm_yield",
    "yield_pct",
    "div_growth",
    "last_div",
    "prev_div",
    "div_cut",
    "div_suspended",
    "div_cut_recent",
    "special_div",
    "fwd_div",
    "fwd_yield",
)


def split_payments(dividends: pd.Series, special_mult: float) -> tuple[pd.Series, pd.Series]:
    """Split payment events into (regular, special).

    A payment is *special* when it exceeds ``special_mult`` x the median of the previous
    (up to 8) payments. The first payment is always regular."""
    pay = dividends[dividends > 0].astype("float64")
    prior_med = pay.shift(1).rolling(PRIOR_PAYMENTS_FOR_MEDIAN, min_periods=1).median()
    special = pay > special_mult * prior_med
    return pay[~special], pay[special]


def infer_frequency(gap_days: float) -> float:
    """Payments per year (1, 2, 4 or 12) nearest to 365.25 / median gap; NaN if unknown."""
    if not np.isfinite(gap_days) or gap_days <= 0:
        return float("nan")
    raw = 365.25 / gap_days
    return float(min(ALLOWED_FREQUENCIES, key=lambda f: (abs(f - raw), f)))


def payment_frequency(regular: pd.Series) -> np.ndarray:
    """Payments per year in effect at each regular payment (median of the last 4 gaps)."""
    gaps = pd.Series(regular.index, index=regular.index).diff().dt.days
    return gaps.rolling(4, min_periods=1).median().map(infer_frequency).to_numpy(dtype=float)


def ttm_dividends(regular: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Trailing-twelve-month regular dividends, aligned to ``idx``.

    Defined as the sum of the most recent N regular payments, where N (payments per year) is
    inferred from the median of the last 4 gaps between payments. A fixed 365-day window is
    *not* used because normal pay-date jitter (weekends, holidays) makes it catch 3 or 5
    quarterly payments and creates fake yield spikes/drops.

    * no payment in the trailing 365 days (non-payer or suspended) -> 0, once the history
      covers 365 days; NaN before that, and NaN while fewer than N payments are known.
    """
    out = pd.Series(np.nan, index=idx, dtype="float64")
    if len(idx) == 0:
        return out
    if len(regular):
        freq = payment_frequency(regular)
        cs = regular.cumsum().to_numpy()
        j = np.arange(len(regular))
        ttm_pay = np.full(len(regular), np.nan)
        ok = ~np.isnan(freq)
        k = (j[ok] - freq[ok]).astype(int)  # index of the payment just before the window
        vals = np.where(k >= 0, cs[ok] - cs[np.clip(k, 0, None)], np.where(k == -1, cs[ok], np.nan))
        ttm_pay[ok] = vals
        out = pd.Series(ttm_pay, index=regular.index).reindex(idx).ffill()
        last_pay = pd.Series(regular.index, index=regular.index).reindex(idx).ffill()
        days_since = (pd.Series(idx, index=idx) - last_pay).dt.days
        out = out.mask(days_since > 365, 0.0)
    covered = idx >= idx[0] + pd.Timedelta(days=365)
    no_recent = ~(regular.reindex(idx, fill_value=0.0).rolling("365D").sum() > 0)
    return out.mask(covered & no_recent.to_numpy(), 0.0)


def dividend_features(frame: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """Compute dividend columns aligned to ``frame.index`` (needs ``dividends``, ``raw_close``).

    * ``ttm_yield`` (%) = :func:`ttm_dividends` / raw (unadjusted) close.
    * ``yield_pct``: fraction of the trailing ``yield_window_years`` of TTM yields that are <=
      today's (min 1 year of valid yields).
    * ``div_cut``: on a payment date, regular payment < previous regular x (1 - div_cut_pct).
    * ``div_suspended``: paid before, but no regular payment for > max(2x median gap, 120d).
    * ``div_cut_recent``: a cut or suspension within ``div_cut_lookback_days``.
    * ``fwd_yield`` (%): latest regular payment x payments per year / raw close (0 if suspended).
    """
    idx = frame.index
    out = pd.DataFrame(index=idx)
    divs = frame["dividends"].fillna(0.0)
    regular, special = split_payments(divs, cfg.special_div_mult)

    out["ttm_div"] = ttm_dividends(regular, idx)
    out["ttm_yield"] = (out["ttm_div"] / frame["raw_close"] * 100).replace(
        [np.inf, -np.inf], np.nan
    )
    window = int(cfg.yield_window_years * TRADING_DAYS_PER_YEAR)
    out["yield_pct"] = (
        out["ttm_yield"]
        .rolling(window, min_periods=TRADING_DAYS_PER_YEAR)
        .rank(method="max", pct=True)
    )
    ttm_year_ago = out["ttm_div"].shift(TRADING_DAYS_PER_YEAR)
    out["div_growth"] = ((out["ttm_div"] / ttm_year_ago - 1) * 100).replace(
        [np.inf, -np.inf], np.nan
    )

    out["last_div"] = regular.reindex(idx).ffill()
    prev = regular.shift(1)
    out["prev_div"] = prev.reindex(idx).ffill()
    cut_events = regular < prev * (1 - cfg.div_cut_pct)
    out["div_cut"] = cut_events.reindex(idx, fill_value=False).astype(bool)

    # Suspension: days since the last regular payment vs typical gap between payments.
    pay_dates = pd.Series(regular.index, index=regular.index)
    gaps = pay_dates.diff().dt.days
    median_gap = gaps.rolling(4, min_periods=1).median()
    last_pay = pay_dates.reindex(idx).ffill()
    gap_now = median_gap.reindex(idx).ffill()
    days_since = (pd.Series(idx, index=idx) - last_pay).dt.days
    limit = np.maximum(2 * gap_now, MIN_SUSPENSION_DAYS)
    out["div_suspended"] = (days_since > limit).fillna(False).astype(bool)

    flagged = (out["div_cut"] | out["div_suspended"]).astype(float)
    recent = flagged.rolling(f"{cfg.div_cut_lookback_days}D").max()
    out["div_cut_recent"] = (recent > 0).astype(bool)
    out["special_div"] = special.reindex(idx, fill_value=0.0)

    # Indicated (forward) yield: latest regular payment annualised. Unlike TTM it reflects a
    # cut immediately; 0 once the dividend is suspended.
    if len(regular):
        indicated = pd.Series(regular.to_numpy() * payment_frequency(regular), index=regular.index)
        fwd = indicated.reindex(idx).ffill().mask(out["div_suspended"], 0.0)
    else:
        fwd = pd.Series(np.nan, index=idx, dtype="float64")
    out["fwd_div"] = fwd
    out["fwd_yield"] = (fwd / frame["raw_close"] * 100).replace([np.inf, -np.inf], np.nan)
    return out
