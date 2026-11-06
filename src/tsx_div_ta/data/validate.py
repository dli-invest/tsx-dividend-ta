"""OHLCV sanity checks, cleaning, partial-bar removal and price adjustment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

import numpy as np
import pandas as pd

from tsx_div_ta.clock import TORONTO

STD_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "splits",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

BIG_MOVE = 0.40
ZERO_VOLUME_STREAK = 5


@dataclass
class QualityReport:
    """Per-symbol data-quality findings. Issues are informational; rows are only dropped
    when they are unusable (missing / non-positive close)."""

    symbol: str
    rows_in: int = 0
    rows_dropped: int = 0
    issues: list[str] = field(default_factory=list)


def _fmt_dates(idx: pd.Index, limit: int = 3) -> str:
    dates = [d.strftime("%Y-%m-%d") for d in idx[:limit]]
    more = f" (+{len(idx) - limit} more)" if len(idx) > limit else ""
    return ", ".join(dates) + more


def _max_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    best = cur = 0
    for v in mask:  # tiny boolean pass; cheaper than building group-bys
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def clean_ohlcv(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, QualityReport]:
    """Return a cleaned copy of ``df`` (columns :data:`STD_COLUMNS`, float64, sorted, unique
    tz-naive dates) plus a :class:`QualityReport`."""
    rep = QualityReport(symbol=symbol, rows_in=len(df))
    empty = pd.DataFrame(columns=list(STD_COLUMNS), index=pd.DatetimeIndex([]), dtype=float)
    if df is None or df.empty:
        rep.issues.append("no data")
        return empty, rep
    missing = [c for c in ("close",) if c not in df.columns]
    if missing:
        rep.issues.append(f"missing required column(s): {missing}")
        rep.rows_dropped = len(df)
        return empty, rep

    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert(TORONTO).tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"

    if not out.index.is_monotonic_increasing:
        rep.issues.append("index was unsorted (sorted)")
        out = out.sort_index(kind="mergesort")
    dup = out.index.duplicated(keep="last")
    if dup.any():
        rep.issues.append(f"{int(dup.sum())} duplicate date(s) removed (kept last)")
        out = out[~dup]

    for col in STD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[list(STD_COLUMNS)].apply(pd.to_numeric, errors="coerce").astype("float64")

    bad = out["close"].isna() | (out["close"] <= 0)
    if bad.any():
        rep.issues.append(f"{int(bad.sum())} row(s) with missing/non-positive close dropped")
        rep.rows_dropped += int(bad.sum())
        out = out[~bad]

    filled = out[["open", "high", "low"]].isna().any(axis=1) | (
        out[["open", "high", "low"]] <= 0
    ).any(axis=1)
    if filled.any():
        rep.issues.append(
            f"{int(filled.sum())} row(s) with missing open/high/low filled from close"
        )
        for col in ("open", "high", "low"):
            out[col] = out[col].where(out[col] > 0, out["close"])
    adj_bad = out["adj_close"].isna() | (out["adj_close"] <= 0)
    out["adj_close"] = out["adj_close"].where(~adj_bad, out["close"])
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    out["dividends"] = out["dividends"].fillna(0.0).clip(lower=0.0)
    out["splits"] = out["splits"].fillna(0.0)

    hl = out["high"] < out["low"]
    if hl.any():
        rep.issues.append(f"high < low on {_fmt_dates(out.index[hl])}")
    tol = 1e-6
    outside = (out["close"] > out["high"] * (1 + tol)) | (out["close"] < out["low"] * (1 - tol))
    if outside.any():
        rep.issues.append(f"close outside [low, high] on {_fmt_dates(out.index[outside])}")
    if _max_run((out["volume"] == 0).to_numpy()) >= ZERO_VOLUME_STREAK:
        rep.issues.append(f"zero volume for >= {ZERO_VOLUME_STREAK} consecutive days")
    move = (out["close"] / out["close"].shift(1) - 1).abs() > BIG_MOVE
    if move.any():
        rep.issues.append(
            f"one-day move > {BIG_MOVE:.0%} on {_fmt_dates(out.index[move])} "
            "(bad data or unadjusted split?)"
        )
    return out, rep


def drop_partial_bar(df: pd.DataFrame, now: datetime, close_buffer: time) -> pd.DataFrame:
    """Drop bars dated in the future, and today's bar if the session is not yet complete
    (``now`` before ``close_buffer`` Toronto time). Only completed bars are analysed."""
    if df.empty:
        return df
    local = now.astimezone(TORONTO)
    today = pd.Timestamp(local.date())
    out = df[df.index <= today]
    if len(out) and out.index[-1] == today and local.time() < close_buffer:
        out = out.iloc[:-1]
    return out


def analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Dividend-adjusted OHLC for indicators plus raw close / dividends for yields.

    Yahoo's ``Close`` is split-adjusted; ``Adj Close`` is split- *and* dividend-adjusted. The
    factor ``adj_close / close`` is applied to open/high/low so the whole bar is consistent and
    ex-dividend drops don't look like breakdowns.
    """
    factor = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pd.DataFrame(
        {
            "open": df["open"] * factor,
            "high": df["high"] * factor,
            "low": df["low"] * factor,
            "close": df["adj_close"],
            "volume": df["volume"],
            "raw_close": df["close"],
            "dividends": df["dividends"],
        },
        index=df.index,
    )
