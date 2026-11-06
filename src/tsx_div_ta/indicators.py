"""All technical-indicator computation. This is the ONLY module that imports pandas-ta.

pandas-ta's generated column names differ between releases/forks (e.g. ``BBL_20_2.0`` vs
``BBL_20_2.0_2.0``), so they are resolved by prefix and mapped onto stable internal names.
Only the indicators the bot needs are computed - never an "All" strategy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tsx_div_ta.config import IndicatorConfig

try:
    import pandas_ta_classic as ta

    TA_BACKEND = "pandas-ta-classic"
except ImportError:  # pragma: no cover - alternative backend
    import pandas_ta as ta

    TA_BACKEND = "pandas-ta"

TA_VERSION = str(getattr(ta, "version", getattr(ta, "__version__", "unknown")))

HIGH_LOW_MEDIAN_WINDOW = 252

INDICATOR_COLUMNS: tuple[str, ...] = (
    "sma_fast",
    "sma_slow",
    "ema",
    "sma_slow_slope",
    "dist_sma_slow",
    "rsi",
    "roc",
    "atr",
    "atr_pct",
    "atr_pct_median",
    "bb_lower",
    "bb_mid",
    "bb_upper",
    "adx",
    "dmp",
    "dmn",
    "macd",
    "macd_signal",
    "macd_hist",
    "st_line",
    "st_dir",
    "obv",
    "obv_slope",
    "rvol",
    "cmf",
    "dc_upper",
    "dc_lower",
    "hi_52w",
    "lo_52w",
    "dist_hi_52w",
    "dist_lo_52w",
    "rs",
    "rs_roc",
    "rs_prior_low",
    "rs_prior_high",
    "rs_pos",
)


class IndicatorError(RuntimeError):
    """pandas-ta returned an unexpected shape (usually a backend version change)."""


def pick(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Return the single column whose name starts with ``prefix``."""
    matches = [c for c in frame.columns if str(c).startswith(prefix)]
    if len(matches) != 1:
        raise IndicatorError(f"expected one column with prefix {prefix!r}, got {matches}")
    return frame[matches[0]].astype("float64")


def _series(x: pd.Series | pd.DataFrame | None, index: pd.Index) -> pd.Series:
    """pandas-ta returns ``None`` when the input is shorter than the window."""
    if x is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.astype("float64").reindex(index)


def _cols(
    frame: pd.DataFrame | None, index: pd.Index, mapping: dict[str, str]
) -> dict[str, pd.Series]:
    if frame is None:
        return {k: pd.Series(np.nan, index=index, dtype="float64") for k in mapping}
    return {k: pick(frame, p).reindex(index) for k, p in mapping.items()}


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def compute_indicators(
    df: pd.DataFrame, cfg: IndicatorConfig, benchmark_close: pd.Series | None = None
) -> pd.DataFrame:
    """Compute every indicator the bot uses from an :func:`analysis_frame`.

    Every value at bar *t* depends only on bars <= *t* (trailing windows only)."""
    idx = df.index
    o, h, lo, c, v = (df[k].astype("float64") for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=idx)
    for k in ("open", "high", "low", "close", "volume"):
        out[k] = df[k].astype("float64")
    for k in ("raw_close", "dividends"):
        out[k] = df[k].astype("float64") if k in df else (c if k == "raw_close" else 0.0)
    del o

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cols: dict[str, pd.Series] = {}
        cols["sma_fast"] = _series(ta.sma(c, length=cfg.sma_fast), idx)
        cols["sma_slow"] = _series(ta.sma(c, length=cfg.sma_slow), idx)
        cols["ema"] = _series(ta.ema(c, length=cfg.ema), idx)
        cols["rsi"] = _series(ta.rsi(c, length=cfg.rsi), idx)
        cols["roc"] = _series(ta.roc(c, length=cfg.roc), idx)
        cols["atr"] = _series(ta.atr(h, lo, c, length=cfg.atr), idx)
        cols.update(
            _cols(
                ta.bbands(c, length=cfg.bb_length, std=cfg.bb_std),
                idx,
                {"bb_lower": "BBL_", "bb_mid": "BBM_", "bb_upper": "BBU_"},
            )
        )
        cols.update(
            _cols(
                ta.adx(h, lo, c, length=cfg.adx), idx, {"adx": "ADX_", "dmp": "DMP_", "dmn": "DMN_"}
            )
        )
        cols.update(
            _cols(
                ta.macd(c, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal),
                idx,
                {"macd": "MACD_", "macd_hist": "MACDh_", "macd_signal": "MACDs_"},
            )
        )
        cols.update(
            _cols(
                ta.supertrend(
                    h, lo, c, length=cfg.supertrend_length, multiplier=cfg.supertrend_mult
                ),
                idx,
                {"st_line": "SUPERT_", "st_dir": "SUPERTd_"},
            )
        )
        cols["obv"] = _series(ta.obv(c, v), idx)
        cols["cmf"] = _series(ta.cmf(h, lo, c, v, length=cfg.cmf), idx)
        cols.update(
            _cols(
                ta.donchian(h, lo, lower_length=cfg.donchian, upper_length=cfg.donchian),
                idx,
                {"dc_lower": "DCL_", "dc_upper": "DCU_"},
            )
        )

    # Supertrend emits a direction from bar 0 although its line needs `length` bars of ATR.
    warm = np.arange(len(idx)) < cfg.supertrend_length
    st_invalid = warm | cols["st_line"].isna().to_numpy() | (cols["st_line"] <= 0).to_numpy()
    cols["st_line"] = cols["st_line"].mask(st_invalid)
    cols["st_dir"] = cols["st_dir"].mask(st_invalid)
    # Flat prices make RSI 0/0 - leave it NaN (no signal) rather than inventing a value.
    cols["cmf"] = cols["cmf"].replace([np.inf, -np.inf], np.nan)

    sma_slow = cols["sma_slow"]
    cols["sma_slow_slope"] = (_safe_div(sma_slow, sma_slow.shift(cfg.sma_slope_days)) - 1) * 100
    cols["dist_sma_slow"] = (_safe_div(c, sma_slow) - 1) * 100
    cols["atr_pct"] = _safe_div(cols["atr"], c) * 100
    cols["atr_pct_median"] = (
        cols["atr_pct"]
        .rolling(HIGH_LOW_MEDIAN_WINDOW, min_periods=HIGH_LOW_MEDIAN_WINDOW // 2)
        .median()
    )
    vol_avg = v.rolling(cfg.volume_sma, min_periods=cfg.volume_sma).mean()
    cols["rvol"] = _safe_div(v, vol_avg)
    obv = cols["obv"]
    cols["obv_slope"] = _safe_div(obv - obv.shift(cfg.volume_sma), vol_avg * cfg.volume_sma)

    w = cfg.high_low_window
    cols["hi_52w"] = h.rolling(w, min_periods=w).max()
    cols["lo_52w"] = lo.rolling(w, min_periods=w).min()
    cols["dist_hi_52w"] = (_safe_div(c, cols["hi_52w"]) - 1) * 100
    cols["dist_lo_52w"] = (_safe_div(c, cols["lo_52w"]) - 1) * 100

    if benchmark_close is not None and not benchmark_close.empty:
        bench = benchmark_close.astype("float64").reindex(idx).ffill(limit=5)
        rs = _safe_div(c, bench)
    else:
        rs = pd.Series(np.nan, index=idx, dtype="float64")
    n = cfg.rs_window
    cols["rs"] = rs
    cols["rs_roc"] = (_safe_div(rs, rs.shift(n)) - 1) * 100
    cols["rs_prior_low"] = rs.shift(1).rolling(n, min_periods=n).min()
    cols["rs_prior_high"] = rs.shift(1).rolling(n, min_periods=n).max()
    rng = cols["rs_prior_high"] - cols["rs_prior_low"]
    cols["rs_pos"] = _safe_div(rs - cols["rs_prior_low"], rng).clip(0, 1)

    for name in INDICATOR_COLUMNS:
        out[name] = cols[name].astype("float64")
    return out
