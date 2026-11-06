"""Assemble the per-symbol analysis frame: adjusted prices + indicators + dividend features +
stock/market regime labels. This is the single input to signals, scoring and evaluation."""

from __future__ import annotations

import pandas as pd

from tsx_div_ta.config import AppConfig
from tsx_div_ta.data.validate import analysis_frame
from tsx_div_ta.dividends import dividend_features
from tsx_div_ta.indicators import compute_indicators
from tsx_div_ta.regime import align_labels, regime_labels


def build_analysis(
    raw: pd.DataFrame,
    cfg: AppConfig,
    benchmark_close: pd.Series | None = None,
    market_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """``raw`` is a cleaned frame from the data layer (see ``clean_ohlcv``)."""
    base = analysis_frame(raw)
    ind = compute_indicators(base, cfg.indicators, benchmark_close)
    div = dividend_features(base, cfg.signals)
    out = pd.concat([ind, div], axis=1)
    out["stock_regime"] = regime_labels(out, cfg.regime.confirm_bars)
    out["market_regime"] = align_labels(market_labels, out.index)
    return out


def benchmark_regime(
    raw_bench: pd.DataFrame | None, cfg: AppConfig
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (adjusted benchmark close, confirmed market-regime labels)."""
    if raw_bench is None or raw_bench.empty:
        return None, None
    base = analysis_frame(raw_bench)
    ind = compute_indicators(base, cfg.indicators, None)
    return base["close"], regime_labels(ind, cfg.regime.confirm_bars)
