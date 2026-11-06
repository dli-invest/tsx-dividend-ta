"""Market and per-stock trend regimes with hysteresis, plus market breadth."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from tsx_div_ta.config import RegimeConfig


class Regime(StrEnum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


_CODE_TO_LABEL = {1.0: Regime.BULL.value, -1.0: Regime.BEAR.value, 0.0: Regime.NEUTRAL.value}


def raw_regime_codes(ind: pd.DataFrame) -> pd.Series:
    """Per-bar unconfirmed regime: +1 BULL, -1 BEAR, 0 NEUTRAL, NaN when inputs missing.

    * BULL: close > SMA200 and SMA50 > SMA200 and SMA200 slope > 0
    * BEAR: close < SMA200 and SMA50 < SMA200 and SMA200 slope < 0
    """
    c, f, s, slope = ind["close"], ind["sma_fast"], ind["sma_slow"], ind["sma_slow_slope"]
    valid = c.notna() & f.notna() & s.notna() & slope.notna()
    bull = (c > s) & (f > s) & (slope > 0)
    bear = (c < s) & (f < s) & (slope < 0)
    codes = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
    codes[~valid.to_numpy()] = np.nan
    return pd.Series(codes, index=ind.index, dtype="float64")


def confirm_regime(codes: pd.Series, confirm_bars: int) -> pd.Series:
    """Hysteresis: the regime switches only after the raw code has held for ``confirm_bars``
    consecutive bars; until the first confirmation the regime is NaN (UNKNOWN). Vectorised."""
    run_id = (codes != codes.shift()).cumsum()
    run_len = codes.groupby(run_id).cumcount() + 1
    confirmed = codes.where((run_len >= confirm_bars) & codes.notna())
    return confirmed.ffill()


def regime_labels(ind: pd.DataFrame, confirm_bars: int) -> pd.Series:
    """Confirmed regime label per bar ("BULL"/"BEAR"/"NEUTRAL"/"UNKNOWN")."""
    conf = confirm_regime(raw_regime_codes(ind), confirm_bars)
    return conf.map(_CODE_TO_LABEL).fillna(Regime.UNKNOWN.value).astype(object)


def align_labels(labels: pd.Series | None, index: pd.Index) -> pd.Series:
    """Align a (benchmark) label series onto another index using only past values."""
    if labels is None or labels.empty:
        return pd.Series(Regime.UNKNOWN.value, index=index, dtype=object)
    union = labels.index.union(index)
    aligned = labels.reindex(union).ffill().reindex(index)
    return aligned.fillna(Regime.UNKNOWN.value).astype(object)


def compute_breadth(above_sma_slow: Iterable[bool | None]) -> float | None:
    """Fraction of symbols (with a valid SMA200) trading above their SMA200."""
    vals = [bool(v) for v in above_sma_slow if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def breadth_label(breadth: float | None, cfg: RegimeConfig) -> str:
    if breadth is None:
        return Regime.UNKNOWN.value
    if breadth > cfg.breadth_bull:
        return Regime.BULL.value
    if breadth < cfg.breadth_bear:
        return Regime.BEAR.value
    return Regime.NEUTRAL.value


@dataclass(frozen=True)
class MarketState:
    benchmark: str
    date: str | None
    regime: str
    breadth: float | None
    breadth_regime: str
    confirmed: bool | None  # does breadth agree with the price regime?

    @classmethod
    def build(
        cls, benchmark: str, date: str | None, regime: str, breadth: float | None, cfg: RegimeConfig
    ) -> MarketState:
        b_label = breadth_label(breadth, cfg)
        if regime in (Regime.UNKNOWN.value,) or b_label == Regime.UNKNOWN.value:
            confirmed = None
        else:
            confirmed = b_label == regime
        return cls(benchmark, date, regime, breadth, b_label, confirmed)
