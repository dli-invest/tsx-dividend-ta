"""Rule engine. Each rule is a pure, vectorised function of the analysis frame that returns a
boolean Series; a value at bar *t* only uses data up to *t* (``shift(n>0)`` and trailing
windows only - never ``shift(-n)``).

Rules come in two kinds:

* EVENT - a discrete occurrence (a cross, a breakout). Emitted if it fired within the last
  ``lookback_bars`` bars (the most recent occurrence is reported).
* STATE - an ongoing condition (high yield, volatility expansion). Emitted only if true on
  the latest bar.

The analysis frame must contain the indicator columns, the dividend columns, and
``market_regime`` / ``stock_regime`` label columns.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from tsx_div_ta.config import SignalConfig
from tsx_div_ta.regime import Regime

BULL, BEAR = Regime.BULL.value, Regime.BEAR.value


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    WATCH = "WATCH"


class Kind(StrEnum):
    EVENT = "EVENT"
    STATE = "STATE"


class Playbook(StrEnum):
    BULL = "BULL"
    BEAR = "BEAR"
    ANY = "ANY"


@dataclass(frozen=True)
class Signal:
    symbol: str
    date: str
    name: str
    direction: str
    strength: float
    market_regime: str
    stock_regime: str
    playbook: str
    reason: str
    values: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.name}|{self.date}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RuleFn = Callable[[pd.DataFrame, SignalConfig], pd.Series]


@dataclass(frozen=True)
class Rule:
    name: str
    direction: Direction
    playbook: Playbook
    kind: Kind
    base_strength: float
    fn: RuleFn
    reason: str
    value_cols: tuple[str, ...]


# --------------------------------------------------------------------------- helpers
def cross_above(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """True on the bar where ``a`` moves from <= ``b`` to > ``b``. NaNs never cross."""
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return ((a > b) & (a.shift(1) <= b_prev)).fillna(False).astype(bool)


def cross_below(a: pd.Series, b: pd.Series | float) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return ((a < b) & (a.shift(1) >= b_prev)).fillna(False).astype(bool)


def _b(x: pd.Series) -> pd.Series:
    return x.fillna(False).astype(bool)


# --------------------------------------------------------------------------- bull playbook
def golden_cross(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """SMA50 crosses above SMA200."""
    return cross_above(d["sma_fast"], d["sma_slow"])


def pullback_uptrend(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Close > SMA200, RSI crosses back above ``rsi_pullback`` (from <= it), and close is within
    ``atr_proximity`` ATRs of EMA21 or SMA50 (buying a dip in an established uptrend)."""
    near = np.minimum((d["close"] - d["ema"]).abs(), (d["close"] - d["sma_fast"]).abs())
    return _b(
        (d["close"] > d["sma_slow"])
        & cross_above(d["rsi"], cfg.rsi_pullback)
        & (near <= cfg.atr_proximity * d["atr"])
    )


def macd_bull_cross(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """MACD crosses above its signal line while both are above zero."""
    return _b(cross_above(d["macd"], d["macd_signal"]) & (d["macd"] > 0) & (d["macd_signal"] > 0))


def breakout(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Close above the prior bar's Donchian(55) high on relative volume >= ``breakout_rvol``
    (``breakout_rvol_bear`` when the market regime is BEAR)."""
    thr = np.where(d["market_regime"] == BEAR, cfg.breakout_rvol_bear, cfg.breakout_rvol)
    return _b((d["close"] > d["dc_upper"].shift(1)) & (d["rvol"] >= thr))


def bb_reversal(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Low touches the lower Bollinger band, close finishes back inside, close > SMA200."""
    return _b(
        (d["low"] <= d["bb_lower"])
        & (d["close"] > d["bb_lower"])
        & (d["close"] <= d["bb_upper"])
        & (d["close"] > d["sma_slow"])
    )


def supertrend_up(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Supertrend direction flips -1 -> +1 with ADX > ``adx_supertrend``."""
    return _b((d["st_dir"] == 1) & (d["st_dir"].shift(1) == -1) & (d["adx"] > cfg.adx_supertrend))


# --------------------------------------------------------------------------- bear playbook
def death_cross(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """SMA50 crosses below SMA200."""
    return cross_below(d["sma_fast"], d["sma_slow"])


def trend_breakdown(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Close crosses below SMA200 with ADX > ``adx_trend`` and -DI > +DI."""
    return _b(
        cross_below(d["close"], d["sma_slow"]) & (d["adx"] > cfg.adx_trend) & (d["dmn"] > d["dmp"])
    )


def failed_rally(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Downtrend (SMA50 < SMA200, close < SMA50); RSI made a local peak on the prior bar inside
    [rsi_bear_ceiling_low, rsi_bear_ceiling_high] and turned down."""
    r1, r2 = d["rsi"].shift(1), d["rsi"].shift(2)
    peak = (
        (r1 >= cfg.rsi_bear_ceiling_low)
        & (r1 <= cfg.rsi_bear_ceiling_high)
        & (r1 > r2)
        & (d["rsi"] < r1)
    )
    return _b(peak & (d["sma_fast"] < d["sma_slow"]) & (d["close"] < d["sma_fast"]))


def macd_bear_cross(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """MACD crosses below its signal line while both are below zero."""
    return _b(cross_below(d["macd"], d["macd_signal"]) & (d["macd"] < 0) & (d["macd_signal"] < 0))


def relative_weakness(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """RS line (close / benchmark) makes a new low vs the prior ``rs_window`` bars."""
    return _b(d["rs"] < d["rs_prior_low"])


def supertrend_down(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Supertrend direction flips +1 -> -1."""
    return _b((d["st_dir"] == -1) & (d["st_dir"].shift(1) == 1))


def oversold_bounce(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Counter-trend WATCH: prior bar's RSI < ``rsi_oversold`` and today closes above the
    prior bar's high."""
    return _b((d["rsi"].shift(1) < cfg.rsi_oversold) & (d["close"] > d["high"].shift(1)))


# --------------------------------------------------------------------------- any regime
def high_yield(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """TTM yield > 0 and at/above the ``yield_percentile`` of its own trailing history, and no
    dividend cut/suspension within ``div_cut_lookback_days`` (after a cut the TTM yield still
    contains pre-cut payments and overstates what the stock now pays)."""
    return _b(
        (d["ttm_yield"] > 0)
        & (d["yield_pct"] >= cfg.yield_percentile)
        & ~d["div_cut_recent"].fillna(False).astype(bool)
    )


def yield_trap(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Yield at/above the ``yield_percentile`` of its history (cut or not) while the stock
    itself is in a confirmed BEAR trend."""
    return _b(
        (d["ttm_yield"] > 0)
        & (d["yield_pct"] >= cfg.yield_percentile)
        & (d["stock_regime"] == BEAR)
    )


def dividend_cut(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Regular dividend cut (or suspension) within ``div_cut_lookback_days``."""
    return _b(d["div_cut_recent"])


def vol_expansion(d: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """ATR% above ``vol_expansion_mult`` x its trailing 1-year median."""
    return _b(d["atr_pct"] > cfg.vol_expansion_mult * d["atr_pct_median"])


RULES: tuple[Rule, ...] = (
    Rule(
        "golden_cross",
        Direction.BULLISH,
        Playbook.BULL,
        Kind.EVENT,
        0.70,
        golden_cross,
        "SMA50 ({sma_fast:.2f}) crossed above SMA200 ({sma_slow:.2f})",
        ("sma_fast", "sma_slow"),
    ),
    Rule(
        "pullback_uptrend",
        Direction.BULLISH,
        Playbook.BULL,
        Kind.EVENT,
        0.60,
        pullback_uptrend,
        "Dip bought in uptrend: RSI back above threshold ({rsi:.1f}), "
        "close {close:.2f} near EMA21 {ema:.2f}",
        ("rsi", "close", "ema", "sma_fast", "atr"),
    ),
    Rule(
        "macd_bull_cross",
        Direction.BULLISH,
        Playbook.BULL,
        Kind.EVENT,
        0.50,
        macd_bull_cross,
        "MACD ({macd:.3f}) crossed above signal ({macd_signal:.3f}) above zero",
        ("macd", "macd_signal"),
    ),
    Rule(
        "breakout",
        Direction.BULLISH,
        Playbook.BULL,
        Kind.EVENT,
        0.70,
        breakout,
        "Close {close:.2f} broke the prior 55-day high on {rvol:.1f}x relative volume",
        ("close", "dc_upper", "rvol"),
    ),
    Rule(
        "bb_reversal",
        Direction.BULLISH,
        Playbook.BULL,
        Kind.EVENT,
        0.50,
        bb_reversal,
        "Tagged lower Bollinger band ({bb_lower:.2f}) and closed back inside above SMA200",
        ("low", "bb_lower", "close", "sma_slow"),
    ),
    Rule(
        "supertrend_up",
        Direction.BULLISH,
        Playbook.BULL,
        Kind.EVENT,
        0.50,
        supertrend_up,
        "Supertrend flipped up (ADX {adx:.1f})",
        ("st_line", "adx"),
    ),
    Rule(
        "death_cross",
        Direction.BEARISH,
        Playbook.BEAR,
        Kind.EVENT,
        0.70,
        death_cross,
        "SMA50 ({sma_fast:.2f}) crossed below SMA200 ({sma_slow:.2f})",
        ("sma_fast", "sma_slow"),
    ),
    Rule(
        "trend_breakdown",
        Direction.BEARISH,
        Playbook.BEAR,
        Kind.EVENT,
        0.80,
        trend_breakdown,
        "Closed below SMA200 ({sma_slow:.2f}) with strong trend "
        "(ADX {adx:.1f}, -DI {dmn:.1f} > +DI {dmp:.1f})",
        ("close", "sma_slow", "adx", "dmn", "dmp"),
    ),
    Rule(
        "failed_rally",
        Direction.BEARISH,
        Playbook.BEAR,
        Kind.EVENT,
        0.60,
        failed_rally,
        "Bear-market rally failed: RSI rolled over ({rsi:.1f}) below SMA50 ({sma_fast:.2f})",
        ("rsi", "close", "sma_fast"),
    ),
    Rule(
        "macd_bear_cross",
        Direction.BEARISH,
        Playbook.BEAR,
        Kind.EVENT,
        0.50,
        macd_bear_cross,
        "MACD ({macd:.3f}) crossed below signal ({macd_signal:.3f}) below zero",
        ("macd", "macd_signal"),
    ),
    Rule(
        "relative_weakness",
        Direction.BEARISH,
        Playbook.BEAR,
        Kind.EVENT,
        0.50,
        relative_weakness,
        "Relative strength vs benchmark at a new 63-day low (RS ROC {rs_roc:.1f}%)",
        ("rs", "rs_prior_low", "rs_roc"),
    ),
    Rule(
        "supertrend_down",
        Direction.BEARISH,
        Playbook.BEAR,
        Kind.EVENT,
        0.50,
        supertrend_down,
        "Supertrend flipped down (line {st_line:.2f})",
        ("st_line", "adx"),
    ),
    Rule(
        "oversold_bounce",
        Direction.WATCH,
        Playbook.BEAR,
        Kind.EVENT,
        0.30,
        oversold_bounce,
        "Oversold bounce attempt after RSI < threshold (counter-trend, low confidence)",
        ("rsi", "close"),
    ),
    Rule(
        "high_yield",
        Direction.BULLISH,
        Playbook.ANY,
        Kind.STATE,
        0.50,
        high_yield,
        "TTM yield {ttm_yield:.2f}% ranks at the {yield_pct:.0%} percentile of its own history",
        ("ttm_yield", "yield_pct"),
    ),
    Rule(
        "yield_trap",
        Direction.WATCH,
        Playbook.ANY,
        Kind.STATE,
        0.60,
        yield_trap,
        "High yield ({ttm_yield:.2f}%) in a downtrend - verify dividend coverage",
        ("ttm_yield", "yield_pct"),
    ),
    Rule(
        "dividend_cut",
        Direction.BEARISH,
        Playbook.ANY,
        Kind.STATE,
        0.90,
        dividend_cut,
        "Dividend cut or suspended recently (last {last_div:.4f} vs previous {prev_div:.4f})",
        ("last_div", "prev_div"),
    ),
    Rule(
        "vol_expansion",
        Direction.WATCH,
        Playbook.ANY,
        Kind.STATE,
        0.40,
        vol_expansion,
        "Volatility expansion: ATR {atr_pct:.2f}% vs 1y median {atr_pct_median:.2f}%",
        ("atr_pct", "atr_pct_median"),
    ),
)

RULES_BY_NAME: dict[str, Rule] = {r.name: r for r in RULES}


def rule_events(d: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """Boolean frame (one column per rule) over the whole history."""
    return pd.DataFrame({r.name: r.fn(d, cfg).astype(bool) for r in RULES}, index=d.index)


def is_counter_regime(direction: str, market_regime: str) -> bool:
    return (direction == Direction.BULLISH.value and market_regime == BEAR) or (
        direction == Direction.BEARISH.value and market_regime == BULL
    )


def _clean_values(row: pd.Series, cols: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cols:
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out[c] = round(float(v), 6)
    return out


def _reason(rule: Rule, values: dict[str, float]) -> str:
    try:
        return rule.reason.format(**values)
    except (KeyError, ValueError, IndexError):
        return (rule.fn.__doc__ or rule.name).strip().splitlines()[0]


def emit_signals(
    symbol: str, d: pd.DataFrame, cfg: SignalConfig, events: pd.DataFrame | None = None
) -> list[Signal]:
    """Turn rule events into :class:`Signal` objects for the latest bars.

    Counter-regime signals (bullish in a BEAR market, bearish in a BULL market) are scaled by
    ``counter_regime_scale``; with ``gate_counter_regime`` the regime-specific playbook signals
    are dropped instead (dividend/risk signals are never gated)."""
    if d.empty:
        return []
    ev = rule_events(d, cfg) if events is None else events
    n = len(d)
    start = max(0, n - cfg.lookback_bars)
    signals: list[Signal] = []
    for rule in RULES:
        col = ev[rule.name].to_numpy()
        if rule.kind is Kind.STATE:
            i = n - 1 if col[-1] else None
        else:
            hits = np.flatnonzero(col[start:])
            i = start + int(hits[-1]) if hits.size else None
        if i is None:
            continue
        row = d.iloc[i]
        mreg, sreg = str(row["market_regime"]), str(row["stock_regime"])
        strength = rule.base_strength
        if is_counter_regime(rule.direction.value, mreg):
            if cfg.gate_counter_regime and rule.playbook is not Playbook.ANY:
                continue
            strength *= cfg.counter_regime_scale
        values = _clean_values(row, rule.value_cols)
        signals.append(
            Signal(
                symbol=symbol,
                date=pd.Timestamp(d.index[i]).strftime("%Y-%m-%d"),
                name=rule.name,
                direction=rule.direction.value,
                strength=round(strength, 4),
                market_regime=mreg,
                stock_regime=sreg,
                playbook=rule.playbook.value,
                reason=_reason(rule, values),
                values=values,
            )
        )
    signals.sort(key=lambda s: (-s.strength, s.name))
    return signals
