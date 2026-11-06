"""Detailed, explainable composite score in [-100, +100].

Structure::

    score = 100 * sum_c (w_c / W) * component_c          (W = sum of regime weights)
    component_c = mean of its available factors           (each factor in [-1, 1])

Each factor's *points* = 100 * (w_c / W) * factor / n_available_factors_in_c, so the points
of all factors add up exactly to the score - every score is fully attributable.

A factor whose inputs are missing (NaN, e.g. no benchmark or short history) is excluded and
lowers ``confidence`` (= available factors / total factors).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tsx_div_ta.config import COMPONENTS, ScoringConfig
from tsx_div_ta.regime import Regime
from tsx_div_ta.signals import Direction, Signal

OUTLOOKS: tuple[str, ...] = ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH")


def _clip(x: float) -> float:
    return float(min(1.0, max(-1.0, x)))


def _num(row: pd.Series, col: str) -> float:
    v = row.get(col, np.nan)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def _ok(*vals: float) -> bool:
    return all(math.isfinite(v) for v in vals)


@dataclass(frozen=True)
class Factor:
    component: str
    name: str
    value: float | None  # normalised to [-1, 1]; None when unavailable
    raw: float | None  # the underlying input, for explanation
    points: float  # contribution to the final score
    note: str


@dataclass(frozen=True)
class ScoreCard:
    score: float
    outlook: str
    confidence: float
    regime: str
    components: dict[str, float]
    factors: list[Factor] = field(default_factory=list)

    def drivers(self, n: int = 3) -> list[str]:
        """The ``n`` factors with the largest absolute contribution, e.g. ``+9.1 trend.sma200``."""
        ranked = sorted(
            (f for f in self.factors if f.value is not None and abs(f.points) > 0),
            key=lambda f: (-abs(f.points), f.component, f.name),
        )
        return [f"{f.points:+.1f} {f.component}.{f.name}" for f in ranked[:n]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["drivers"] = self.drivers()
        return d


FactorFn = Callable[[pd.Series, pd.Series | None, Sequence[Signal]], tuple[float, float, str]]
NAN3 = (math.nan, math.nan, "")


# --------------------------------------------------------------------------- trend
def f_price_vs_sma200(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    d = _num(r, "dist_sma_slow")
    return (_clip(d / 10), d, "+1 at >=10% above SMA200, -1 at >=10% below") if _ok(d) else NAN3


def f_sma50_vs_sma200(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    f, sl = _num(r, "sma_fast"), _num(r, "sma_slow")
    if not _ok(f, sl) or sl == 0:
        return NAN3
    gap = (f / sl - 1) * 100
    return _clip(gap / 5), gap, "SMA50/SMA200 gap; +/-5% saturates"


def f_sma200_slope(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    v = _num(r, "sma_slow_slope")
    return (_clip(v / 2), v, "SMA200 20-day change; +/-2% saturates") if _ok(v) else NAN3


def f_supertrend(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    v = _num(r, "st_dir")
    return (_clip(v), v, "Supertrend direction") if _ok(v) else NAN3


def f_adx_directional(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    adx, dmp, dmn = _num(r, "adx"), _num(r, "dmp"), _num(r, "dmn")
    if not _ok(adx, dmp, dmn):
        return NAN3
    strength = min(1.0, max(0.0, (adx - 20) / 20))
    return float(np.sign(dmp - dmn)) * strength, adx, "ADX strength (20->40) signed by +DI vs -DI"


# --------------------------------------------------------------------------- momentum
def f_rsi(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    v = _num(r, "rsi")
    return (_clip((v - 50) / 25), v, "RSI 75 -> +1, 25 -> -1") if _ok(v) else NAN3


def f_macd_hist(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    h, atr = _num(r, "macd_hist"), _num(r, "atr")
    if not _ok(h, atr) or atr <= 0:
        return NAN3
    return _clip(h / (0.25 * atr)), h, "MACD histogram scaled by 0.25 ATR"


def f_roc(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    v = _num(r, "roc")
    return (_clip(v / 15), v, "3-month rate of change; +/-15% saturates") if _ok(v) else NAN3


# --------------------------------------------------------------------------- relative strength
def f_rs_roc(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    v = _num(r, "rs_roc")
    return (
        (_clip(v / 10), v, "3-month out/under-performance vs benchmark; +/-10%") if _ok(v) else NAN3
    )


def f_rs_position(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    v = _num(r, "rs_pos")
    return ((v - 0.5) * 2, v, "RS line position in its 63-day range") if _ok(v) else NAN3


# --------------------------------------------------------------------------- volume
def f_cmf(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    v = _num(r, "cmf")
    return (_clip(v / 0.2), v, "Chaikin money flow; +/-0.2 saturates") if _ok(v) else NAN3


def f_obv_slope(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    v = _num(r, "obv_slope")
    return (_clip(v * 2), v, "20-day OBV change as share of volume") if _ok(v) else NAN3


def f_volume_confirmation(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    rv, c = _num(r, "rvol"), _num(r, "close")
    pc = _num(p, "close") if p is not None else math.nan
    if not _ok(rv, c, pc):
        return NAN3
    return (
        float(np.sign(c - pc)) * min(1.0, max(0.0, rv - 1.0)),
        rv,
        "today's direction x relative-volume excess",
    )


# --------------------------------------------------------------------------- dividend
def f_yield_percentile(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    y, pct = _num(r, "ttm_yield"), _num(r, "yield_pct")
    if not _ok(y, pct) or y <= 0 or bool(r.get("div_cut_recent", False)):
        return NAN3  # after a cut the TTM yield overstates the payout: excluded
    return (pct - 0.5) * 2, pct, "TTM yield vs own 5y history (cheap = high percentile)"


def f_dividend_growth(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    g = _num(r, "div_growth")
    return (_clip(g / 10), g, "YoY TTM dividend growth; +/-10% saturates") if _ok(g) else NAN3


def f_dividend_safety(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    cut = r.get("div_cut_recent", False)
    y = _num(r, "ttm_yield")
    if bool(cut):
        return -1.0, 1.0, "dividend cut or suspended within lookback"
    if _ok(y) and y > 0:
        return 0.25, 0.0, "no recent cut"
    return NAN3


# --------------------------------------------------------------------------- risk
def f_volatility(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    a, m = _num(r, "atr_pct"), _num(r, "atr_pct_median")
    if not _ok(a, m) or m <= 0:
        return NAN3
    ratio = a / m
    return _clip((1 - ratio) * 2), ratio, "ATR% vs 1y median (calm +, expanding -)"


def f_drawdown(r: pd.Series, p: pd.Series | None, s: Sequence[Signal]) -> tuple[float, float, str]:
    d = _num(r, "dist_hi_52w")
    if not _ok(d):
        return NAN3
    return _clip(1 + d / 10), d, "distance from 52w high: 0% -> +1, -10% -> 0, -20% -> -1"


def f_overextension(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    d = _num(r, "dist_sma_slow")
    if not _ok(d):
        return NAN3
    return -min(1.0, max(0.0, (abs(d) - 15) / 15)), d, "penalty when >15% from SMA200 (either side)"


# --------------------------------------------------------------------------- signals
def f_net_signals(
    r: pd.Series, p: pd.Series | None, s: Sequence[Signal]
) -> tuple[float, float, str]:
    bull = sum(x.strength for x in s if x.direction == Direction.BULLISH.value)
    bear = sum(x.strength for x in s if x.direction == Direction.BEARISH.value)
    net = bull - bear
    return _clip(net), net, "sum of bullish minus bearish signal strengths"


FACTORS: dict[str, tuple[tuple[str, FactorFn], ...]] = {
    "trend": (
        ("sma200", f_price_vs_sma200),
        ("sma50_vs_200", f_sma50_vs_sma200),
        ("sma200_slope", f_sma200_slope),
        ("supertrend", f_supertrend),
        ("adx", f_adx_directional),
    ),
    "momentum": (("rsi", f_rsi), ("macd_hist", f_macd_hist), ("roc_3m", f_roc)),
    "relative_strength": (("rs_roc_3m", f_rs_roc), ("rs_range", f_rs_position)),
    "volume": (("cmf", f_cmf), ("obv_slope", f_obv_slope), ("confirmation", f_volume_confirmation)),
    "dividend": (
        ("yield_percentile", f_yield_percentile),
        ("growth", f_dividend_growth),
        ("safety", f_dividend_safety),
    ),
    "risk": (
        ("volatility", f_volatility),
        ("drawdown", f_drawdown),
        ("overextension", f_overextension),
    ),
    "signals": (("net", f_net_signals),),
}
assert tuple(FACTORS) == COMPONENTS


def outlook_for(score: float) -> str:
    """>=50 STRONG_BULLISH, >=20 BULLISH, (-20, 20) NEUTRAL, <=-20 BEARISH, <=-50 STRONG_BEARISH."""
    if score >= 50:
        return "STRONG_BULLISH"
    if score >= 20:
        return "BULLISH"
    if score > -20:
        return "NEUTRAL"
    if score > -50:
        return "BEARISH"
    return "STRONG_BEARISH"


def score_row(
    row: pd.Series,
    prev: pd.Series | None,
    signals: Sequence[Signal],
    market_regime: str,
    cfg: ScoringConfig,
) -> ScoreCard:
    regime = market_regime if market_regime in cfg.weights else Regime.NEUTRAL.value
    weights = cfg.weights[regime]
    total_w = sum(weights.values())
    factors: list[Factor] = []
    components: dict[str, float] = {}
    available = total = 0
    score = 0.0
    for comp, specs in FACTORS.items():
        evaluated = [(name, *fn(row, prev, signals)) for name, fn in specs]
        vals = [v for _, v, _, _ in evaluated if math.isfinite(v)]
        total += len(specs)
        available += len(vals)
        comp_val = float(np.mean(vals)) if vals else 0.0
        components[comp] = round(comp_val, 4)
        share = weights[comp] / total_w
        score += 100 * share * comp_val
        for name, v, raw, note in evaluated:
            ok = math.isfinite(v)
            pts = 100 * share * v / len(vals) if ok else 0.0
            factors.append(
                Factor(
                    comp,
                    name,
                    round(v, 4) if ok else None,
                    round(raw, 4) if math.isfinite(raw) else None,
                    round(pts, 3),
                    note,
                )
            )
    score = max(-100.0, min(100.0, score))
    return ScoreCard(
        score=round(score, 2),
        outlook=outlook_for(score),
        confidence=round(available / total, 3) if total else 0.0,
        regime=regime,
        components=components,
        factors=factors,
    )


def score_frame(d: pd.DataFrame, signals: Sequence[Signal], cfg: ScoringConfig) -> ScoreCard:
    """Score the latest bar of an analysis frame."""
    row = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else None
    return score_row(row, prev, signals, str(row.get("market_regime", "NEUTRAL")), cfg)


def rank(items: Sequence[tuple[str, float]], top_n: int) -> tuple[list[str], list[str]]:
    """(top bullish, top bearish) symbols; deterministic tie-break by symbol."""
    bulls = sorted((x for x in items if x[1] > 0), key=lambda x: (-x[1], x[0]))
    bears = sorted((x for x in items if x[1] < 0), key=lambda x: (x[1], x[0]))
    return [s for s, _ in bulls[:top_n]], [s for s, _ in bears[:top_n]]
