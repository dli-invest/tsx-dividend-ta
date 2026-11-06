from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsx_div_ta.config import COMPONENTS, ScoringConfig
from tsx_div_ta.scoring import FACTORS, outlook_for, rank, score_frame, score_row
from tsx_div_ta.signals import Signal

CFG = ScoringConfig()


def row(**over: float | bool | str) -> pd.Series:
    base: dict[str, float | bool | str] = dict(
        close=105.0,
        dist_sma_slow=5.0,
        sma_fast=102.0,
        sma_slow=100.0,
        sma_slow_slope=1.0,
        st_dir=1.0,
        adx=30.0,
        dmp=25.0,
        dmn=15.0,
        rsi=60.0,
        macd_hist=0.1,
        atr=2.0,
        roc=6.0,
        rs_roc=2.0,
        rs_pos=0.7,
        cmf=0.05,
        obv_slope=0.1,
        rvol=1.4,
        ttm_yield=4.0,
        yield_pct=0.7,
        div_growth=5.0,
        div_cut_recent=False,
        atr_pct=1.5,
        atr_pct_median=1.5,
        dist_hi_52w=-3.0,
        market_regime="BULL",
    )
    base.update(over)
    return pd.Series(base)


PREV = pd.Series({"close": 104.0})


def sig(direction: str, strength: float) -> Signal:
    return Signal("X", "2026-01-01", "n", direction, strength, "BULL", "BULL", "BULL", "r")


def test_points_sum_to_score_and_bounds() -> None:
    card = score_row(row(), PREV, [sig("BULLISH", 0.7)], "BULL", CFG)
    assert -100 <= card.score <= 100
    assert sum(f.points for f in card.factors) == pytest.approx(card.score, abs=0.05)
    assert set(card.components) == set(COMPONENTS)
    assert card.confidence == 1.0
    assert card.outlook in {"BULLISH", "STRONG_BULLISH"}
    assert len(card.factors) == sum(len(v) for v in FACTORS.values())


def test_extremes_saturate_at_bounds() -> None:
    best = row(
        dist_sma_slow=12.0,
        sma_fast=110.0,
        sma_slow_slope=5.0,
        adx=50.0,
        rsi=90.0,
        macd_hist=5.0,
        roc=40.0,
        rs_roc=30.0,
        rs_pos=1.0,
        cmf=0.5,
        obv_slope=1.0,
        rvol=3.0,
        yield_pct=1.0,
        div_growth=20.0,
        atr_pct=0.5,
        atr_pct_median=2.0,
        dist_hi_52w=0.0,
    )
    card = score_row(best, PREV, [sig("BULLISH", 2.0)], "BULL", CFG)
    assert card.score <= 100 and card.score > 85
    worst = row(
        dist_sma_slow=-14.0,
        sma_fast=90.0,
        sma_slow_slope=-5.0,
        st_dir=-1.0,
        adx=50.0,
        dmp=5.0,
        dmn=40.0,
        rsi=10.0,
        macd_hist=-5.0,
        roc=-40.0,
        rs_roc=-30.0,
        rs_pos=0.0,
        cmf=-0.5,
        obv_slope=-1.0,
        rvol=3.0,
        close=90.0,
        yield_pct=0.0,
        div_growth=-50.0,
        div_cut_recent=True,
        atr_pct=4.0,
        atr_pct_median=1.0,
        dist_hi_52w=-40.0,
    )
    card = score_row(worst, PREV, [sig("BEARISH", 2.0)], "BEAR", CFG)
    assert card.score >= -100 and card.score < -85
    assert card.outlook == "STRONG_BEARISH"


@pytest.mark.parametrize(
    ("col", "lo", "hi"),
    [
        ("dist_sma_slow", -5.0, 5.0),
        ("rsi", 40.0, 60.0),
        ("roc", -3.0, 3.0),
        ("rs_roc", -2.0, 2.0),
        ("cmf", -0.05, 0.05),
        ("yield_pct", 0.3, 0.9),
        ("div_growth", -5.0, 5.0),
        ("atr_pct", 2.5, 1.0),
        ("dist_hi_52w", -15.0, -2.0),
        ("sma_slow_slope", -1.0, 1.0),
    ],
)
def test_score_monotonic_in_each_factor(col: str, lo: float, hi: float) -> None:
    a = score_row(row(**{col: lo}), PREV, [], "NEUTRAL", CFG).score
    b = score_row(row(**{col: hi}), PREV, [], "NEUTRAL", CFG).score
    assert b > a


def test_signals_component() -> None:
    bull = score_row(row(), PREV, [sig("BULLISH", 0.7)], "BULL", CFG)
    bear = score_row(row(), PREV, [sig("BEARISH", 0.7)], "BULL", CFG)
    watch = score_row(row(), PREV, [sig("WATCH", 0.7)], "BULL", CFG)
    assert bull.components["signals"] == 0.7 and bear.components["signals"] == -0.7
    assert watch.components["signals"] == 0.0
    assert bull.score > watch.score > bear.score


def test_regime_weights_applied() -> None:
    r = row(dist_sma_slow=-8.0, sma_fast=96.0, sma_slow_slope=-1.0, st_dir=-1.0, dmp=10.0, dmn=30.0)
    bull = score_row(r, PREV, [], "BULL", CFG)
    bear = score_row(r, PREV, [], "BEAR", CFG)
    assert bull.regime == "BULL" and bear.regime == "BEAR"
    assert bear.components == bull.components
    assert bear.score != bull.score  # same inputs, different weights
    unknown = score_row(r, PREV, [], "UNKNOWN", CFG)
    assert unknown.regime == "NEUTRAL"


def test_missing_inputs_reduce_confidence_not_crash() -> None:
    r = row(rs_roc=math.nan, rs_pos=math.nan, cmf=math.nan, ttm_yield=math.nan)
    card = score_row(r, None, [], "BULL", CFG)
    assert card.confidence < 1.0
    assert card.components["relative_strength"] == 0.0
    names = {(f.component, f.name): f for f in card.factors}
    assert names[("relative_strength", "rs_roc_3m")].value is None
    assert names[("volume", "confirmation")].value is None  # no previous bar
    assert sum(f.points for f in card.factors) == pytest.approx(card.score, abs=0.05)


def test_dividend_cut_forces_safety_negative() -> None:
    ok = score_row(row(), PREV, [], "BULL", CFG)
    cut = score_row(row(div_cut_recent=True), PREV, [], "BULL", CFG)
    assert cut.components["dividend"] < ok.components["dividend"]
    f = {x.name: x for x in cut.factors if x.component == "dividend"}
    assert f["safety"].value == -1.0
    assert f["yield_percentile"].value is None  # TTM overstates the payout right after a cut


def test_nonnumeric_values_are_missing() -> None:
    card = score_row(row(rsi="n/a"), PREV, [], "BULL", CFG)  # type: ignore[arg-type]
    assert {f.name: f.value for f in card.factors}["rsi"] is None


def test_zero_denominators() -> None:
    card = score_row(row(sma_slow=0.0, atr=0.0, atr_pct_median=0.0), PREV, [], "BULL", CFG)
    vals = {f.name: f.value for f in card.factors}
    assert vals["sma50_vs_200"] is None and vals["macd_hist"] is None and vals["volatility"] is None


@pytest.mark.parametrize(
    ("s", "label"),
    [
        (50, "STRONG_BULLISH"),
        (49.9, "BULLISH"),
        (20, "BULLISH"),
        (19.9, "NEUTRAL"),
        (-19.9, "NEUTRAL"),
        (-20, "BEARISH"),
        (-49.9, "BEARISH"),
        (-50, "STRONG_BEARISH"),
    ],
)
def test_outlook_thresholds(s: float, label: str) -> None:
    assert outlook_for(s) == label


def test_drivers_and_to_dict() -> None:
    card = score_row(row(), PREV, [sig("BULLISH", 0.7)], "BULL", CFG)
    drivers = card.drivers(3)
    assert len(drivers) == 3 and all(d[0] in "+-" for d in drivers)
    d = card.to_dict()
    assert d["drivers"] == drivers and len(d["factors"]) == len(card.factors)


def test_rank_deterministic_ties() -> None:
    items = [("B", 10.0), ("A", 10.0), ("C", -5.0), ("D", -5.0), ("E", 0.0)]
    bulls, bears = rank(items, 5)
    assert bulls == ["A", "B"] and bears == ["C", "D"]
    assert rank(items, 1) == (["A"], ["C"])


def test_score_frame_uses_last_bar() -> None:
    d = pd.DataFrame([row(rsi=10.0), row(rsi=90.0)])
    card = score_frame(d, [], CFG)
    assert {f.name: f.raw for f in card.factors}["rsi"] == 90.0
    single = score_frame(d.iloc[:1], [], CFG)
    assert {f.name: f.value for f in single.factors}["confirmation"] is None


def test_nan_safe_with_numpy_values() -> None:
    card = score_row(row(rsi=np.float64(np.nan)), PREV, [], "BULL", CFG)
    assert {f.name: f.value for f in card.factors}["rsi"] is None
