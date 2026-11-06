"""Every rule: a positive case, a negative case, and an exactly-at-threshold case. The candidate
bar is always the LAST bar of a tiny hand-built frame so the arithmetic is obvious."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from conftest import death_cross_series, golden_cross_series, make_ohlcv, uptrend
from tsx_div_ta.analysis import benchmark_regime, build_analysis
from tsx_div_ta.config import AppConfig, SignalConfig
from tsx_div_ta.data.validate import clean_ohlcv
from tsx_div_ta.signals import (
    RULES,
    RULES_BY_NAME,
    Direction,
    Kind,
    Signal,
    cross_above,
    cross_below,
    emit_signals,
    is_counter_regime,
    rule_events,
)

CFG = SignalConfig()
NAN = np.nan
NUMERIC = (
    "open",
    "high",
    "low",
    "close",
    "sma_fast",
    "sma_slow",
    "ema",
    "rsi",
    "atr",
    "macd",
    "macd_signal",
    "dc_upper",
    "rvol",
    "bb_lower",
    "bb_upper",
    "st_dir",
    "st_line",
    "adx",
    "dmn",
    "dmp",
    "rs",
    "rs_prior_low",
    "rs_roc",
    "ttm_yield",
    "yield_pct",
    "atr_pct",
    "atr_pct_median",
    "last_div",
    "prev_div",
)

Vals = float | Sequence[float]


def mk(n: int = 3, market: str = "NEUTRAL", stock: str = "NEUTRAL", **cols: object) -> pd.DataFrame:
    d = pd.DataFrame(
        {c: np.full(n, NAN) for c in NUMERIC}, index=pd.bdate_range("2024-01-01", periods=n)
    )
    for k, v in cols.items():
        d[k] = v
    d["market_regime"] = market
    d["stock_regime"] = stock
    if "div_cut_recent" not in cols:
        d["div_cut_recent"] = False
    return d


def fires(name: str, d: pd.DataFrame, cfg: SignalConfig = CFG) -> bool:
    return bool(RULES_BY_NAME[name].fn(d, cfg).iloc[-1])


# =========================================================================== helpers
def test_cross_helpers() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, NAN, 5.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert cross_above(a, b).tolist() == [False, False, True, False, False, False, False]
    assert cross_below(a, b).tolist() == [False, False, False, False, False, False, False]
    assert cross_below(pd.Series([3.0, 1.0]), 2.0).tolist() == [False, True]
    assert cross_above(pd.Series([2.0, 2.0]), 2.0).tolist() == [False, False]  # equal != above


def test_rule_registry_is_consistent() -> None:
    assert len({r.name for r in RULES}) == len(RULES)
    for r in RULES:
        assert 0 < r.base_strength <= 1
        assert r.fn.__doc__, f"{r.name} needs a docstring with its exact definition"


# =========================================================================== bull playbook
def test_golden_cross() -> None:
    assert fires("golden_cross", mk(sma_fast=[99.0, 99.0, 101.0], sma_slow=100.0))
    assert fires("golden_cross", mk(sma_fast=[99.0, 100.0, 100.5], sma_slow=100.0))  # from equal
    assert not fires("golden_cross", mk(sma_fast=[101.0, 101.0, 102.0], sma_slow=100.0))
    assert not fires("golden_cross", mk(sma_fast=[99.0, 99.0, 100.0], sma_slow=100.0))  # touch


def _pullback(**over: Vals) -> pd.DataFrame:
    base: dict[str, Vals] = dict(
        close=[105.0, 105.0, 105.0],
        sma_slow=100.0,
        rsi=[35.0, 39.0, 41.0],
        ema=104.0,
        sma_fast=103.0,
        atr=1.0,
    )
    base.update(over)
    return mk(**base)  # type: ignore[arg-type]


def test_pullback_uptrend() -> None:
    assert fires("pullback_uptrend", _pullback())
    assert fires("pullback_uptrend", _pullback(rsi=[35.0, 40.0, 40.1]))  # prior exactly 40
    assert not fires("pullback_uptrend", _pullback(rsi=[35.0, 39.0, 40.0]))  # needs > 40
    assert not fires("pullback_uptrend", _pullback(sma_slow=106.0))  # below SMA200
    assert fires("pullback_uptrend", _pullback(ema=104.0, atr=1.0))  # distance exactly 1 ATR
    assert not fires(
        "pullback_uptrend", _pullback(ema=103.0, sma_fast=103.0, atr=1.0)
    )  # 2 ATR away


def test_macd_bull_cross() -> None:
    assert fires("macd_bull_cross", mk(macd=[0.5, 0.5, 0.7], macd_signal=0.6))
    assert not fires("macd_bull_cross", mk(macd=[-0.5, -0.5, -0.2], macd_signal=-0.3))  # below 0
    assert not fires("macd_bull_cross", mk(macd=[-0.1, -0.1, 0.1], macd_signal=0.0))  # signal == 0


def _breakout(rvol: float, market: str = "NEUTRAL", close: float = 101.0) -> pd.DataFrame:
    return mk(market=market, close=[99.0, 99.0, close], dc_upper=100.0, rvol=rvol)


def test_breakout_thresholds_and_regime() -> None:
    assert fires("breakout", _breakout(1.5))  # exactly at threshold (>=)
    assert not fires("breakout", _breakout(1.49))
    assert not fires("breakout", _breakout(3.0, close=100.0))  # equal to prior high
    assert not fires("breakout", _breakout(1.5, market="BEAR"))  # stricter in BEAR
    assert fires("breakout", _breakout(2.0, market="BEAR"))


def test_breakout_uses_prior_bar_channel() -> None:
    """Today's own high is inside today's Donchian high; must compare to yesterday's."""
    d = mk(close=[99.0, 99.0, 105.0], dc_upper=[100.0, 100.0, 106.0], rvol=2.0)
    assert fires("breakout", d)


def test_bb_reversal() -> None:
    base = dict(close=101.0, bb_lower=100.0, bb_upper=110.0, sma_slow=95.0)
    assert fires("bb_reversal", mk(low=[NAN, NAN, 99.0], **base))
    assert fires("bb_reversal", mk(low=[NAN, NAN, 100.0], **base))  # exactly touches
    assert not fires("bb_reversal", mk(low=[NAN, NAN, 100.01], **base))
    assert not fires("bb_reversal", mk(low=[NAN, NAN, 99.0], **{**base, "sma_slow": 102.0}))
    assert not fires("bb_reversal", mk(low=[NAN, NAN, 99.0], **{**base, "close": 99.5}))


def test_supertrend_up() -> None:
    assert fires("supertrend_up", mk(st_dir=[-1.0, -1.0, 1.0], adx=21.0))
    assert not fires("supertrend_up", mk(st_dir=[-1.0, -1.0, 1.0], adx=20.0))  # needs > 20
    assert not fires("supertrend_up", mk(st_dir=[1.0, 1.0, 1.0], adx=30.0))
    assert not fires("supertrend_up", mk(st_dir=[NAN, NAN, 1.0], adx=30.0))


# =========================================================================== bear playbook
def test_death_cross() -> None:
    assert fires("death_cross", mk(sma_fast=[101.0, 101.0, 99.0], sma_slow=100.0))
    assert not fires("death_cross", mk(sma_fast=[99.0, 99.0, 98.0], sma_slow=100.0))


def test_trend_breakdown() -> None:
    base = dict(close=[101.0, 101.0, 99.0], sma_slow=100.0, dmn=30.0, dmp=10.0)
    assert fires("trend_breakdown", mk(adx=26.0, **base))
    assert not fires("trend_breakdown", mk(adx=25.0, **base))  # needs > 25
    assert not fires("trend_breakdown", mk(adx=30.0, **{**base, "dmn": 10.0, "dmp": 30.0}))


@pytest.mark.parametrize(
    ("peak", "expected"), [(55.0, True), (60.0, True), (57.0, True), (54.9, False), (60.1, False)]
)
def test_failed_rally_rsi_ceiling(peak: float, expected: bool) -> None:
    d = mk(rsi=[50.0, peak, peak - 3], sma_fast=95.0, sma_slow=100.0, close=90.0)
    assert fires("failed_rally", d) is expected


def test_failed_rally_requires_downtrend_and_turn() -> None:
    assert not fires(
        "failed_rally", mk(rsi=[50.0, 57.0, 54.0], sma_fast=105.0, sma_slow=100.0, close=90.0)
    )
    assert not fires(
        "failed_rally", mk(rsi=[50.0, 57.0, 54.0], sma_fast=95.0, sma_slow=100.0, close=96.0)
    )
    assert not fires(
        "failed_rally", mk(rsi=[50.0, 57.0, 58.0], sma_fast=95.0, sma_slow=100.0, close=90.0)
    )
    assert not fires(
        "failed_rally", mk(rsi=[58.0, 57.0, 54.0], sma_fast=95.0, sma_slow=100.0, close=90.0)
    )


def test_macd_bear_cross() -> None:
    assert fires("macd_bear_cross", mk(macd=[-0.5, -0.5, -0.7], macd_signal=-0.6))
    assert not fires("macd_bear_cross", mk(macd=[0.5, 0.5, 0.2], macd_signal=0.3))
    assert not fires("macd_bear_cross", mk(macd=[0.1, 0.1, -0.1], macd_signal=0.0))


def test_relative_weakness() -> None:
    assert fires("relative_weakness", mk(rs=[1.0, 1.0, 0.89], rs_prior_low=0.9))
    assert not fires("relative_weakness", mk(rs=[1.0, 1.0, 0.9], rs_prior_low=0.9))  # equal
    assert not fires("relative_weakness", mk(rs=1.0, rs_prior_low=NAN))


def test_supertrend_down() -> None:
    assert fires("supertrend_down", mk(st_dir=[1.0, 1.0, -1.0]))
    assert not fires("supertrend_down", mk(st_dir=[-1.0, -1.0, -1.0]))


def test_oversold_bounce() -> None:
    assert fires(
        "oversold_bounce", mk(rsi=[20.0, 24.9, 30.0], high=[10.0, 10.0, 12.0], close=[9, 9, 10.5])
    )
    assert not fires("oversold_bounce", mk(rsi=[20.0, 25.0, 30.0], high=10.0, close=[9, 9, 10.5]))
    assert not fires("oversold_bounce", mk(rsi=[20.0, 20.0, 22.0], high=10.0, close=[9, 9, 10.0]))


# =========================================================================== any regime
@pytest.mark.parametrize(
    ("pct", "yld", "expected"),
    [(0.8, 5.0, True), (0.79, 5.0, False), (0.95, 0.0, False), (NAN, 5.0, False)],
)
def test_high_yield(pct: float, yld: float, expected: bool) -> None:
    assert fires("high_yield", mk(yield_pct=pct, ttm_yield=yld)) is expected


def test_high_yield_suppressed_after_cut_but_trap_still_fires() -> None:
    cut = [False, False, True]
    assert not fires("high_yield", mk(yield_pct=0.95, ttm_yield=9.0, div_cut_recent=cut))
    assert fires("yield_trap", mk(stock="BEAR", yield_pct=0.95, ttm_yield=9.0, div_cut_recent=cut))


def test_yield_trap_needs_bear_stock() -> None:
    assert fires("yield_trap", mk(stock="BEAR", yield_pct=0.9, ttm_yield=7.0))
    assert not fires("yield_trap", mk(stock="BULL", yield_pct=0.9, ttm_yield=7.0))


def test_dividend_cut_state() -> None:
    assert fires("dividend_cut", mk(div_cut_recent=[False, False, True]))
    assert not fires("dividend_cut", mk(div_cut_recent=[True, True, False]))


def test_vol_expansion() -> None:
    assert fires("vol_expansion", mk(atr_pct=3.1, atr_pct_median=2.0))
    assert not fires("vol_expansion", mk(atr_pct=3.0, atr_pct_median=2.0))  # exactly 1.5x
    assert not fires("vol_expansion", mk(atr_pct=3.1, atr_pct_median=NAN))


def test_all_nan_frame_fires_nothing() -> None:
    ev = rule_events(mk(n=10), CFG)
    assert not ev.to_numpy().any()


# =========================================================================== emission
def _event_at(n: int, i: int, **extra: object) -> pd.DataFrame:
    fast = np.full(n, 99.0)
    fast[i:] = 101.0
    return mk(n=n, sma_fast=fast, sma_slow=100.0, **extra)


def test_emit_lookback_window_boundary() -> None:
    n = 20
    assert [s.name for s in emit_signals("X", _event_at(n, n - 5), CFG)] == ["golden_cross"]
    assert emit_signals("X", _event_at(n, n - 6), CFG) == []


def test_emit_reports_most_recent_occurrence() -> None:
    fast = np.array([99, 101, 99, 101, 101, 101.0])
    d = mk(n=6, sma_fast=fast, sma_slow=100.0)
    sig = {s.name: s for s in emit_signals("X", d, SignalConfig(lookback_bars=6))}
    assert set(sig) == {"golden_cross", "death_cross"}  # the dip at bar 2 is a real death cross
    assert sig["golden_cross"].date == d.index[3].strftime("%Y-%m-%d")  # latest of bars 1 and 3


def test_state_rules_only_on_last_bar() -> None:
    d = mk(n=5, atr_pct=[3.1, 3.1, 3.1, 3.1, 1.0], atr_pct_median=2.0)
    assert emit_signals("X", d, CFG) == []
    d2 = mk(n=5, atr_pct=[1.0, 1.0, 1.0, 1.0, 3.1], atr_pct_median=2.0)
    assert [s.name for s in emit_signals("X", d2, CFG)] == ["vol_expansion"]
    assert RULES_BY_NAME["vol_expansion"].kind is Kind.STATE


def test_counter_regime_scaling_and_gating() -> None:
    bull_event = _event_at(10, 9)
    s = emit_signals("X", bull_event.assign(market_regime="BEAR"), CFG)
    assert s[0].strength == pytest.approx(0.70 * 0.6)
    s2 = emit_signals("X", bull_event.assign(market_regime="BULL"), CFG)
    assert s2[0].strength == pytest.approx(0.70)
    gated = SignalConfig(gate_counter_regime=True)
    assert emit_signals("X", bull_event.assign(market_regime="BEAR"), gated) == []
    # ANY-playbook signals are never gated, only scaled
    d = mk(n=3, market="BEAR", yield_pct=0.9, ttm_yield=6.0)
    hy = emit_signals("X", d, gated)
    assert [x.name for x in hy] == ["high_yield"] and hy[0].strength == pytest.approx(0.5 * 0.6)
    bear_in_bull = mk(n=3, market="BULL", sma_fast=[101.0, 101.0, 99.0], sma_slow=100.0)
    assert emit_signals("X", bear_in_bull, CFG)[0].strength == pytest.approx(0.7 * 0.6)


def test_is_counter_regime() -> None:
    assert is_counter_regime("BULLISH", "BEAR")
    assert is_counter_regime("BEARISH", "BULL")
    assert not is_counter_regime("WATCH", "BEAR")
    assert not is_counter_regime("BULLISH", "NEUTRAL")


def test_signal_fields_reason_and_sorting() -> None:
    d = mk(n=3, sma_fast=[99.0, 99.0, 101.0], sma_slow=100.0, atr_pct=3.1, atr_pct_median=2.0)
    sigs = emit_signals("RY.TO", d, CFG)
    assert [s.name for s in sigs] == ["golden_cross", "vol_expansion"]  # by strength desc
    g = sigs[0]
    assert g.reason == "SMA50 (101.00) crossed above SMA200 (100.00)"
    assert g.values == {"sma_fast": 101.0, "sma_slow": 100.0}
    assert g.key == f"RY.TO|golden_cross|{g.date}"
    assert g.to_dict()["direction"] == Direction.BULLISH.value
    assert isinstance(g, Signal)


def test_reason_falls_back_to_docstring_when_values_missing() -> None:
    d = mk(n=3, div_cut_recent=True)  # last_div / prev_div NaN
    s = emit_signals("X", d, CFG)
    assert s[0].name == "dividend_cut"
    assert s[0].reason.startswith("Regular dividend cut")


def test_emit_empty_frame() -> None:
    assert emit_signals("X", mk(n=3).iloc[:0], CFG) == []


# =========================================================================== integration
def _analysis(close: np.ndarray, bench: np.ndarray | None = None) -> pd.DataFrame:
    cfg = AppConfig()
    raw = clean_ohlcv(make_ohlcv(close), "X")[0]
    braw = clean_ohlcv(make_ohlcv(bench if bench is not None else uptrend(len(close))), "B")[0]
    bclose, labels = benchmark_regime(braw, cfg)
    return build_analysis(raw, cfg, bclose, labels)


def test_golden_cross_scenario_fires_once_near_cross() -> None:
    d = _analysis(golden_cross_series())
    ev = rule_events(d, CFG)
    crosses = ev.index[ev["golden_cross"]]
    assert len(crosses) == 1
    i = d.index.get_loc(crosses[0])
    assert d["sma_fast"].iloc[i] > d["sma_slow"].iloc[i]
    assert d["sma_fast"].iloc[i - 1] <= d["sma_slow"].iloc[i - 1]


def test_death_cross_scenario() -> None:
    d = _analysis(death_cross_series())
    ev = rule_events(d, CFG)
    assert ev["death_cross"].sum() == 1
    assert d["stock_regime"].iloc[-1] == "BEAR"


@pytest.mark.parametrize("cut", [300, 420, 555, 699])
def test_no_lookahead_signals(cut: int) -> None:
    """Events computed on data up to t equal events from the full history, restricted to <= t."""
    close = golden_cross_series()
    full = rule_events(_analysis(close), CFG)
    part = rule_events(_analysis(close[:cut], uptrend(len(close))[:cut]), CFG)
    pd.testing.assert_frame_equal(part, full.iloc[:cut])
