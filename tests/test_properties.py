"""Property-based tests: random but valid OHLCV must never crash the pipeline, and outputs must
respect their documented bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import make_ohlcv
from tsx_div_ta.analysis import build_analysis
from tsx_div_ta.config import AppConfig
from tsx_div_ta.data.validate import clean_ohlcv
from tsx_div_ta.scoring import score_frame
from tsx_div_ta.signals import emit_signals, rule_events

CFG = AppConfig()


@st.composite
def price_paths(draw: st.DrawFn) -> pd.DataFrame:
    n = draw(st.integers(min_value=2, max_value=420))
    returns = draw(
        st.lists(
            st.floats(min_value=-0.15, max_value=0.15, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    start = draw(st.floats(min_value=0.5, max_value=500.0))
    close = start * np.exp(np.cumsum(returns))
    spread = draw(st.floats(min_value=0.0, max_value=0.05))
    vol = draw(st.sampled_from([0.0, 1.0, 1e6]))
    divs = draw(st.dictionaries(st.integers(0, n - 1), st.floats(0.01, 5.0), max_size=12))
    seed = draw(st.integers(0, 1000))
    return make_ohlcv(close, spread=spread, volume=vol, dividends=divs, seed=seed)


SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(raw=price_paths())
def test_pipeline_never_crashes_and_respects_bounds(raw: pd.DataFrame) -> None:
    clean, _ = clean_ohlcv(raw, "P")
    d = build_analysis(clean, CFG, clean["adj_close"] * 1.1, None)
    assert len(d) == len(clean)
    rsi = d["rsi"].dropna()
    assert ((rsi >= -1e-9) & (rsi <= 100 + 1e-9)).all()
    assert d["yield_pct"].dropna().between(0, 1).all()
    assert not np.isinf(d.select_dtypes("number").to_numpy(dtype=float)).any()
    sigs = emit_signals("P", d, CFG.signals)
    last = d.index[-1].strftime("%Y-%m-%d")
    assert all(s.date <= last for s in sigs)
    assert all(0 < s.strength <= 1 for s in sigs)
    card = score_frame(d, sigs, CFG.scoring)
    assert -100 <= card.score <= 100
    assert 0 <= card.confidence <= 1
    assert abs(sum(f.points for f in card.factors) - card.score) < 0.1


@SETTINGS
@given(raw=price_paths(), frac=st.floats(min_value=0.3, max_value=0.95))
def test_events_have_no_lookahead(raw: pd.DataFrame, frac: float) -> None:
    clean, _ = clean_ohlcv(raw, "P")
    cut = max(1, int(len(clean) * frac))
    full = rule_events(build_analysis(clean, CFG, None, None), CFG.signals)
    part = rule_events(build_analysis(clean.iloc[:cut], CFG, None, None), CFG.signals)
    pd.testing.assert_frame_equal(part, full.iloc[:cut])
