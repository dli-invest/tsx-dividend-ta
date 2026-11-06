from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import downtrend, make_ohlcv, sideways, uptrend
from tsx_div_ta.config import IndicatorConfig, RegimeConfig
from tsx_div_ta.data.validate import analysis_frame, clean_ohlcv
from tsx_div_ta.indicators import compute_indicators
from tsx_div_ta.regime import (
    MarketState,
    Regime,
    align_labels,
    breadth_label,
    compute_breadth,
    confirm_regime,
    raw_regime_codes,
    regime_labels,
)

NAN = np.nan


def ind_for(close: np.ndarray) -> pd.DataFrame:
    return compute_indicators(
        analysis_frame(clean_ohlcv(make_ohlcv(close), "X")[0]), IndicatorConfig()
    )


def test_uptrend_is_bull_downtrend_is_bear() -> None:
    assert regime_labels(ind_for(uptrend()), 3).iloc[-1] == "BULL"
    assert regime_labels(ind_for(downtrend()), 3).iloc[-1] == "BEAR"


def test_sideways_is_not_bull_or_bear_at_end() -> None:
    labels = regime_labels(ind_for(sideways(700)), 3)
    assert labels.iloc[-1] in {"NEUTRAL", "BULL", "BEAR"}
    codes = raw_regime_codes(ind_for(sideways(700))).dropna()
    assert (codes == 0).mean() > 0.5  # mostly neutral in a range


def test_insufficient_history_unknown() -> None:
    labels = regime_labels(ind_for(uptrend(150)), 3)
    assert (labels == Regime.UNKNOWN.value).all()


def test_raw_codes_exact_rules() -> None:
    d = pd.DataFrame(
        {
            "close": [110, 90, 105, 110, 100, NAN],
            "sma_fast": [105, 95, 95, 105, 100, 100],
            "sma_slow": [100, 100, 100, 100, 100, 100],
            "sma_slow_slope": [0.1, -0.1, 0.1, 0.0, 0.1, 0.1],
        }
    )
    codes = raw_regime_codes(d).tolist()
    assert codes[:5] == [1.0, -1.0, 0.0, 0.0, 0.0]  # flat slope / equality => NEUTRAL
    assert np.isnan(codes[5])


def test_hysteresis_blocks_noise_and_switches_on_third_bar() -> None:
    codes = pd.Series([1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 0], dtype=float)
    out = confirm_regime(codes, 3).tolist()
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2:] == [1, 1, 1, 1, 1, 1, 1, 1, -1, -1]


def test_hysteresis_confirm_one_is_raw() -> None:
    codes = pd.Series([1, -1, 0, 1], dtype=float)
    assert confirm_regime(codes, 1).tolist() == codes.tolist()


def test_hysteresis_nan_gap_keeps_previous_regime() -> None:
    codes = pd.Series([NAN, -1, -1, NAN, -1], dtype=float)
    out = confirm_regime(codes, 2).tolist()
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2:] == [-1, -1, -1]


def test_align_labels_uses_past_only() -> None:
    labels = pd.Series(["BULL", "BEAR"], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
    assert align_labels(labels, idx).tolist() == ["UNKNOWN", "BULL", "BEAR"]
    assert align_labels(None, idx).tolist() == ["UNKNOWN"] * 3
    assert align_labels(pd.Series(dtype=object), idx).tolist() == ["UNKNOWN"] * 3


def test_breadth() -> None:
    assert compute_breadth([True, False, None, True]) == pytest.approx(2 / 3)
    assert compute_breadth([None]) is None
    assert compute_breadth([]) is None


@pytest.mark.parametrize(
    ("b", "label"),
    [(None, "UNKNOWN"), (0.61, "BULL"), (0.60, "NEUTRAL"), (0.40, "NEUTRAL"), (0.39, "BEAR")],
)
def test_breadth_label_thresholds(b: float | None, label: str) -> None:
    assert breadth_label(b, RegimeConfig()) == label


def test_market_state_confirmation() -> None:
    cfg = RegimeConfig()
    assert MarketState.build("X", "2026-01-01", "BULL", 0.8, cfg).confirmed is True
    assert MarketState.build("X", "2026-01-01", "BULL", 0.2, cfg).confirmed is False
    assert MarketState.build("X", None, "UNKNOWN", 0.8, cfg).confirmed is None
    s = MarketState.build("X", "2026-01-01", "BEAR", None, cfg)
    assert s.confirmed is None and s.breadth_regime == "UNKNOWN"
