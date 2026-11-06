"""Indicators are checked against independent reference implementations written here, so a
backend swap (pandas-ta <-> pandas-ta-classic) or version bump that changes maths is caught."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import downtrend, make_ohlcv, sideways, trend, uptrend
from tsx_div_ta.config import IndicatorConfig
from tsx_div_ta.data.validate import analysis_frame, clean_ohlcv
from tsx_div_ta.indicators import (
    INDICATOR_COLUMNS,
    TA_BACKEND,
    IndicatorError,
    compute_indicators,
    pick,
)

CFG = IndicatorConfig()
ATOL = 1e-6


def frame(close: np.ndarray, **kw: object) -> pd.DataFrame:
    return analysis_frame(clean_ohlcv(make_ohlcv(close, **kw), "X")[0])  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def noisy() -> pd.DataFrame:
    return frame(trend(600, daily=0.0003, noise=0.012, seed=42))


@pytest.fixture(scope="module")
def ind(noisy: pd.DataFrame) -> pd.DataFrame:
    bench = noisy["close"] * np.linspace(1.0, 1.2, len(noisy))
    return compute_indicators(noisy, CFG, bench)


# ---------------------------------------------------------------- reference implementations
def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing seeded with the SMA of the first n valid values."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    first = int(np.argmax(~np.isnan(vals)))
    if len(vals) - first < n:
        return pd.Series(out, index=s.index)
    out[first + n - 1] = vals[first : first + n].mean()
    for i in range(first + n, len(vals)):
        out[i] = (out[i - 1] * (n - 1) + vals[i]) / n
    return pd.Series(out, index=s.index)


def ema_ref(s: pd.Series, n: int) -> pd.Series:
    """EMA (alpha=2/(n+1)) seeded with the SMA of the first n valid values."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    first = int(np.argmax(~np.isnan(vals)))
    a = 2 / (n + 1)
    out[first + n - 1] = vals[first : first + n].mean()
    for i in range(first + n, len(vals)):
        out[i] = a * vals[i] + (1 - a) * out[i - 1]
    return pd.Series(out, index=s.index)


def rsi_ref(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def atr_ref(h: pd.Series, lo: pd.Series, c: pd.Series, n: int) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    return wilder(tr, n)


def close_enough(a: pd.Series, b: pd.Series, atol: float = ATOL) -> None:
    both = a.notna() & b.notna()
    assert both.sum() > 0
    np.testing.assert_allclose(a[both].to_numpy(), b[both].to_numpy(), atol=atol, rtol=1e-9)


# ---------------------------------------------------------------- tests
def test_backend_is_known() -> None:
    assert TA_BACKEND in {"pandas-ta-classic", "pandas-ta"}


def test_all_stable_columns_present(ind: pd.DataFrame) -> None:
    for col in INDICATOR_COLUMNS:
        assert col in ind.columns
    assert (ind[list(INDICATOR_COLUMNS)].dtypes == "float64").all()


def test_sma_matches_rolling(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    close_enough(ind["sma_fast"], noisy["close"].rolling(50).mean())
    close_enough(ind["sma_slow"], noisy["close"].rolling(200).mean())
    assert ind["sma_slow"].first_valid_index() == noisy.index[199]  # 199 warm-up NaNs


def test_ema_matches_sma_seeded_reference(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    close_enough(ind["ema"], ema_ref(noisy["close"], 21))


def test_rsi_matches_wilder(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    close_enough(ind["rsi"], rsi_ref(noisy["close"], 14))
    assert ind["rsi"].first_valid_index() == noisy.index[14]
    assert ind["rsi"].dropna().between(0, 100).all()


def test_rsi_known_values() -> None:
    """Monotonic rise -> RSI 100; monotonic fall -> RSI 0."""
    up = compute_indicators(frame(np.linspace(10, 20, 40)), CFG)
    dn = compute_indicators(frame(np.linspace(20, 10, 40)), CFG)
    assert up["rsi"].iloc[-1] == pytest.approx(100.0)
    assert dn["rsi"].iloc[-1] == pytest.approx(0.0)


def test_atr_matches_wilder_true_range(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    close_enough(ind["atr"], atr_ref(noisy["high"], noisy["low"], noisy["close"], 14))
    close_enough(ind["atr_pct"], ind["atr"] / noisy["close"] * 100)


def test_macd_matches_ema_difference(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    c = noisy["close"]
    macd = ema_ref(c, 12) - ema_ref(c, 26)
    close_enough(ind["macd"], macd)
    sig = ema_ref(macd.dropna(), 9).reindex(c.index)
    close_enough(ind["macd_signal"], sig)
    close_enough(ind["macd_hist"], ind["macd"] - ind["macd_signal"])


def test_bollinger_matches_population_std(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    c = noisy["close"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    close_enough(ind["bb_mid"], mid)
    close_enough(ind["bb_upper"], mid + 2 * sd)
    close_enough(ind["bb_lower"], mid - 2 * sd)


def test_donchian_roc_obv_cmf(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    h, lo, c, v = noisy["high"], noisy["low"], noisy["close"], noisy["volume"]
    close_enough(ind["dc_upper"], h.rolling(55).max())
    close_enough(ind["dc_lower"], lo.rolling(55).min())
    close_enough(ind["roc"], (c / c.shift(63) - 1) * 100)
    obv_step = np.sign(c.diff()) * v
    close_enough(ind["obv"].diff(), obv_step, atol=1e-3)
    mfv = ((c - lo) - (h - c)) / (h - lo) * v
    close_enough(ind["cmf"], mfv.rolling(20).sum() / v.rolling(20).sum())


def test_adx_bounds_and_direction() -> None:
    up = compute_indicators(frame(uptrend(400)), CFG)
    dn = compute_indicators(frame(downtrend(400)), CFG)
    for d in (up, dn):
        assert d["adx"].dropna().between(0, 100).all()
    assert up["dmp"].iloc[-1] > up["dmn"].iloc[-1]
    assert dn["dmn"].iloc[-1] > dn["dmp"].iloc[-1]


def test_supertrend_direction_and_warmup() -> None:
    up = compute_indicators(frame(uptrend(300)), CFG)
    dn = compute_indicators(frame(downtrend(300)), CFG)
    assert up["st_dir"].iloc[-1] == 1 and dn["st_dir"].iloc[-1] == -1
    assert up["st_dir"].iloc[: CFG.supertrend_length].isna().all()
    assert set(up["st_dir"].dropna().unique()) <= {1.0, -1.0}


def test_derived_columns(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    c = noisy["close"]
    close_enough(ind["sma_slow_slope"], (ind["sma_slow"] / ind["sma_slow"].shift(20) - 1) * 100)
    close_enough(ind["dist_sma_slow"], (c / ind["sma_slow"] - 1) * 100)
    close_enough(ind["hi_52w"], noisy["high"].rolling(252).max())
    close_enough(ind["rvol"], noisy["volume"] / noisy["volume"].rolling(20).mean())
    assert (ind["dist_hi_52w"].dropna() <= 1e-9).all()
    assert (ind["dist_lo_52w"].dropna() >= -1e-9).all()


def test_relative_strength(noisy: pd.DataFrame, ind: pd.DataFrame) -> None:
    bench = noisy["close"] * np.linspace(1.0, 1.2, len(noisy))
    close_enough(ind["rs"], noisy["close"] / bench)
    assert ind["rs_roc"].iloc[-1] < 0  # benchmark outperformed
    close_enough(ind["rs_prior_low"], ind["rs"].shift(1).rolling(63).min())
    assert ind["rs_pos"].dropna().between(0, 1).all()


def test_relative_strength_misaligned_benchmark_dates(noisy: pd.DataFrame) -> None:
    bench = noisy["close"].iloc[::2]  # benchmark missing every other date
    out = compute_indicators(noisy, CFG, bench)
    assert out["rs"].notna().sum() >= len(noisy) - 2  # gaps forward-filled (past data only)


def test_no_benchmark_gives_nan_rs(noisy: pd.DataFrame) -> None:
    out = compute_indicators(noisy, CFG, None)
    assert out["rs"].isna().all() and out["rs_pos"].isna().all()


def test_short_history_returns_nan_not_error() -> None:
    out = compute_indicators(frame(uptrend(30)), CFG)
    assert out["sma_slow"].isna().all()
    assert out["adx"].notna().sum() >= 0  # may be all-NaN; must not raise
    assert len(out) == 30


def test_very_short_history() -> None:
    out = compute_indicators(frame(uptrend(3)), CFG)
    assert len(out) == 3 and out["macd"].isna().all()


def test_flat_prices_do_not_crash() -> None:
    out = compute_indicators(frame(np.full(300, 50.0), spread=0.0), CFG)
    assert out["sma_slow"].iloc[-1] == pytest.approx(50.0)
    assert np.isnan(out["rsi"].iloc[-1]) or 0 <= out["rsi"].iloc[-1] <= 100
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()


def test_zero_volume_gives_nan_not_inf() -> None:
    out = compute_indicators(frame(sideways(300), volume=0.0), CFG)
    assert out["rvol"].isna().all()
    assert not np.isinf(out["cmf"].to_numpy(dtype=float)).any()


def test_no_lookahead_indicators(noisy: pd.DataFrame) -> None:
    """Values at bar t must not change when future bars are appended."""
    bench = noisy["close"] * 1.01
    full = compute_indicators(noisy, CFG, bench)
    for t in (260, 400, 555):
        part = compute_indicators(noisy.iloc[:t], CFG, bench.iloc[:t])
        pd.testing.assert_frame_equal(part, full.iloc[:t], check_exact=False, atol=1e-9, rtol=1e-9)


def test_pick_resolves_prefix_and_rejects_ambiguity() -> None:
    f = pd.DataFrame({"BBL_20_2.0_2.0": [1.0], "BBM_20_2.0_2.0": [2.0]})
    assert pick(f, "BBL_").iloc[0] == 1.0
    with pytest.raises(IndicatorError):
        pick(f, "BB")
    with pytest.raises(IndicatorError):
        pick(f, "ZZZ")


def test_dividend_drop_not_breakdown_with_adjustment() -> None:
    """Raw close drops 5% on ex-date but adjusted series is smooth."""
    n = 300
    raw = np.full(n, 100.0)
    raw[200:] = 95.0
    factor = np.ones(n)
    factor[:200] = 0.95  # Yahoo back-adjusts history before the ex-date
    f = frame(raw, adj_factor=factor)
    assert f["close"].pct_change().abs().max() < 1e-9
    assert f["raw_close"].iloc[-1] == 95.0
