from __future__ import annotations

import random
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import FakeFetcher, clock_at, make_ohlcv, uptrend
from tsx_div_ta.clock import TORONTO, FixedClock, SystemClock
from tsx_div_ta.config import DataConfig
from tsx_div_ta.data.cache import ParquetCache, safe_name
from tsx_div_ta.data.fetcher import YFinanceFetcher, standardize
from tsx_div_ta.data.store import (
    DataStore,
    history_start,
    incremental_start,
    needs_full_refresh,
)
from tsx_div_ta.data.validate import (
    STD_COLUMNS,
    analysis_frame,
    clean_ohlcv,
    drop_partial_bar,
)


# =========================================================================== clock
def test_fixed_clock_naive_is_toronto() -> None:
    assert FixedClock(datetime(2026, 1, 2, 9, 0)).now().tzinfo == TORONTO


def test_fixed_clock_converts_aware() -> None:
    from zoneinfo import ZoneInfo

    utc = datetime(2026, 7, 2, 20, 0, tzinfo=ZoneInfo("UTC"))
    assert FixedClock(utc).now().hour == 16  # EDT = UTC-4


def test_system_clock_is_aware() -> None:
    assert SystemClock().now().tzinfo == TORONTO


# =========================================================================== validation
def test_clean_valid_frame_unchanged() -> None:
    df = make_ohlcv(uptrend(50))
    out, rep = clean_ohlcv(df, "X")
    assert list(out.columns) == list(STD_COLUMNS)
    assert len(out) == 50
    assert rep.issues == []
    assert (out.dtypes == "float64").all()


def test_clean_empty_and_none() -> None:
    out, rep = clean_ohlcv(pd.DataFrame(), "X")
    assert out.empty and rep.issues == ["no data"]


def test_clean_missing_close_column() -> None:
    frame = pd.DataFrame({"open": [1.0]}, index=pd.bdate_range("2024-01-01", periods=1))
    out, rep = clean_ohlcv(frame, "X")
    assert out.empty and "missing required" in rep.issues[0]


def test_clean_dirty_frame() -> None:
    df = make_ohlcv(uptrend(30))
    df.iloc[3, df.columns.get_loc("close")] = np.nan
    df.iloc[4, df.columns.get_loc("close")] = -1.0
    df.iloc[5, df.columns.get_loc("high")] = np.nan
    df.iloc[6, df.columns.get_loc("adj_close")] = np.nan
    dup = df.iloc[[10]].copy()
    dup["close"] = 999.0
    df = pd.concat([df, dup]).iloc[::-1]  # unsorted + duplicate
    out, rep = clean_ohlcv(df, "X")
    text = " | ".join(rep.issues)
    assert "unsorted" in text
    assert "duplicate" in text
    assert "non-positive close dropped" in text
    assert "filled from close" in text
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert rep.rows_dropped == 2
    assert len(out) == 28
    assert not out[["open", "high", "low", "adj_close"]].isna().any().any()


def test_duplicate_keeps_last() -> None:
    df = make_ohlcv([10.0, 11.0, 12.0])
    dup = df.iloc[[1]].copy()
    dup[["close", "adj_close", "high"]] = [50.0, 50.0, 51.0]
    out, _ = clean_ohlcv(pd.concat([df, dup]), "X")
    assert out["close"].iloc[1] == 50.0


@pytest.mark.parametrize(
    ("mutate", "issue"),
    [
        (lambda d: d.__setitem__("high", d["low"] - 1), "high < low"),
        (lambda d: d.__setitem__("close", d["high"] * 1.5), "close outside"),
        (lambda d: d.__setitem__("volume", 0.0), "zero volume"),
    ],
)
def test_quality_flags(mutate, issue: str) -> None:  # type: ignore[no-untyped-def]
    df = make_ohlcv(uptrend(20))
    mutate(df)
    _, rep = clean_ohlcv(df, "X")
    assert any(issue in i for i in rep.issues), rep.issues


def test_big_move_flagged_not_dropped() -> None:
    c = np.r_[np.full(10, 100.0), np.full(10, 50.0)]  # split-like 50% drop
    out, rep = clean_ohlcv(make_ohlcv(c), "X")
    assert len(out) == 20
    assert any("one-day move" in i for i in rep.issues)


def test_tz_aware_index_converted() -> None:
    df = make_ohlcv([1.0, 2.0, 3.0])
    df.index = df.index.tz_localize("America/Toronto")
    out, _ = clean_ohlcv(df, "X")
    assert out.index.tz is None


def test_missing_optional_columns_filled() -> None:
    df = make_ohlcv([1.0, 2.0])[["open", "high", "low", "close"]]
    out, _ = clean_ohlcv(df, "X")
    assert (out["adj_close"] == out["close"]).all()
    assert (out["volume"] == 0).all() and (out["dividends"] == 0).all()


# --------------------------------------------------------------------------- partial bar
@pytest.mark.parametrize(
    ("hh", "mm", "expect_len"),
    [(10, 0, 4), (16, 29, 4), (16, 30, 5), (20, 0, 5)],
)
def test_drop_partial_bar_by_clock(hh: int, mm: int, expect_len: int) -> None:
    df = make_ohlcv([1.0, 2, 3, 4, 5], start="2026-11-02")  # Mon..Fri 2026-11-06
    now = datetime(2026, 11, 6, hh, mm, tzinfo=TORONTO)
    assert len(drop_partial_bar(df, now, time(16, 30))) == expect_len


def test_drop_future_bars_and_empty() -> None:
    df = make_ohlcv([1.0, 2, 3, 4, 5], start="2026-11-02")
    now = datetime(2026, 11, 4, 20, 0, tzinfo=TORONTO)
    assert drop_partial_bar(df, now, time(16, 30)).index[-1] == pd.Timestamp("2026-11-04")
    assert drop_partial_bar(df.iloc[:0], now, time(16, 30)).empty


def test_analysis_frame_adjusts_ohlc() -> None:
    df = make_ohlcv([10.0, 10.0], adj_factor=[0.9, 1.0])
    out, _ = clean_ohlcv(df, "X")
    a = analysis_frame(out)
    assert a["close"].iloc[0] == pytest.approx(9.0)
    assert a["high"].iloc[0] == pytest.approx(out["high"].iloc[0] * 0.9)
    assert a["raw_close"].iloc[0] == 10.0
    assert list(a.columns) == ["open", "high", "low", "close", "volume", "raw_close", "dividends"]


# =========================================================================== fetcher
def _yf_like(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a yf.download(group_by='ticker') shaped frame."""
    inv = {
        v: k
        for k, v in {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "splits",
        }.items()
    }
    parts = {s: f.rename(columns=inv) for s, f in frames.items()}
    return pd.concat(parts, axis=1)


def test_standardize_ticker_first_and_field_first() -> None:
    a, b = make_ohlcv([1.0, 2.0]), make_ohlcv([3.0, 4.0])
    raw = _yf_like({"A": a, "B": b})
    out = standardize(raw, ["A", "B", "MISSING"])
    assert set(out) == {"A", "B"}
    assert list(out["A"].columns) == list(STD_COLUMNS)
    swapped = raw.swaplevel(0, 1, axis=1)
    assert set(standardize(swapped, ["A", "B"])) == {"A", "B"}


def test_standardize_single_level_and_empty() -> None:
    a = _yf_like({"A": make_ohlcv([1.0, 2.0])})["A"]
    assert "A" in standardize(a, ["A"])
    assert standardize(a, ["A", "B"]) == {}
    assert standardize(pd.DataFrame(), ["A"]) == {}
    assert standardize(None, ["A"]) == {}


def test_standardize_drops_all_nan_symbol() -> None:
    a = make_ohlcv([1.0, 2.0])
    bad = a * np.nan
    out = standardize(_yf_like({"A": a, "BAD": bad}), ["A", "BAD"])
    assert set(out) == {"A"}


def test_standardize_without_close_column() -> None:
    raw = pd.DataFrame({("A", "Open"): [1.0]})
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    assert standardize(raw, ["A"]) == {}


class _Boom(Exception):
    pass


class YFRateLimitError(Exception):  # same name as yfinance's class
    pass


def test_fetch_batches_and_delays() -> None:
    data = {s: make_ohlcv([1.0, 2.0]) for s in "ABCDE"}
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def dl(symbols: list[str], start: date) -> pd.DataFrame:
        calls.append(symbols)
        return _yf_like({s: data[s] for s in symbols})

    f = YFinanceFetcher(batch_size=2, batch_delay_s=0.5, downloader=dl, sleep=sleeps.append)
    res = f.fetch(["A", "B", "C", "D", "E", "A"], date(2020, 1, 1))
    assert calls == [["A", "B"], ["C", "D"], ["E"]]
    assert sleeps == [0.5, 0.5]
    assert set(res.frames) == set("ABCDE") and res.errors == {}


def test_fetch_retries_with_backoff_then_succeeds() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def dl(symbols: list[str], start: date) -> pd.DataFrame:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _Boom("temporary")
        return _yf_like({"A": make_ohlcv([1.0])})

    f = YFinanceFetcher(
        max_retries=3, backoff_base_s=1.0, downloader=dl, sleep=sleeps.append, rng=random.Random(0)
    )
    res = f.fetch(["A"], date(2020, 1, 1))
    assert "A" in res.frames
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 2.0 and 2.0 <= sleeps[1] <= 3.0  # base*2^n + U(0, base)


def test_fetch_rate_limit_backs_off_longer() -> None:
    f = YFinanceFetcher(backoff_base_s=1.0, rng=random.Random(0))
    normal = f.backoff(0, rate_limited=False)
    f2 = YFinanceFetcher(backoff_base_s=1.0, rng=random.Random(0))
    assert f2.backoff(0, rate_limited=True) == pytest.approx(normal * 3)

    sleeps: list[float] = []

    def dl(symbols: list[str], start: date) -> pd.DataFrame:
        raise YFRateLimitError("Too Many Requests")

    res = YFinanceFetcher(
        max_retries=2, backoff_base_s=1.0, downloader=dl, sleep=sleeps.append, rng=random.Random(0)
    ).fetch(["A"], date(2020, 1, 1))
    assert res.frames == {} and "YFRateLimitError" in res.errors["A"]
    assert len(sleeps) == 1 and sleeps[0] >= 3.0


def test_fetch_gives_up_and_reports_errors() -> None:
    sleeps: list[float] = []
    res = YFinanceFetcher(
        max_retries=3, downloader=lambda s, d: pd.DataFrame(), sleep=sleeps.append
    ).fetch(["BAD"], date(2020, 1, 1))
    assert res.errors == {"BAD": "empty response"}
    assert len(sleeps) == 2  # no sleep after the final attempt


def test_fetch_partial_batch_marks_missing_symbol() -> None:
    res = YFinanceFetcher(
        downloader=lambda s, d: _yf_like({"A": make_ohlcv([1.0])}), sleep=lambda x: None
    ).fetch(["A", "BAD"], date(2020, 1, 1))
    assert "A" in res.frames
    assert "invalid or delisted" in res.errors["BAD"]


# =========================================================================== cache
def test_cache_roundtrip_and_safe_name(tmp_path: Path) -> None:
    c = ParquetCache(tmp_path / "c")
    assert c.load("RY.TO") is None
    df, _ = clean_ohlcv(make_ohlcv(uptrend(10)), "RY.TO")
    c.save("^GSPTSE", df)
    assert c.path("^GSPTSE").name == "_GSPTSE.parquet"
    assert safe_name("REI-UN.TO") == "REI-UN.TO"
    back = c.load("^GSPTSE")
    assert back is not None
    pd.testing.assert_frame_equal(back, df, check_freq=False, check_names=False)


def test_cache_corrupt_file_recovered(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    c = ParquetCache(tmp_path)
    c.path("X").write_bytes(b"not parquet")
    assert c.load("X") is None
    assert not c.path("X").exists()
    assert "unreadable" in caplog.text


# =========================================================================== store
def test_history_and_incremental_start() -> None:
    assert history_start(date(2026, 11, 5), 5) == date(2021, 11, 5)
    assert incremental_start(pd.Timestamp("2026-11-06"), 5) == date(2026, 10, 30)


def _cleaned(df: pd.DataFrame) -> pd.DataFrame:
    return clean_ohlcv(df, "X")[0]


def test_needs_full_refresh_cases() -> None:
    base = _cleaned(make_ohlcv(uptrend(30)))
    old, new = base.iloc[:25], base.iloc[20:]
    assert needs_full_refresh(old, new) is None
    assert needs_full_refresh(old, new.iloc[:0]) is None
    assert needs_full_refresh(old, base.iloc[26:]) == "no overlap between cache and new data"

    with_div = new.copy()
    with_div.iloc[-1, with_div.columns.get_loc("dividends")] = 0.5
    assert needs_full_refresh(old, with_div) == "new dividend"

    with_split = new.copy()
    with_split.iloc[-1, with_split.columns.get_loc("splits")] = 2.0
    assert needs_full_refresh(old, with_split) == "new split"

    revised_div = new.copy()
    revised_div.iloc[0, revised_div.columns.get_loc("dividends")] = 0.3
    assert needs_full_refresh(old, revised_div) == "dividend/split revised in overlap"

    revised_adj = new.copy()
    revised_adj["adj_close"] *= 0.99
    assert needs_full_refresh(old, revised_adj) == "adjustment factor revised"


def _store(tmp_path: Path, fetcher, clock, cfg: DataConfig) -> DataStore:  # type: ignore[no-untyped-def]
    return DataStore(ParquetCache(tmp_path / "cache"), fetcher, clock, cfg)


def test_store_full_then_incremental(tmp_path: Path, data_cfg: DataConfig) -> None:
    full = make_ohlcv(uptrend(400), start="2025-06-02")  # ends 2026-12-..; clock limits
    clock = clock_at(2026, 6, 1)
    fetcher = FakeFetcher({"A": full})
    store = _store(tmp_path, fetcher, clock, data_cfg)

    r1 = store.load(["A"])
    assert r1.full_refreshed == ["A"]
    assert r1.frames["A"].index[-1] == pd.Timestamp("2026-06-01")
    assert fetcher.calls[-1][1] == date(2021, 6, 1)

    store.clock = clock_at(2026, 6, 10)
    r2 = store.load(["A"])
    assert r2.full_refreshed == []
    assert fetcher.calls[-1][1] == date(2026, 5, 25)  # 5 business days before last cached bar
    assert r2.frames["A"].index[-1] == pd.Timestamp("2026-06-10")
    assert r2.frames["A"].index.is_unique


def test_store_incremental_new_dividend_triggers_full(tmp_path: Path, data_cfg: DataConfig) -> None:
    full = make_ohlcv(uptrend(300), start="2025-06-02")
    fetcher = FakeFetcher({"A": full})
    store = _store(tmp_path, fetcher, clock_at(2026, 3, 2), data_cfg)
    store.load(["A"])
    full2 = full.copy()
    full2.loc[pd.Timestamp("2026-03-04"), "dividends"] = 0.4
    fetcher.data["A"] = full2
    store.clock = clock_at(2026, 3, 6)
    r = store.load(["A"])
    assert r.full_refreshed == ["A"]
    assert r.frames["A"]["dividends"].sum() == pytest.approx(0.4)


def test_store_partial_bar_not_cached(tmp_path: Path, data_cfg: DataConfig) -> None:
    full = make_ohlcv(uptrend(300), start="2025-06-02")
    store = _store(tmp_path, FakeFetcher({"A": full}), clock_at(2026, 3, 4, 11, 0), data_cfg)
    r = store.load(["A"])
    assert r.frames["A"].index[-1] == pd.Timestamp("2026-03-03")
    cached = store.cache.load("A")
    assert cached is not None and cached.index[-1] == pd.Timestamp("2026-03-03")


def test_store_failures_and_stale(tmp_path: Path, data_cfg: DataConfig) -> None:
    full = make_ohlcv(uptrend(300), start="2025-06-02")
    fetcher = FakeFetcher({"A": full, "B": full})
    store = _store(tmp_path, fetcher, clock_at(2026, 3, 2), data_cfg)
    r = store.load(["A", "B", "BAD"])
    assert set(r.frames) == {"A", "B"}
    assert r.failures == {"BAD": "fake failure"}

    fetcher.fail = {"A"}  # incremental refresh fails -> stale cached data used
    store.clock = clock_at(2026, 3, 6)
    r2 = store.load(["A", "B"])
    assert r2.stale == ["A"]
    assert set(r2.frames) == {"A", "B"}
    assert r2.frames["A"].index[-1] == pd.Timestamp("2026-03-02")

    r3 = store.load(["A"], full=True)  # forced full refresh fails -> fall back to cache
    assert r3.stale == ["A"] and "A" in r3.frames


def test_store_offline(tmp_path: Path, data_cfg: DataConfig) -> None:
    full = make_ohlcv(uptrend(300), start="2025-06-02")
    store = _store(tmp_path, FakeFetcher({"A": full}), clock_at(2026, 3, 2), data_cfg)
    store.load(["A"])
    off = _store(tmp_path, None, clock_at(2026, 3, 2), data_cfg)
    r = off.load(["A", "B"], offline=True)
    assert "A" in r.frames
    assert r.failures == {"B": "not in cache (offline mode)"}


def test_store_offline_empty_after_clean(tmp_path: Path, data_cfg: DataConfig) -> None:
    c = ParquetCache(tmp_path / "cache")
    df, _ = clean_ohlcv(make_ohlcv([1.0, 2.0], start="2030-01-01"), "A")  # all future bars
    c.save("A", df)
    r = DataStore(c, None, clock_at(2026, 3, 2), data_cfg).load(["A"], offline=True)
    assert r.failures["A"] == "cached data empty after cleaning"


def test_store_unusable_rows(tmp_path: Path, data_cfg: DataConfig) -> None:
    future = make_ohlcv([1.0, 2.0], start="2030-01-01")
    r = _store(tmp_path, FakeFetcher({"A": future}), clock_at(2026, 3, 2), data_cfg).load(["A"])
    assert r.failures["A"] == "no usable rows after cleaning"


def test_store_trims_to_history_window(tmp_path: Path) -> None:
    cfg = DataConfig(history_years=1, batch_delay_s=0, backoff_base_s=0)
    full = make_ohlcv(uptrend(800), start="2023-01-02")
    r = _store(tmp_path, FakeFetcher({"A": full}), clock_at(2025, 6, 2), cfg).load(["A"])
    assert r.frames["A"].index[0] >= pd.Timestamp("2024-06-02")
