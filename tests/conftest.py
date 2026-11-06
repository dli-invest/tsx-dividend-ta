"""Shared fixtures: deterministic synthetic OHLCV scenarios, a fake fetcher, a network blocker."""

from __future__ import annotations

import socket
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsx_div_ta.clock import FixedClock
from tsx_div_ta.config import AppConfig, DataConfig, config_from_dict
from tsx_div_ta.data.fetcher import FetchResult

START = "2020-01-01"


# --------------------------------------------------------------------------- network guard
@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never touch the network unless explicitly marked ``network``."""
    if request.node.get_closest_marker("network"):
        return

    def guard(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


# --------------------------------------------------------------------------- data factories
def make_ohlcv(
    close: Sequence[float] | np.ndarray,
    start: str = START,
    spread: float = 0.01,
    volume: float | Sequence[float] | np.ndarray = 1_000_000.0,
    dividends: dict[int, float] | None = None,
    adj_factor: float | Sequence[float] | np.ndarray = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Valid raw OHLCV frame (yfinance-standardised column names) on business days.

    open = previous close; high/low wrap open & close by ``spread``; volume jittered ±20%."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    idx = pd.bdate_range(start, periods=n, name="date")
    o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * (1 + spread)
    lo = np.minimum(o, c) * (1 - spread)
    rng = np.random.default_rng(seed)
    vol = np.broadcast_to(np.asarray(volume, dtype=float), (n,)) * rng.uniform(0.8, 1.2, n)
    div = np.zeros(n)
    for i, amt in (dividends or {}).items():
        div[i] = amt
    adj = c * np.broadcast_to(np.asarray(adj_factor, dtype=float), (n,))
    return pd.DataFrame(
        {
            "open": o,
            "high": hi,
            "low": lo,
            "close": c,
            "adj_close": adj,
            "volume": vol,
            "dividends": div,
            "splits": np.zeros(n),
        },
        index=idx,
    )


def trend(
    n: int, start: float = 100.0, daily: float = 0.001, noise: float = 0.0, seed: int = 1
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = daily + rng.normal(0, noise, n)
    steps[0] = 0.0
    return start * np.exp(np.cumsum(steps))


def uptrend(n: int = 600, seed: int = 1) -> np.ndarray:
    return trend(n, daily=0.0015, noise=0.006, seed=seed)


def downtrend(n: int = 600, seed: int = 2) -> np.ndarray:
    return trend(n, daily=-0.0015, noise=0.006, seed=seed)


def sideways(n: int = 600) -> np.ndarray:
    t = np.arange(n)
    return 100 + 5 * np.sin(t / 15.0)


def v_shape(n: int = 600) -> np.ndarray:
    half = n // 2
    return np.concatenate(
        [trend(half, 100, -0.002), trend(n - half, 100 * np.exp(-0.002 * (half - 1)), 0.003)]
    )


def golden_cross_series(n: int = 700) -> np.ndarray:
    """Long decline then a strong rally: SMA50 crosses above SMA200 near the end."""
    down = trend(400, 100, -0.001)
    up = trend(n - 400, down[-1], 0.004)
    return np.concatenate([down, up])


def death_cross_series(n: int = 700) -> np.ndarray:
    up = trend(400, 100, 0.001)
    down = trend(n - 400, up[-1], -0.004)
    return np.concatenate([up, down])


@pytest.fixture
def frames_factory():  # type: ignore[no-untyped-def]
    return make_ohlcv


# --------------------------------------------------------------------------- fake fetcher
class FakeFetcher:
    """In-memory fetcher: returns the slice of a stored frame from ``start`` onward."""

    def __init__(self, data: dict[str, pd.DataFrame], fail: Sequence[str] = ()) -> None:
        self.data = data
        self.fail = set(fail)
        self.calls: list[tuple[tuple[str, ...], date]] = []

    def fetch(self, symbols: Sequence[str], start: date) -> FetchResult:
        self.calls.append((tuple(symbols), start))
        res = FetchResult()
        for s in symbols:
            if s in self.fail or s not in self.data:
                res.errors[s] = "fake failure"
                continue
            df = self.data[s]
            sl = df[df.index >= pd.Timestamp(start)]
            if sl.empty:
                res.errors[s] = "no data"
            else:
                res.frames[s] = sl.copy()
        return res


@pytest.fixture
def fake_fetcher_cls() -> type[FakeFetcher]:
    return FakeFetcher


def clock_at(y: int, m: int, d: int, hh: int = 18, mm: int = 0) -> FixedClock:
    return FixedClock(datetime(y, m, d, hh, mm))


@pytest.fixture
def tmp_config(tmp_path: Path) -> AppConfig:
    """Small-universe config rooted in a temp dir; zero delays so tests are instant."""
    return config_from_dict(
        {
            "universe": {"symbols": ["AAA.TO", "BBB.TO", "CCC.TO"], "benchmark": "XIC.TO"},
            "data": {"batch_delay_s": 0.0, "backoff_base_s": 0.0},
        },
        base_dir=tmp_path,
    )


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(batch_delay_s=0.0, backoff_base_s=0.0)


# --------------------------------------------------------------------------- e2e universe
END = "2026-11-05"
SCAN_CLOCK = FixedClock(datetime(2026, 11, 5, 18, 0))


def start_for(n: int, end: str = END) -> str:
    return pd.bdate_range(end=end, periods=n)[0].strftime("%Y-%m-%d")


def quarterly_divs(
    n: int, amount: float, every: int = 63, cut_at: int | None = None, cut_to: float = 0.0
) -> dict[int, float]:
    out = {}
    for i in range(10, n, every):
        out[i] = cut_to if (cut_at is not None and i >= cut_at) else amount
    return {k: v for k, v in out.items() if v > 0}


def universe_data(n: int = 700) -> dict[str, pd.DataFrame]:
    """Benchmark in an uptrend; AAA strong uptrend payer; BBB downtrend with a dividend cut;
    CCC range-bound high payer. Last bar = 2026-11-05."""
    s = start_for(n)
    return {
        "XIC.TO": make_ohlcv(trend(n, 30, 0.0006, 0.007, seed=11), start=s, seed=11),
        "AAA.TO": make_ohlcv(
            trend(n, 50, 0.0012, 0.009, seed=12), start=s, seed=12, dividends=quarterly_divs(n, 0.5)
        ),
        "BBB.TO": make_ohlcv(
            trend(n, 80, -0.0012, 0.012, seed=13),
            start=s,
            seed=13,
            dividends=quarterly_divs(n, 1.0, cut_at=n - 120, cut_to=0.5),
        ),
        "CCC.TO": make_ohlcv(
            100 + 4 * np.sin(np.arange(n) / 20) + trend(n, 1, 0, 0.004, seed=14),
            start=s,
            seed=14,
            dividends=quarterly_divs(n, 1.3),
        ),
    }


@pytest.fixture
def universe() -> dict[str, pd.DataFrame]:
    return universe_data()
