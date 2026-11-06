"""Resource budget: a 30-symbol x 5-year scan must stay small and fast (prompt section 2)."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import pytest

from conftest import SCAN_CLOCK, FakeFetcher, make_ohlcv, start_for, trend
from tsx_div_ta.config import config_from_dict
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.store import DataStore
from tsx_div_ta.pipeline import run_scan

N_SYMBOLS, N_BARS = 30, 1260
TIME_BUDGET_S = 60.0
MEMORY_BUDGET_MB = 200.0


@pytest.mark.slow
def test_scan_within_resource_budget(tmp_path: Path) -> None:
    s = start_for(N_BARS)
    divs = {i: 0.5 for i in range(10, N_BARS, 63)}
    data = {
        f"S{i:02d}.TO": make_ohlcv(
            trend(N_BARS, 50, 0.0003, 0.012, seed=i), start=s, seed=i, dividends=divs
        )
        for i in range(N_SYMBOLS)
    }
    data["XIC.TO"] = make_ohlcv(trend(N_BARS, 30, 0.0003, 0.008, seed=99), start=s, seed=99)
    cfg = config_from_dict(
        {
            "universe": {"symbols": sorted(k for k in data if k != "XIC.TO")},
            "data": {"batch_delay_s": 0.0},
        },
        base_dir=tmp_path,
    )
    store = DataStore(ParquetCache(tmp_path / "cache"), FakeFetcher(data), SCAN_CLOCK, cfg.data)
    run_scan(cfg, store, SCAN_CLOCK)  # cold: populates the cache

    tracemalloc.start()
    t0 = time.perf_counter()
    result = run_scan(cfg, store, SCAN_CLOCK)  # warm cache, incremental path
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.exit_code == 0 and len(result.rows) == N_SYMBOLS
    peak_mb = peak / 1e6
    print(f"\nwarm scan: {elapsed:.2f}s (with tracemalloc), peak traced {peak_mb:.1f} MB")
    assert elapsed < TIME_BUDGET_S
    assert peak_mb < MEMORY_BUDGET_MB
