from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from conftest import SCAN_CLOCK, FakeFetcher, universe_data
from tsx_div_ta.config import AppConfig, config_from_dict
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.store import DataStore
from tsx_div_ta.evaluate import EVAL_COLUMNS, evaluate
from tsx_div_ta.pipeline import STATUS_OK, ScanResult, run_scan, truncate
from tsx_div_ta.report import (
    CSV_COLUMNS,
    SCHEMA_VERSION,
    render_table,
    to_csv,
    to_json,
    write_reports,
)
from tsx_div_ta.state import load_seen, save_seen


def cfg_for(tmp_path: Path, **data: object) -> AppConfig:
    return config_from_dict(
        {
            "universe": {"symbols": ["AAA.TO", "BBB.TO", "CCC.TO"], "benchmark": "XIC.TO"},
            "data": {"batch_delay_s": 0.0, "backoff_base_s": 0.0, **data},
        },
        base_dir=tmp_path,
    )


def scan(tmp_path: Path, data: dict[str, pd.DataFrame] | None = None, **kw: object) -> ScanResult:
    cfg = cfg_for(tmp_path)
    store = DataStore(
        ParquetCache(tmp_path / "cache"),
        FakeFetcher(universe_data() if data is None else data),
        SCAN_CLOCK,
        cfg.data,
    )
    return run_scan(cfg, store, SCAN_CLOCK, **kw)  # type: ignore[arg-type]


def test_full_scan_shapes(tmp_path: Path) -> None:
    r = scan(tmp_path)
    assert r.exit_code == 0
    assert r.as_of == "2026-11-05"
    assert r.market.regime == "BULL"
    assert {x.symbol for x in r.rows} == {"AAA.TO", "BBB.TO", "CCC.TO"}
    assert all(x.status == STATUS_OK for x in r.rows)
    scores = [x.score for x in r.rows]
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
    by = {x.symbol: x for x in r.rows}
    assert by["AAA.TO"].score > by["BBB.TO"].score  # type: ignore[operator]
    assert by["BBB.TO"].stock_regime == "BEAR"
    assert "dividend_cut" in by["BBB.TO"].signals
    assert by["AAA.TO"].stop_long < by["AAA.TO"].close < by["AAA.TO"].stop_short  # type: ignore[operator]
    assert r.market.breadth is not None
    assert r.top_bullish and r.top_bullish[0] == r.rows[0].symbol
    assert r.meta["ta_backend"].startswith("pandas-ta")
    for x in r.rows:
        assert x.confidence is not None and 0 < x.confidence <= 1
        assert len(x.drivers) == 3


def test_new_signal_flags(tmp_path: Path) -> None:
    first = scan(tmp_path)
    assert all(set(x.new_signals) == set(x.signals) for x in first.rows)
    second = scan(tmp_path, seen=set(first.signal_keys))
    assert all(x.new_signals == [] for x in second.rows)


def test_partial_failure_and_insufficient_history(tmp_path: Path) -> None:
    data = universe_data()
    data["CCC.TO"] = data["CCC.TO"].iloc[-100:]
    del data["BBB.TO"]
    r = scan(tmp_path, data)
    by = {x.symbol: x for x in r.rows}
    assert by["BBB.TO"].status == "FAILED"
    assert by["CCC.TO"].status == "INSUFFICIENT_HISTORY" and by["CCC.TO"].bars == 100
    assert r.failures.keys() == {"BBB.TO"}
    assert r.exit_code == 1


def test_no_benchmark_degrades_gracefully(tmp_path: Path) -> None:
    data = universe_data()
    del data["XIC.TO"]
    r = scan(tmp_path, data)
    assert r.market.regime == "UNKNOWN"
    assert r.exit_code == 1  # benchmark failure counts as partial
    assert all(x.status == STATUS_OK for x in r.rows)


def test_nothing_analysable_is_fatal(tmp_path: Path) -> None:
    r = scan(tmp_path, {})
    assert r.exit_code == 2


def test_analysis_error_is_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tsx_div_ta.pipeline as pl

    real = pl.build_analysis

    def boom(raw, cfg, b, labels):  # type: ignore[no-untyped-def]
        if raw["close"].iloc[0] == universe_data()["BBB.TO"]["close"].iloc[0]:
            raise ValueError("synthetic failure")
        return real(raw, cfg, b, labels)

    monkeypatch.setattr(pl, "build_analysis", boom)
    r = scan(tmp_path)
    by = {x.symbol: x for x in r.rows}
    assert by["BBB.TO"].status == "ERROR" and "synthetic failure" in (by["BBB.TO"].message or "")
    assert by["AAA.TO"].status == STATUS_OK and r.exit_code == 1


def test_as_of_truncates(tmp_path: Path) -> None:
    r = scan(tmp_path, as_of=date(2026, 6, 30))
    assert r.as_of == "2026-06-30"
    assert all(x.date == "2026-06-30" for x in r.rows)
    assert r.meta["requested_as_of"] == "2026-06-30"
    assert truncate({"A": universe_data()["AAA.TO"]}, None)["A"].index[-1] == pd.Timestamp(
        "2026-11-05"
    )


def test_as_of_equals_scan_on_truncated_data(tmp_path: Path) -> None:
    """No look-ahead end-to-end: --as-of on full data == scan on data that ends that day."""
    cut = pd.Timestamp("2026-08-14")
    hist = scan(tmp_path / "a", as_of=cut.date())
    trimmed = {k: v[v.index <= cut] for k, v in universe_data().items()}
    live = scan(tmp_path / "b", trimmed)
    assert [(x.symbol, x.score, x.signals) for x in hist.rows] == [
        (x.symbol, x.score, x.signals) for x in live.rows
    ]


def test_offline_scan_uses_cache(tmp_path: Path) -> None:
    scan(tmp_path)
    cfg = cfg_for(tmp_path)
    store = DataStore(ParquetCache(tmp_path / "cache"), None, SCAN_CLOCK, cfg.data)
    r = run_scan(cfg, store, SCAN_CLOCK, offline=True)
    assert r.exit_code == 0 and len(r.rows) == 3


def test_symbols_override(tmp_path: Path) -> None:
    r = scan(tmp_path, symbols=["AAA.TO"])
    assert [x.symbol for x in r.rows] == ["AAA.TO"]


# =========================================================================== reports
def test_json_payload(tmp_path: Path) -> None:
    r = scan(tmp_path)
    p = json.loads(to_json(r))
    assert p["schema_version"] == SCHEMA_VERSION
    assert "Not financial advice" in p["disclaimer"]
    assert p["market"]["regime"] == "BULL"
    assert len(p["rows"]) == 3 and "factors" in p["rows"][0]
    assert p["rows"][0]["factors"][0].keys() >= {"component", "name", "points"}
    assert to_json(r) == to_json(r)  # deterministic


def test_csv_and_table(tmp_path: Path) -> None:
    r = scan(tmp_path)
    lines = to_csv(r).splitlines()
    assert lines[0].split(",") == list(CSV_COLUMNS)
    assert len(lines) == 4
    t = render_table(r)
    assert "Market regime (XIC.TO): BULL" in t
    assert "Top bullish:" in t and "Score drivers" in t and "Not financial advice" in t
    assert "*" in t  # new-signal markers on first run


def test_table_shows_failures_status_and_review(tmp_path: Path) -> None:
    data = universe_data()
    del data["BBB.TO"]
    r = scan(tmp_path, data)
    r.review = {"model": "m", "text": "line one\nline two"}
    r.rows[0].stale = True
    t = render_table(r)
    assert "FAILED" in t and "Failures:" in t and "AI review (m, ?, 0 turn(s), est. n/a):" in t
    assert "  line two" in t
    assert "[stale]" in t


def test_write_reports(tmp_path: Path) -> None:
    r = scan(tmp_path)
    paths = write_reports(r, tmp_path / "reports")
    assert paths["json"] == tmp_path / "reports" / "2026-11-05" / "signals.json"
    assert all(p.exists() for p in paths.values())
    assert not list((tmp_path / "reports").rglob("*.tmp"))


# =========================================================================== state
def test_state_roundtrip_and_corrupt(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / "s" / "state.json"
    assert load_seen(p) == set()
    save_seen(p, ["b", "a", "a"], "2026-01-01")
    assert load_seen(p) == {"a", "b"}
    p.write_text("{not json")
    assert load_seen(p) == set() and "unreadable" in caplog.text


# =========================================================================== evaluate
def test_evaluate_table() -> None:
    cfg = AppConfig()
    t = evaluate(universe_data(), cfg, [5, 20], ["AAA.TO", "BBB.TO", "CCC.TO", "MISSING"])
    assert list(t.columns) == list(EVAL_COLUMNS)
    assert set(t["horizon"]) == {5, 20}
    gc = t[(t["rule"] == "dividend_cut") & (t["horizon"] == 20)].iloc[0]
    assert gc["count"] > 0 and 0 <= gc["hit_rate"] <= 1
    empty = t[t["count"] == 0]
    assert empty["hit_rate"].isna().all()
