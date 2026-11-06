from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
from tests.test_notify import HOOK, FakeOpener, RecordingNotifier, _cfg

from conftest import SCAN_CLOCK, FakeFetcher, universe_data
from tsx_div_ta.__main__ import main
from tsx_div_ta.chart import BEAR, BULL, NEUTRAL, _polarity_color, render_chart, spread_labels
from tsx_div_ta.config import config_from_dict
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.store import DataStore
from tsx_div_ta.notify import DiscordNotifier, build_payload
from tsx_div_ta.pipeline import BENCHMARK_CHART_BARS, ScanResult, run_scan

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def result(tmp_path: Path) -> ScanResult:
    cfg = config_from_dict(
        {"universe": {"symbols": ["AAA.TO", "BBB.TO", "CCC.TO"]}, "data": {"batch_delay_s": 0.0}},
        base_dir=tmp_path,
    )
    store = DataStore(
        ParquetCache(tmp_path / "c"), FakeFetcher(universe_data()), SCAN_CLOCK, cfg.data
    )
    return run_scan(cfg, store, SCAN_CLOCK)


# --------------------------------------------------------------------------- helpers
def test_spread_labels_separates_and_keeps_order() -> None:
    placed = spread_labels([57.23, 56.90, 54.91], 1.0)
    assert placed[0] > placed[1] > placed[2]
    s = sorted(placed)
    assert all(b - a >= 1.0 - 1e-9 for a, b in itertools.pairwise(s))
    assert abs(sum(placed) / 3 - (57.23 + 56.90 + 54.91) / 3) < 1e-9  # stays centred


def test_spread_labels_leaves_distant_values_alone() -> None:
    assert spread_labels([10.0, 50.0], 1.0) == [10.0, 50.0]
    assert spread_labels([3.0], 1.0) == [3.0]
    assert spread_labels([], 1.0) == []


@pytest.mark.parametrize(
    ("score", "color"),
    [(20.0, BULL), (19.9, NEUTRAL), (-19.9, NEUTRAL), (-20.0, BEAR), (0.0, NEUTRAL)],
)
def test_polarity_thresholds_match_outlook(score: float, color: str) -> None:
    assert _polarity_color(score) == color


# --------------------------------------------------------------------------- rendering
def test_result_carries_one_year_of_benchmark(result: ScanResult) -> None:
    assert result.benchmark is not None
    assert len(result.benchmark) == BENCHMARK_CHART_BARS
    assert list(result.benchmark.columns) == ["close", "sma_fast", "sma_slow", "regime"]


def test_render_two_panels(result: ScanResult, tmp_path: Path) -> None:
    p = render_chart(result, tmp_path / "out" / "chart.png")
    assert p is not None and p.read_bytes()[:8] == PNG_MAGIC
    assert 10_000 < p.stat().st_size < 1_000_000


def test_render_without_benchmark_or_rows(result: ScanResult, tmp_path: Path) -> None:
    bench = result.benchmark
    result.benchmark = None
    assert render_chart(result, tmp_path / "a.png") is not None  # scores only
    result.benchmark = bench
    rows = result.rows
    result.rows = []
    assert render_chart(result, tmp_path / "b.png") is not None  # benchmark only
    result.benchmark = None
    assert render_chart(result, tmp_path / "c.png") is None  # nothing to plot
    result.rows = rows


def test_render_without_matplotlib(
    result: ScanResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert render_chart(result, tmp_path / "x.png") is None
    assert "matplotlib not installed" in caplog.text


# --------------------------------------------------------------------------- discord wiring
def test_payload_embeds_attached_image(result: ScanResult) -> None:
    p = build_payload(result, "chart.png")
    assert p["embeds"][0]["image"] == {"url": "attachment://chart.png"}
    assert "image" not in build_payload(result)["embeds"][0]


def test_send_attaches_chart(result: ScanResult, tmp_path: Path) -> None:
    img = render_chart(result, tmp_path / "chart.png")
    op = FakeOpener([200])
    DiscordNotifier(HOOK, opener=op, sleep=lambda s: None).send(result, [], image=img)
    body = op.requests[0].data
    assert b'filename="chart.png"' in body and b"attachment://chart.png" in body
    assert PNG_MAGIC in body


def test_send_ignores_missing_image(result: ScanResult, tmp_path: Path) -> None:
    op = FakeOpener([200])
    DiscordNotifier(HOOK, opener=op, sleep=lambda s: None).send(
        result, [], image=tmp_path / "nope.png"
    )
    assert b"attachment://" not in op.requests[0].data


def test_cli_notify_sends_chart(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rn = RecordingNotifier()
    code = main(
        ["scan", "--config", str(_cfg(tmp_path)), "--notify"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        notifier=rn,
    )
    assert code == 0
    assert rn.image == tmp_path / "reports" / "2026-11-05" / "chart.png"
    assert rn.image.read_bytes()[:8] == PNG_MAGIC  # kept with the saved reports


def test_cli_notify_no_save_uses_temp_chart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rn = RecordingNotifier()
    main(
        ["scan", "--config", str(_cfg(tmp_path)), "--notify", "--no-save"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        notifier=rn,
    )
    assert rn.image is not None and rn.image.name == "chart.png"
    assert not rn.image.exists()  # temp dir cleaned up after sending


def test_cli_chart_failure_still_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tsx_div_ta import chart

    def boom(result: ScanResult, path: Path) -> None:
        raise RuntimeError("font cache exploded")

    monkeypatch.setattr(chart, "render_chart", boom)
    rn = RecordingNotifier()
    code = main(
        ["scan", "--config", str(_cfg(tmp_path)), "--notify", "--no-save"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        notifier=rn,
    )
    assert code == 0 and len(rn.sent) == 1 and rn.image is None


def test_cli_chart_disabled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfgp = _cfg(tmp_path)
    cfgp.write_text(cfgp.read_text() + "[notify]\nchart = false\n")
    rn = RecordingNotifier()
    main(
        ["scan", "--config", str(cfgp), "--notify", "--no-save"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        notifier=rn,
    )
    assert len(rn.sent) == 1 and rn.image is None
