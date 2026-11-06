from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from conftest import SCAN_CLOCK, FakeFetcher, universe_data
from tsx_div_ta.__main__ import build_parser, main

GOLDEN = Path(__file__).parent / "golden" / "scan_golden.json"


def write_cfg(tmp_path: Path, extra: str = "") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "config.toml"
    p.write_text(
        '[universe]\nsymbols = ["AAA.TO", "BBB.TO", "CCC.TO"]\nbenchmark = "XIC.TO"\n'
        "[data]\nbatch_delay_s = 0.0\nbackoff_base_s = 0.0\n" + extra
    )
    return p


def run(tmp_path: Path, *args: str, fetcher: Any = None, **kw: Any) -> int:
    cfg = write_cfg(tmp_path)
    return main(
        [args[0], "--config", str(cfg), *args[1:]],
        fetcher=fetcher or FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        **kw,
    )


def approx_equal(a: Any, b: Any, path: str = "$") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"keys differ at {path}"
        for k in a:
            approx_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"length differs at {path}"
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            approx_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-3), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_scan_table_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "scan") == 0
    out = capsys.readouterr().out
    assert "Market regime (XIC.TO)" in out
    assert (tmp_path / "reports" / "2026-11-05" / "signals.json").exists()
    assert (tmp_path / "state" / "last_run.json").exists()
    assert (tmp_path / "logs" / "bot.log").exists()


def test_scan_json_matches_golden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "scan", "--format", "json", "--no-save") == 0
    payload = json.loads(capsys.readouterr().out)
    payload["meta"].pop("ta_backend")  # backend/version string may legitimately differ
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN.read_text())
    approx_equal(payload, golden)


def test_scan_csv_and_no_save(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "scan", "--format", "csv", "--no-save") == 0
    assert capsys.readouterr().out.startswith("symbol,status,date")
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "state").exists()


def test_second_run_has_no_new_signals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(tmp_path, "scan", "--format", "json")
    first = json.loads(capsys.readouterr().out)
    run(tmp_path, "scan", "--format", "json")
    second = json.loads(capsys.readouterr().out)
    assert any(r["new_signals"] for r in first["rows"])
    assert all(r["new_signals"] == [] for r in second["rows"])


def test_offline_and_as_of(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(tmp_path, "scan", "--no-save")
    capsys.readouterr()
    state = tmp_path / "state" / "last_run.json"
    assert run(tmp_path, "scan", "--offline", "--as-of", "2026-09-30", "--format", "json") == 0
    p = json.loads(capsys.readouterr().out)
    assert p["as_of"] == "2026-09-30"
    assert not state.exists()  # historical runs never touch state


def test_partial_failure_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "scan", fetcher=FakeFetcher(universe_data(), fail=["BBB.TO"])) == 1


def test_fatal_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "scan", fetcher=FakeFetcher({})) == 2
    assert run(tmp_path, "scan", "--offline") == 2  # empty cache


def test_bad_config_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[data]\nnope = 1\n")
    assert main(["scan", "--config", str(p)]) == 2
    assert "config error" in capsys.readouterr().err


def test_default_config_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    write_cfg(tmp_path)
    assert main(["scan", "--no-save"], fetcher=FakeFetcher(universe_data()), clock=SCAN_CLOCK) == 0
    (tmp_path / "config.toml").unlink()
    # no config.toml -> pure defaults (24-symbol universe, none available from the fake)
    assert (
        main(
            ["scan", "--no-save", "--symbols", "AAA.TO"],
            fetcher=FakeFetcher(universe_data()),
            clock=SCAN_CLOCK,
        )
        == 0
    )


def test_update_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "update-cache") == 0
    assert "updated 4/4" in capsys.readouterr().out
    assert run(tmp_path, "update-cache", "--full", "--symbols", "AAA.TO,NOPE.TO") == 1
    assert "FAILED NOPE.TO" in capsys.readouterr().out
    assert run(tmp_path, "update-cache", fetcher=FakeFetcher({})) == 0  # stale cache is used
    assert "stale: 4" in capsys.readouterr().out
    assert run(tmp_path / "empty", "update-cache", fetcher=FakeFetcher({})) == 2


def test_evaluate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "evaluate", "--horizons", "5,20") == 0
    out = capsys.readouterr().out
    assert "hit_rate" in out and "golden_cross" in out
    assert (tmp_path / "reports" / "evaluate_2026-11-05.csv").exists()
    assert run(tmp_path / "x", "evaluate", "--offline") == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["scan", "--as-of", "2026-13-01"],
        ["scan", "--symbols", ","],
        ["evaluate", "--horizons", "a"],
        ["evaluate", "--horizons", "0"],
        ["scan", "--review", "--no-review"],
        [],
    ],
)
def test_argument_validation(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(argv)
    assert e.value.code == 2


def test_symbols_are_normalised() -> None:
    a = build_parser().parse_args(["scan", "--symbols", " ry.to, td.to ,"])
    assert a.symbols == ["RY.TO", "TD.TO"]


def test_verbose_logging(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "scan", "-v", "--no-save", "--symbols", "AAA.TO") == 0
