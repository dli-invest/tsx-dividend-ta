from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from conftest import SCAN_CLOCK, FakeFetcher, universe_data
from tsx_div_ta.__main__ import main
from tsx_div_ta.config import ReviewConfig, config_from_dict
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.store import DataStore
from tsx_div_ta.pipeline import ScanResult, run_scan
from tsx_div_ta.report import render_table
from tsx_div_ta.review import (
    SYSTEM_PROMPT,
    TOOLS,
    Usage,
    build_digest,
    make_client,
    make_tool_handlers,
    review_report,
    run_review,
)

CFG = ReviewConfig()


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> ScanResult:
    tmp = tmp_path_factory.mktemp("rv")
    cfg = config_from_dict(
        {
            "universe": {"symbols": ["AAA.TO", "BBB.TO", "CCC.TO", "ZZZ.TO"]},
            "data": {"batch_delay_s": 0.0, "backoff_base_s": 0.0},
        },
        base_dir=tmp,
    )
    store = DataStore(ParquetCache(tmp / "c"), FakeFetcher(universe_data()), SCAN_CLOCK, cfg.data)
    return run_scan(cfg, store, SCAN_CLOCK)


def text(t: str) -> NS:
    return NS(type="text", text=t)


def tool_use(name: str, inp: dict[str, Any], id_: str = "tu_1") -> NS:
    return NS(type="tool_use", name=name, input=inp, id=id_)


def resp(
    content: list[NS], stop: str = "end_turn", inp: int = 1000, out: int = 200, **kw: Any
) -> NS:
    return NS(
        content=content,
        stop_reason=stop,
        usage=NS(
            input_tokens=inp,
            output_tokens=out,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        **kw,
    )


class FakeClient:
    """Records each messages.create call and replays scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(json.loads(json.dumps(kwargs, default=vars)))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# =========================================================================== digest & tools
def test_digest_is_compact_and_complete(result: ScanResult) -> None:
    d = build_digest(result, 12)
    assert d["market"]["regime"] == "BULL"
    assert {r["symbol"] for r in d["rows"]} == {"AAA.TO", "BBB.TO", "CCC.TO"}
    assert d["not_analysed"] == [
        {"symbol": "ZZZ.TO", "status": "FAILED", "message": "fake failure"}
    ]
    assert "factors" not in json.dumps(d)  # detail only via tool
    assert len(json.dumps(d)) < 4000


def test_digest_keeps_strongest_and_weakest(result: ScanResult) -> None:
    d = build_digest(result, 2)
    shown = [r["symbol"] for r in d["rows"]]
    ok = [r.symbol for r in result.rows if r.status == "OK"]
    assert shown == [ok[0], ok[-1]] and d["rows_shown"] == 2


def test_tool_handlers(result: ScanResult) -> None:
    h = make_tool_handlers(result)
    detail = json.loads(h["get_symbol_detail"]({"symbol": "bbb.to"}))
    assert detail["symbol"] == "BBB.TO" and "factors" in detail and detail["signal_details"]
    with pytest.raises(KeyError, match="unknown symbol"):
        h["get_symbol_detail"]({"symbol": "NOPE"})
    rule = json.loads(h["get_rule_definition"]({"name": "failed_rally"}))
    assert rule["direction"] == "BEARISH" and "RSI" in rule["definition"]
    with pytest.raises(KeyError, match="unknown rule"):
        h["get_rule_definition"]({"name": "moon"})


def test_tool_schemas_are_strict() -> None:
    for t in TOOLS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False
        assert t["input_schema"]["required"]


# =========================================================================== agent loop
def test_single_turn_review(result: ScanResult) -> None:
    client = FakeClient([resp([NS(type="thinking", thinking=""), text("Market is BULL.")])])
    r = run_review(result, CFG, client)
    assert r.status == "ok" and r.text == "Market is BULL." and r.turns == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == CFG.max_tokens
    assert call["output_config"] == {"effort": "low"}
    assert call["system"] == SYSTEM_PROMPT
    assert "temperature" not in call and "thinking" not in call
    assert call["messages"][-1]["role"] == "user"  # no assistant prefill
    assert call["tool_choice"] == {"type": "auto"}


def test_tool_round_trip_appends_history(result: ScanResult) -> None:
    client = FakeClient(
        [
            resp(
                [
                    text("checking"),
                    tool_use("get_symbol_detail", {"symbol": "BBB.TO"}, "a"),
                    tool_use("get_rule_definition", {"name": "nope"}, "b"),
                ],
                stop="tool_use",
            ),
            resp([text("BBB.TO cut its dividend.")]),
        ]
    )
    r = run_review(result, CFG, client)
    assert r.status == "ok" and r.turns == 2 and len(r.tool_calls) == 2
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant"  # full content echoed back unchanged
    results = second[2]["content"]
    assert [x["tool_use_id"] for x in results] == ["a", "b"]  # all results in ONE user message
    assert results[0]["is_error"] is False and "BBB.TO" in results[0]["content"]
    assert results[1]["is_error"] is True and "unknown rule" in results[1]["content"]
    assert r.usage.input_tokens == 2000 and r.usage.output_tokens == 400


def test_unknown_tool_name_is_error_result(result: ScanResult) -> None:
    client = FakeClient([resp([tool_use("rm_rf", {})], stop="tool_use"), resp([text("ok")])])
    run_review(result, CFG, client)
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_turn_cap_forces_text_on_last_turn(result: ScanResult) -> None:
    cfg = ReviewConfig(max_turns=2)
    client = FakeClient(
        [
            resp([tool_use("get_symbol_detail", {"symbol": "AAA.TO"})], stop="tool_use"),
            resp([text("final")]),
        ]
    )
    r = run_review(result, cfg, client)
    assert client.calls[0]["tool_choice"] == {"type": "auto"}
    assert client.calls[1]["tool_choice"] == {"type": "none"}
    assert r.text == "final" and r.turns == 2


def test_tool_use_on_last_turn_is_not_executed(result: ScanResult) -> None:
    client = FakeClient(
        [resp([text("partial"), tool_use("get_symbol_detail", {"symbol": "A"})], stop="tool_use")]
    )
    r = run_review(result, ReviewConfig(max_turns=1), client)
    assert r.tool_calls == [] and r.text == "partial"


def test_refusal(result: ScanResult) -> None:
    client = FakeClient([resp([], stop="refusal", stop_details=NS(category="cyber"))])
    r = run_review(result, CFG, client)
    assert r.status == "refused" and "cyber" in (r.error or "")


def test_max_tokens_truncation(result: ScanResult) -> None:
    r = run_review(result, CFG, FakeClient([resp([text("half")], stop="max_tokens")]))
    assert r.status == "truncated" and r.text.endswith("[review truncated at max_tokens]")


def test_empty_text_is_error(result: ScanResult) -> None:
    r = run_review(
        result, CFG, FakeClient([resp([NS(type="thinking", thinking="")], stop="max_tokens")])
    )
    assert r.status == "truncated"
    r2 = run_review(result, CFG, FakeClient([resp([NS(type="thinking", thinking="")])]))
    assert r2.status == "error" and "no text" in (r2.error or "")


# =========================================================================== cost & wrapper
def test_cost_estimate() -> None:
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd("claude-haiku-5-5") == pytest.approx(0.60)
    assert u.cost_usd("claude-opus-5-5") == pytest.approx(24.0)
    assert u.cost_usd("some-future-model") is None
    c = Usage(cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    assert c.cost_usd("claude-haiku-5-5") == pytest.approx(0.01 + 0.125)


def test_typical_review_costs_well_under_a_cent(result: ScanResult) -> None:
    client = FakeClient([resp([text("x")], inp=3000, out=600)])
    d = review_report(result, CFG, client)
    assert d["est_cost_usd"] is not None and d["est_cost_usd"] < 0.001


def test_review_report_never_raises(result: ScanResult) -> None:
    class APIConnectionError(Exception):
        pass

    d = review_report(result, CFG, FakeClient([APIConnectionError("offline")]))
    assert d["status"] == "error" and "APIConnectionError" in d["error"]
    assert json.dumps(d)


def test_review_report_ok_dict(result: ScanResult) -> None:
    d = review_report(result, CFG, FakeClient([resp([text("fine")])]))
    assert d["status"] == "ok" and d["text"] == "fine" and d["usage"]["input_tokens"] == 1000


def test_missing_package(result: ScanResult, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    d = review_report(result, CFG, None)
    assert d["status"] == "unavailable" and "pip install" in d["error"]


def test_client_construction_error(result: ScanResult, monkeypatch: pytest.MonkeyPatch) -> None:
    import tsx_div_ta.review as rv

    def boom(cfg: ReviewConfig) -> None:
        raise RuntimeError("no credentials")

    monkeypatch.setattr(rv, "make_client", boom)
    assert review_report(result, CFG, None)["status"] == "unavailable"


def test_make_client_uses_configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "sk-test")
    c = make_client(ReviewConfig(api_key_env="MY_KEY", timeout_s=12.0))
    assert c.api_key == "sk-test" and c.max_retries == 2


def test_table_shows_review(result: ScanResult) -> None:
    result.review = review_report(result, CFG, FakeClient([resp([text("Breadth confirms.")])]))
    t = render_table(result)
    assert "AI review (claude-haiku-5-5, ok, 1 turn(s), est. $0.0002)" in t
    assert "  Breadth confirms." in t
    result.review = {"model": "m", "status": "error", "error": "boom", "est_cost_usd": None}
    assert "(boom)" in render_table(result) and "est. n/a" in render_table(result)
    result.review = None


# =========================================================================== CLI
def test_cli_review_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfgp = tmp_path / "config.toml"
    cfgp.write_text('[universe]\nsymbols = ["AAA.TO", "BBB.TO"]\n[data]\nbatch_delay_s = 0.0\n')
    client = FakeClient([resp([text("Looks constructive.")])])
    code = main(
        ["scan", "--config", str(cfgp), "--review", "--format", "json"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        review_client=client,
    )
    assert code == 0
    p = json.loads(capsys.readouterr().out)
    assert p["review"]["text"] == "Looks constructive."
    saved = json.loads((tmp_path / "reports" / "2026-11-05" / "signals.json").read_text())
    assert saved["review"]["status"] == "ok"


def test_cli_review_enabled_in_config_and_no_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfgp = tmp_path / "config.toml"
    cfgp.write_text(
        '[universe]\nsymbols = ["AAA.TO"]\n[data]\nbatch_delay_s = 0.0\n[review]\nenabled = true\n'
    )
    client = FakeClient([resp([text("auto")])])
    main(
        ["scan", "--config", str(cfgp), "--no-save", "--format", "json"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        review_client=client,
    )
    assert json.loads(capsys.readouterr().out)["review"]["text"] == "auto"
    main(
        ["scan", "--config", str(cfgp), "--no-save", "--no-review", "--format", "json"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        review_client=client,
    )
    assert json.loads(capsys.readouterr().out)["review"] is None
