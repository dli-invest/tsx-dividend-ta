from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from conftest import SCAN_CLOCK, FakeFetcher, universe_data
from tsx_div_ta.__main__ import main
from tsx_div_ta.config import config_from_dict
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.store import DataStore
from tsx_div_ta.notify import (
    EMBED_TOTAL_MAX,
    FIELD_VALUE_MAX,
    USER_AGENT,
    DiscordNotifier,
    NotifyError,
    _multipart,
    build_embed,
    build_payload,
    notifier_from_env,
)
from tsx_div_ta.pipeline import ScanResult, run_scan

HOOK = "https://discord.com/api/webhooks/123/SECRET-TOKEN"


@pytest.fixture
def result(tmp_path: Path) -> ScanResult:
    cfg = config_from_dict(
        {
            "universe": {"symbols": ["AAA.TO", "BBB.TO", "CCC.TO", "ZZZ.TO"]},
            "data": {"batch_delay_s": 0.0, "backoff_base_s": 0.0},
        },
        base_dir=tmp_path,
    )
    store = DataStore(
        ParquetCache(tmp_path / "c"), FakeFetcher(universe_data()), SCAN_CLOCK, cfg.data
    )
    return run_scan(cfg, store, SCAN_CLOCK)


class Resp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> Resp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


class FakeOpener:
    """Scripted responses: an int is an HTTP status, an Exception is raised."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: float) -> Resp:
        self.requests.append(req)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return Resp(item)


def http_error(code: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(HOOK, code, "err", {}, io.BytesIO(body))  # type: ignore[arg-type]


def notifier(script: list[Any], url: str = HOOK) -> tuple[DiscordNotifier, FakeOpener, list[float]]:
    op, sleeps = FakeOpener(script), []
    return DiscordNotifier(url, opener=op, sleep=sleeps.append), op, sleeps


# =========================================================================== embed
def test_embed_content(result: ScanResult) -> None:
    e = build_embed(result)
    assert e["title"] == "TSX dividend TA · 2026-11-05"
    assert "Market regime (XIC.TO): **BULL**" in e["description"]
    assert e["color"] == 0x2ECC71
    names = [f["name"] for f in e["fields"]]
    assert names[0] == "Top bullish" and "Top bearish" in names
    assert any(n.startswith("New signals") for n in names)
    assert "Dividend alerts" in names  # BBB.TO cut its dividend
    assert "Data problems" in names  # ZZZ.TO failed
    assert "Not financial advice" in e["footer"]["text"]


def test_embed_without_new_signals_and_neutral_color(result: ScanResult) -> None:
    for r in result.rows:
        r.new_signals = []
    object.__setattr__(result.market, "regime", "NEUTRAL")
    e = build_embed(result)
    assert e["color"] == 0x95A5A6
    assert not any(f["name"].startswith("New signals") for f in e["fields"])


def test_embed_respects_discord_limits(result: ScanResult) -> None:
    result.failures = {f"F{i:03d}.TO": "x" * 200 for i in range(60)}
    result.meta["stale"] = ["S1.TO"]
    result.review = {"model": "claude-haiku-5-5", "text": "word " * 2000, "est_cost_usd": 0.001}
    e = build_embed(result)
    assert all(len(f["value"]) <= FIELD_VALUE_MAX for f in e["fields"])
    total = (
        len(e["title"])
        + len(e["description"])
        + len(e["footer"]["text"])
        + sum(len(f["name"]) + len(f["value"]) for f in e["fields"])
    )
    assert total <= EMBED_TOTAL_MAX
    assert any(
        f["name"].startswith("AI review (claude-haiku-5-5, est. $0.0010)") for f in e["fields"]
    )


def test_payload_and_multipart(result: ScanResult, tmp_path: Path) -> None:
    p = build_payload(result)
    assert p["allowed_mentions"] == {"parse": []} and len(p["embeds"]) == 1
    f = tmp_path / "report.txt"
    f.write_text("hello report")
    body, ctype = _multipart(p, [f])
    boundary = ctype.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(
        f"--{boundary}--\r\n".encode()
    )
    assert b'name="payload_json"' in body and b'name="files[0]"; filename="report.txt"' in body
    assert b"hello report" in body


# =========================================================================== delivery
def test_send_posts_multipart_with_wait(result: ScanResult, tmp_path: Path) -> None:
    f = tmp_path / "signals.csv"
    f.write_text("a,b\n")
    n, op, _ = notifier([200])
    assert n.send(result, [f, tmp_path / "missing.txt"]) == 200
    req = op.requests[0]
    assert req.get_method() == "POST" and req.full_url == HOOK + "?wait=true"
    assert req.get_header("User-agent") == USER_AGENT
    assert req.get_header("Content-type").startswith("multipart/form-data")
    assert b'filename="signals.csv"' in req.data and b"missing.txt" not in req.data


def test_wait_param_appends_to_existing_query(result: ScanResult) -> None:
    n, op, _ = notifier([204], url=HOOK + "?thread_id=9")
    n.send(result)
    assert op.requests[0].full_url == HOOK + "?thread_id=9&wait=true"


def test_check_is_get_without_body() -> None:
    n, op, _ = notifier([200])
    assert n.check() == 200
    assert op.requests[0].get_method() == "GET" and op.requests[0].data is None
    assert op.requests[0].full_url == HOOK


def test_rate_limit_retry(result: ScanResult) -> None:
    n, op, sleeps = notifier([http_error(429, b'{"retry_after": 1.5}'), 204])
    assert n.send(result) == 204
    assert sleeps == [1.5] and len(op.requests) == 2


def test_rate_limit_bad_body_defaults(result: ScanResult) -> None:
    n, _, sleeps = notifier([http_error(429, b"not json"), 204])
    n.send(result)
    assert sleeps == [1.0]


def test_server_error_retried_then_fails_without_leaking_url(result: ScanResult) -> None:
    n, op, sleeps = notifier([http_error(502), http_error(502), http_error(502)])
    with pytest.raises(NotifyError) as e:
        n.send(result)
    assert "HTTP 502" in str(e.value) and "SECRET-TOKEN" not in str(e.value)
    assert len(op.requests) == 3 and sleeps == [2.0, 4.0]


def test_client_error_not_retried(result: ScanResult) -> None:
    n, op, _ = notifier([http_error(404)])
    with pytest.raises(NotifyError, match="HTTP 404"):
        n.send(result)
    assert len(op.requests) == 1


def test_network_error(result: ScanResult) -> None:
    errs = [urllib.error.URLError("dns failure")] * 2 + [TimeoutError("slow")]
    n, op, _ = notifier(errs)
    with pytest.raises(NotifyError) as e:
        n.send(result)
    assert "cannot reach Discord" in str(e.value) and "SECRET-TOKEN" not in str(e.value)
    assert len(op.requests) == 3


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (HOOK, True),
        ("https://canary.discord.com/api/webhooks/1/x", True),
        ("https://discordapp.com/api/webhooks/1/x", True),
        ("http://discord.com/api/webhooks/1/x", False),
        ("https://evil.example.com/api/webhooks/1/x", False),
        ("https://discord.com.evil.io/api/webhooks/1/x", False),
        ("https://discord.com/not-a-webhook", False),
        ("", False),
    ],
)
def test_notifier_from_env_validation(url: str, ok: bool) -> None:
    env = {"HOOK": url}
    if ok:
        assert notifier_from_env("HOOK", env).webhook_url == url
    else:
        with pytest.raises(NotifyError) as e:
            notifier_from_env("HOOK", env)
        assert not url or url not in str(e.value)  # the URL (a secret) is never echoed


# =========================================================================== CLI
def _cfg(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text('[universe]\nsymbols = ["AAA.TO", "BBB.TO"]\n[data]\nbatch_delay_s = 0.0\n')
    return p


class RecordingNotifier:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[ScanResult, list[Path]]] = []
        self.checked = 0

    def send(self, result: ScanResult, files: list[Path]) -> int:
        if self.fail:
            raise NotifyError("Discord webhook returned HTTP 401")
        self.sent.append((result, files))
        return 200

    def check(self) -> int:
        self.checked += 1
        if self.fail:
            raise NotifyError("Discord webhook returned HTTP 401")
        return 200


def test_cli_scan_notify_attaches_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rn = RecordingNotifier()
    code = main(
        ["scan", "--config", str(_cfg(tmp_path)), "--notify"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        notifier=rn,
    )
    assert code == 0 and len(rn.sent) == 1
    assert [f.name for f in rn.sent[0][1]] == ["report.txt", "signals.csv"]
    assert (tmp_path / "state" / "last_run.json").exists()


def test_cli_notify_failure_exit3_and_state_not_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["scan", "--config", str(_cfg(tmp_path)), "--notify"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
        notifier=RecordingNotifier(fail=True),
    )
    assert code == 3
    assert not (tmp_path / "state" / "last_run.json").exists()  # signals stay NEW for next run
    assert "Discord notification failed" in capsys.readouterr().err


def test_cli_notify_missing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    code = main(
        ["scan", "--config", str(_cfg(tmp_path)), "--notify", "--no-save"],
        fetcher=FakeFetcher(universe_data()),
        clock=SCAN_CLOCK,
    )
    assert code == 3
    assert "DISCORD_WEBHOOK_URL is not set" in capsys.readouterr().err


def test_cli_check_webhook(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rn = RecordingNotifier()
    assert main(["check-webhook", "--config", str(_cfg(tmp_path))], notifier=rn) == 0
    assert rn.checked == 1 and "nothing was posted" in capsys.readouterr().out
    assert (
        main(
            ["check-webhook", "--config", str(_cfg(tmp_path))],
            notifier=RecordingNotifier(fail=True),
        )
        == 3
    )
    assert "HTTP 401" in capsys.readouterr().err


def test_cli_check_webhook_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.com/hook")
    assert main(["check-webhook", "--config", str(_cfg(tmp_path))]) == 3
    assert "does not look like a Discord webhook" in capsys.readouterr().err


def test_payload_json_is_valid(result: ScanResult) -> None:
    assert json.loads(json.dumps(build_payload(result)))["username"] == "TSX dividend TA"
