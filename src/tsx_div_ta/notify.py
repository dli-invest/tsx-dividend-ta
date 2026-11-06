"""Post the daily report to a Discord channel through a webhook (stdlib only).

The message is one embed (market backdrop, top bullish/bearish, new signals, dividend alerts,
failures, AI review) plus the full ``report.txt`` and ``signals.csv`` as attachments.

Embed limits (Discord): title 256, description 4096, 25 fields, field name 256 / value 1024,
footer 2048, and 6000 characters in total. Everything is truncated to fit.

The webhook URL is a secret: it is read from an environment variable and never logged or
included in error messages.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsx_div_ta import DISCLAIMER, __version__
from tsx_div_ta.pipeline import ScanResult, SymbolSummary

log = logging.getLogger(__name__)

USER_AGENT = f"tsx-div-ta/{__version__} (+https://github.com/dli-invest/tsx-dividend-ta)"
COLORS = {"BULL": 0x2ECC71, "BEAR": 0xE74C3C}
DEFAULT_COLOR = 0x95A5A6
TITLE_MAX, DESC_MAX, FIELD_NAME_MAX, FIELD_VALUE_MAX, FOOTER_MAX = 256, 4096, 256, 1024, 2048
EMBED_TOTAL_MAX = 6000
MAX_FIELDS = 25
MAX_ATTEMPTS = 3
WEBHOOK_HOSTS = ("discord.com", "discordapp.com")

Opener = Callable[[urllib.request.Request, float], Any]


class NotifyError(RuntimeError):
    """Delivery failed. Messages never contain the webhook URL."""


def _trunc(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _row_line(r: SymbolSummary, new: bool = True) -> str:
    sigs = ", ".join(f"{s}{'*' if new and s in r.new_signals else ''}" for s in r.signals[:3])
    score = "-" if r.score is None else f"{r.score:+.1f}"
    y = "" if r.ttm_yield is None else f" · yld {r.ttm_yield:.2f}%"
    return f"**{r.symbol}** {score} {r.outlook or ''}{y}" + (f" · {sigs}" if sigs else "")


def build_embed(result: ScanResult, max_rows: int = 5) -> dict[str, Any]:
    m = result.market
    breadth = "n/a" if m.breadth is None else f"{m.breadth:.0%}"
    agree = {True: "confirms", False: "diverges", None: "n/a"}[m.confirmed]
    by_sym = {r.symbol: r for r in result.rows}
    ok = [r for r in result.rows if r.status == "OK"]
    desc = [
        f"Market regime ({m.benchmark}): **{m.regime}** · breadth above SMA200 {breadth} ({agree})",
        f"Analysed {len(ok)}/{len(result.rows)} symbols · data as of {result.as_of or '-'}",
    ]
    fields: list[dict[str, Any]] = []

    def add(name: str, lines: list[str]) -> None:
        if lines and len(fields) < MAX_FIELDS:
            value = _trunc("\n".join(lines), FIELD_VALUE_MAX)
            fields.append({"name": _trunc(name, FIELD_NAME_MAX), "value": value, "inline": False})

    add("Top bullish", [_row_line(by_sym[s]) for s in result.top_bullish[:max_rows]])
    add("Top bearish", [_row_line(by_sym[s]) for s in result.top_bearish[:max_rows]])
    new = [s for s in result.signals if s.name in by_sym[s.symbol].new_signals]
    add(
        f"New signals ({len(new)})",
        [f"{s.symbol} · {s.direction.lower()} · {s.name} ({s.strength:.2f})" for s in new[:15]],
    )
    alerts = [
        f"{s.symbol} · {s.name}: {s.reason}"
        for s in result.signals
        if s.name in ("dividend_cut", "yield_trap")
    ]
    add("Dividend alerts", alerts)
    stale = result.meta.get("stale") or []
    add(
        "Data problems",
        [f"{k}: {v}" for k, v in sorted(result.failures.items())]
        + ([f"stale data: {', '.join(stale)}"] if stale else []),
    )
    rv = result.review
    if rv and rv.get("text"):
        cost = rv.get("est_cost_usd")
        cost_txt = "" if cost is None else f", est. ${cost:.4f}"
        add(f"AI review ({rv.get('model')}{cost_txt})", [str(rv["text"])])

    embed: dict[str, Any] = {
        "title": _trunc(f"TSX dividend TA · {result.as_of or 'no data'}", TITLE_MAX),
        "description": _trunc("\n".join(desc), DESC_MAX),
        "color": COLORS.get(m.regime, DEFAULT_COLOR),
        "fields": fields,
        "footer": {"text": _trunc(DISCLAIMER, FOOTER_MAX)},
    }
    _fit_total(embed)
    return embed


def _embed_size(e: dict[str, Any]) -> int:
    return (
        len(e.get("title", ""))
        + len(e.get("description", ""))
        + len(e.get("footer", {}).get("text", ""))
        + sum(len(f["name"]) + len(f["value"]) for f in e.get("fields", []))
    )


def _fit_total(embed: dict[str, Any]) -> None:
    """Shrink the longest field values until the whole embed is within 6000 characters."""
    while _embed_size(embed) > EMBED_TOTAL_MAX:
        longest = max(embed["fields"], key=lambda f: len(f["value"]), default=None)
        excess = _embed_size(embed) - EMBED_TOTAL_MAX
        if longest is None or len(longest["value"]) <= 20:
            embed["footer"]["text"] = _trunc(embed["footer"]["text"], 100)
            break
        longest["value"] = _trunc(longest["value"], max(20, len(longest["value"]) - excess))


def build_payload(result: ScanResult) -> dict[str, Any]:
    return {
        "username": "TSX dividend TA",
        "embeds": [build_embed(result)],
        "allowed_mentions": {"parse": []},  # never ping anyone
    }


def _multipart(payload: dict[str, Any], files: list[Path]) -> tuple[bytes, str]:
    boundary = f"----tsxdivta{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def part(headers: str, body: bytes) -> None:
        parts.append(f"--{boundary}\r\n{headers}\r\n\r\n".encode() + body + b"\r\n")

    part(
        'Content-Disposition: form-data; name="payload_json"\r\nContent-Type: application/json',
        json.dumps(payload).encode(),
    )
    for i, f in enumerate(files):
        part(
            f'Content-Disposition: form-data; name="files[{i}]"; filename="{f.name}"\r\n'
            "Content-Type: application/octet-stream",
            f.read_bytes(),
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _default_opener(req: urllib.request.Request, timeout: float) -> Any:  # pragma: no cover
    return urllib.request.urlopen(req, timeout=timeout)


@dataclass
class DiscordNotifier:
    webhook_url: str
    timeout_s: float = 20.0
    opener: Opener = _default_opener
    sleep: Callable[[float], None] = time.sleep

    def _request(self, method: str, body: bytes | None, ctype: str | None) -> int:
        sep = "&" if "?" in self.webhook_url else "?"
        url = self.webhook_url + (f"{sep}wait=true" if method == "POST" else "")
        headers = {"User-Agent": USER_AGENT}
        if ctype:
            headers["Content-Type"] = ctype
        for attempt in range(1, MAX_ATTEMPTS + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with self.opener(req, self.timeout_s) as resp:
                    return int(resp.status)
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < MAX_ATTEMPTS:
                    try:
                        wait = float(json.loads(exc.read() or b"{}").get("retry_after", 1.0))
                    except (ValueError, AttributeError):
                        wait = 1.0
                    log.warning("Discord rate limited; retrying in %.1fs", wait)
                    self.sleep(min(wait, 30.0))
                    continue
                if exc.code >= 500 and attempt < MAX_ATTEMPTS:
                    self.sleep(2.0 * attempt)
                    continue
                raise NotifyError(f"Discord webhook returned HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < MAX_ATTEMPTS:
                    self.sleep(2.0 * attempt)
                    continue
                reason = getattr(exc, "reason", exc)
                raise NotifyError(f"cannot reach Discord: {type(exc).__name__}: {reason}") from None
        raise NotifyError("Discord webhook: retries exhausted")  # pragma: no cover

    def check(self) -> int:
        """Validate the webhook with a GET (posts nothing). Returns the HTTP status."""
        return self._request("GET", None, None)

    def send(self, result: ScanResult, attachments: list[Path] | None = None) -> int:
        files = [p for p in (attachments or []) if p.is_file()]
        body, ctype = _multipart(build_payload(result), files)
        status = self._request("POST", body, ctype)
        log.info("Discord report posted (HTTP %d, %d attachment(s))", status, len(files))
        return status


def notifier_from_env(env_name: str, environ: dict[str, str]) -> DiscordNotifier:
    url = (environ.get(env_name) or "").strip()
    if not url:
        raise NotifyError(f"environment variable {env_name} is not set")
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https"
        or not (host in WEBHOOK_HOSTS or host.endswith((".discord.com", ".discordapp.com")))
        or not parts.path.startswith("/api/webhooks/")
    ):
        raise NotifyError(f"{env_name} does not look like a Discord webhook URL")
    return DiscordNotifier(url)
