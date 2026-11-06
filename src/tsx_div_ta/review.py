"""Optional AI review of the final scan report by Claude, kept deliberately cheap.

Cost controls:

* the model sees a compact JSON *digest* (market state + top rows), not the full report;
* it can pull more detail for at most a few symbols through two read-only tools;
* the agent loop is hard-capped at ``max_turns`` API calls, and on the last turn tools are
  disabled (``tool_choice: none``) so it must answer;
* default model ``claude-haiku-5-5`` at ``effort: low`` with a small ``max_tokens``.

Typical cost with the defaults is a fraction of a US cent per scan; the exact token usage and
an estimated USD cost are recorded in the report.

The review never affects signals or scores and never fails the scan: any problem (package
missing, no credentials, network/API error, refusal) is returned as a status in the result.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tsx_div_ta.config import ReviewConfig
from tsx_div_ta.pipeline import ScanResult, SymbolSummary
from tsx_div_ta.signals import RULES_BY_NAME

log = logging.getLogger(__name__)

# USD per million tokens (input, output) for prompts <= 100K tokens - see Anthropic pricing.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25
MAX_TOOL_RESULT_CHARS = 6000

SYSTEM_PROMPT = """\
You review the output of a mechanical technical-analysis scanner for core Canadian (TSX) \
dividend stocks. The scanner computes indicators, a market regime from a benchmark ETF, rule-\
based signals and a composite score in [-100, 100] built from weighted factors (each factor's \
contribution in points is available). You get a JSON digest of one scan.

Write a concise review for an investor who reads the scan once a day:
1. Market backdrop: the regime, whether breadth confirms it, and what that implies for how \
much weight bullish vs bearish signals deserve.
2. The most notable bullish and bearish setups, citing the symbols and the specific signals or \
score drivers behind them.
3. Red flags and contradictions worth a closer look: yield traps, dividend cuts, signals that \
fight the market regime, low confidence scores, stale or poor-quality data, failed symbols.
4. One line on what to watch next session.

Rules:
- Use only the data provided or returned by the tools. Do not invent prices, news, earnings or \
fundamentals, and say so when the data cannot answer something.
- This is not personalised financial advice: do not tell anyone to buy or sell, do not give \
price targets or position sizes.
- Call a tool only when a symbol's details would change your review; at most 3 tool calls.
- Plain text, short bullet points, under 250 words total."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_symbol_detail",
        "description": (
            "Full detail for one symbol from this scan: every score factor with its points, all "
            "signals with reasons and indicator values, dividend metrics, stops and data-quality "
            "issues. Use when the digest is not enough to judge a symbol."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Symbol, e.g. RY.TO"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_rule_definition",
        "description": "Exact definition of a signal rule by name (e.g. 'failed_rally').",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Signal rule name"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, u: Any) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            setattr(self, name, getattr(self, name) + int(getattr(u, name, 0) or 0))

    def cost_usd(self, model: str) -> float | None:
        price = PRICES_PER_MTOK.get(model)
        if price is None:
            return None
        pin, pout = price
        dollars = (
            self.input_tokens * pin
            + self.cache_read_input_tokens * pin * CACHE_READ_MULT
            + self.cache_creation_input_tokens * pin * CACHE_WRITE_MULT
            + self.output_tokens * pout
        ) / 1_000_000
        return round(dollars, 6)


@dataclass
class ReviewResult:
    status: str  # ok | truncated | refused | error | unavailable
    model: str
    text: str = ""
    turns: int = 0
    tool_calls: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "text": self.text,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "stop_reason": self.stop_reason,
            "usage": vars(self.usage).copy(),
            "est_cost_usd": self.usage.cost_usd(self.model),
            "error": self.error,
        }


# --------------------------------------------------------------------------- digest & tools
def _row_digest(r: SymbolSummary) -> dict[str, Any]:
    if r.status != "OK":
        return {"symbol": r.symbol, "status": r.status, "message": r.message}
    d: dict[str, Any] = {
        "symbol": r.symbol,
        "score": r.score,
        "outlook": r.outlook,
        "confidence": r.confidence,
        "trend": r.stock_regime,
        "drivers": r.drivers,
        "signals": r.signals,
        "new": r.new_signals,
        "ttm_yield_pct": r.ttm_yield,
        "fwd_yield_pct": r.fwd_yield,
        "yield_rank": r.yield_pct,
        "rsi": r.rsi,
        "vs_sma200_pct": r.dist_sma_slow,
    }
    if r.stale:
        d["stale"] = True
    if r.quality_issues:
        d["data_issues"] = len(r.quality_issues)
    return {k: v for k, v in d.items() if v not in (None, [], "")}


def build_digest(result: ScanResult, max_symbols: int) -> dict[str, Any]:
    """Compact view of the scan: all non-OK rows, plus the strongest and weakest OK rows."""
    ok = [r for r in result.rows if r.status == "OK" and r.score is not None]
    ranked = sorted(ok, key=lambda r: (-(r.score or 0.0), r.symbol))
    if len(ranked) <= max_symbols:
        chosen = ranked
    else:
        top = (max_symbols + 1) // 2
        chosen = ranked[:top] + ranked[-(max_symbols - top) :]
    m = result.market
    return {
        "as_of": result.as_of,
        "market": {
            "benchmark": m.benchmark,
            "regime": m.regime,
            "breadth_above_sma200": None if m.breadth is None else round(m.breadth, 3),
            "breadth_confirms": m.confirmed,
        },
        "universe": len(result.rows),
        "rows_shown": len(chosen),
        "rows": [_row_digest(r) for r in chosen],
        "not_analysed": [_row_digest(r) for r in result.rows if r.status != "OK"],
        "top_bullish": result.top_bullish,
        "top_bearish": result.top_bearish,
        "failures": result.failures,
    }


def make_tool_handlers(result: ScanResult) -> dict[str, Callable[[dict[str, Any]], str]]:
    rows = {r.symbol.upper(): r for r in result.rows}

    def symbol_detail(args: dict[str, Any]) -> str:
        sym = str(args.get("symbol", "")).strip().upper()
        r = rows.get(sym)
        if r is None:
            raise KeyError(f"unknown symbol {sym!r}; available: {sorted(rows)}")
        detail = r.to_dict(detail=True)
        detail["signal_details"] = [s.to_dict() for s in result.signals if s.symbol == r.symbol]
        return json.dumps(detail, sort_keys=True, default=str)

    def rule_definition(args: dict[str, Any]) -> str:
        name = str(args.get("name", "")).strip()
        rule = RULES_BY_NAME.get(name)
        if rule is None:
            raise KeyError(f"unknown rule {name!r}; available: {sorted(RULES_BY_NAME)}")
        doc = " ".join((rule.fn.__doc__ or "").split())
        return json.dumps(
            {
                "name": name,
                "direction": rule.direction.value,
                "playbook": rule.playbook.value,
                "kind": rule.kind.value,
                "base_strength": rule.base_strength,
                "definition": doc,
            }
        )

    return {"get_symbol_detail": symbol_detail, "get_rule_definition": rule_definition}


# --------------------------------------------------------------------------- client & loop
def make_client(cfg: ReviewConfig) -> Any:
    """Anthropic client. Credentials: the env var named in ``api_key_env`` if set, otherwise
    the SDK's own resolution (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`)."""
    import anthropic  # optional dependency: pip install "tsx-div-ta[ai]"

    key = os.environ.get(cfg.api_key_env) or None
    return anthropic.Anthropic(api_key=key, timeout=cfg.timeout_s, max_retries=2)


def _text_of(content: Any) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def run_review(result: ScanResult, cfg: ReviewConfig, client: Any) -> ReviewResult:
    digest = json.dumps(build_digest(result, cfg.max_symbols), sort_keys=True, default=str)
    handlers = make_tool_handlers(result)
    out = ReviewResult(status="ok", model=cfg.model)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Scan digest (JSON):\n{digest}\n\nWrite the review.",
        }
    ]
    for turn in range(1, cfg.max_turns + 1):
        last = turn == cfg.max_turns
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "none"} if last else {"type": "auto"},
            output_config={"effort": "low"},
            messages=messages,
        )
        out.turns = turn
        out.usage.add(resp.usage)
        out.stop_reason = resp.stop_reason
        if resp.stop_reason == "refusal":
            out.status = "refused"
            details = getattr(resp, "stop_details", None)
            out.error = f"model declined (category: {getattr(details, 'category', None)})"
            return out
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if resp.stop_reason == "tool_use" and tool_uses and not last:
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                out.tool_calls.append(f"{tu.name}({json.dumps(tu.input, sort_keys=True)})")
                handler = handlers.get(tu.name)
                try:
                    if handler is None:
                        raise KeyError(f"unknown tool {tu.name!r}")
                    content, is_error = handler(dict(tu.input))[:MAX_TOOL_RESULT_CHARS], False
                except (KeyError, TypeError, ValueError) as exc:
                    content, is_error = f"Error: {exc}", True
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})
            continue
        out.text = _text_of(resp.content)
        if resp.stop_reason == "max_tokens":
            out.status = "truncated"
            out.text = (out.text + "\n[review truncated at max_tokens]").strip()
        elif not out.text:
            out.status = "error"
            out.error = f"no text in final response (stop_reason={resp.stop_reason})"
        return out
    out.status = "error"  # pragma: no cover - loop always returns on the last turn
    return out  # pragma: no cover


def review_report(result: ScanResult, cfg: ReviewConfig, client: Any = None) -> dict[str, Any]:
    """Run the review; never raises. Returns a JSON-serialisable dict for the report."""
    try:
        client = client if client is not None else make_client(cfg)
    except ImportError:
        return ReviewResult(
            "unavailable",
            cfg.model,
            error='anthropic package not installed: pip install "tsx-div-ta[ai]"',
        ).to_dict()
    except Exception as exc:  # credential resolution etc.
        return ReviewResult(
            "unavailable", cfg.model, error=f"{type(exc).__name__}: {exc}"
        ).to_dict()
    try:
        res = run_review(result, cfg, client)
    except Exception as exc:  # API/network errors must never fail the scan
        log.warning("AI review failed: %s: %s", type(exc).__name__, exc)
        res = ReviewResult("error", cfg.model, error=f"{type(exc).__name__}: {exc}")
    d = res.to_dict()
    log.info(
        "AI review %s: %d turn(s), %s tokens in / %s out, est. $%s",
        d["status"],
        d["turns"],
        d["usage"]["input_tokens"],
        d["usage"]["output_tokens"],
        d["est_cost_usd"],
    )
    return d
