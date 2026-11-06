"""Console table, CSV and JSON output. JSON is versioned and deterministic (sorted keys,
rounded floats, stable row order) so it can be diffed and golden-tested."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from tsx_div_ta import DISCLAIMER
from tsx_div_ta.config import COMPONENTS
from tsx_div_ta.pipeline import ScanResult, SymbolSummary

SCHEMA_VERSION = "1.1"

CSV_COLUMNS: tuple[str, ...] = (
    "symbol",
    "status",
    "date",
    "close",
    "stock_regime",
    "score",
    "outlook",
    "confidence",
    *(f"c_{c}" for c in COMPONENTS),
    "ttm_yield",
    "fwd_yield",
    "yield_pct",
    "div_growth",
    "rsi",
    "adx",
    "atr_pct",
    "dist_sma_slow",
    "stop_long",
    "stop_short",
    "signals",
    "new_signals",
    "drivers",
    "stale",
    "message",
)


def to_payload(result: ScanResult, detail: bool = True) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "disclaimer": DISCLAIMER,
        "generated_at": result.generated_at,
        "as_of": result.as_of,
        "market": {
            "benchmark": result.market.benchmark,
            "date": result.market.date,
            "regime": result.market.regime,
            "breadth": None if result.market.breadth is None else round(result.market.breadth, 4),
            "breadth_regime": result.market.breadth_regime,
            "breadth_confirms": result.market.confirmed,
        },
        "top_bullish": result.top_bullish,
        "top_bearish": result.top_bearish,
        "rows": [r.to_dict(detail=detail) for r in result.rows],
        "signals": [s.to_dict() for s in result.signals],
        "failures": dict(sorted(result.failures.items())),
        "meta": result.meta,
        "review": result.review,
    }


def to_json(result: ScanResult) -> str:
    return json.dumps(to_payload(result), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _csv_row(r: SymbolSummary) -> dict[str, Any]:
    d = r.to_dict(detail=False)
    row = {k: d.get(k) for k in CSV_COLUMNS if k in d}
    for c in COMPONENTS:
        row[f"c_{c}"] = r.components.get(c)
    row["signals"] = "; ".join(r.signals)
    row["new_signals"] = "; ".join(r.new_signals)
    row["drivers"] = "; ".join(r.drivers)
    return row


def to_csv(result: ScanResult) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    w.writeheader()
    for r in result.rows:
        w.writerow(_csv_row(r))
    return buf.getvalue()


def _fmt(v: float | None, spec: str = ".2f", none: str = "-") -> str:
    return none if v is None else format(v, spec)


def render_table(result: ScanResult) -> str:
    m = result.market
    breadth = "-" if m.breadth is None else f"{m.breadth:.0%}"
    agree = {True: "confirms", False: "DIVERGES", None: "n/a"}[m.confirmed]
    lines = [
        f"TSX dividend TA scan  |  data as of {result.as_of or '-'}  |  "
        f"generated {result.generated_at}",
        f"Market regime ({m.benchmark}): {m.regime}   breadth above SMA200: {breadth} "
        f"({m.breadth_regime}, {agree})",
        "",
    ]
    header = (
        f"{'Symbol':<11}{'Close':>9} {'Trend':<8}{'Score':>7} {'Outlook':<15}{'Conf':>5}"
        f"{'TTM%':>6}{'Fwd%':>6}{'YPct':>5}{'ATR%':>6}{'vs200':>7}  Signals (* = new)"
    )
    lines += [header, "-" * len(header)]
    for r in result.rows:
        if r.status != "OK":
            lines.append(f"{r.symbol:<11}{'':>9} {r.status:<30} {r.message or ''}")
            continue
        sig = ", ".join(f"{s}{'*' if s in r.new_signals else ''}" for s in r.signals[:3])
        if len(r.signals) > 3:
            sig += f" (+{len(r.signals) - 3})"
        stale = " [stale]" if r.stale else ""
        lines.append(
            f"{r.symbol:<11}{_fmt(r.close):>9} {r.stock_regime:<8}{_fmt(r.score, '+.1f'):>7} "
            f"{(r.outlook or '-'):<15}{_fmt(r.confidence, '.0%'):>5}{_fmt(r.ttm_yield, '.2f'):>6}"
            f"{_fmt(r.fwd_yield, '.2f'):>6}"
            f"{_fmt(None if r.yield_pct is None else r.yield_pct * 100, '.0f'):>5}"
            f"{_fmt(r.atr_pct, '.2f'):>6}{_fmt(r.dist_sma_slow, '+.1f'):>7}  {sig}{stale}"
        )
    lines += [
        "",
        f"Top bullish: {', '.join(result.top_bullish) or '-'}",
        f"Top bearish: {', '.join(result.top_bearish) or '-'}",
    ]
    top = [r for r in result.rows if r.symbol in set(result.top_bullish + result.top_bearish)]
    if top:
        lines.append("")
        lines.append("Score drivers (points):")
        for r in top:
            lines.append(f"  {r.symbol:<11}{'; '.join(r.drivers)}")
    if result.signals:
        lines += ["", "Signals:"]
        for s in result.signals:
            lines.append(
                f"  {s.symbol:<11}{s.date}  {s.direction:<8}{s.strength:>5.2f}  "
                f"{s.name:<18} {s.reason}"
            )
    if result.failures:
        lines += ["", "Failures:"] + [f"  {k}: {v}" for k, v in sorted(result.failures.items())]
    if result.review:
        rv = result.review
        cost = rv.get("est_cost_usd")
        cost_txt = "n/a" if cost is None else f"${cost:.4f}"
        lines += [
            "",
            f"AI review ({rv.get('model', '')}, {rv.get('status', '?')}, "
            f"{rv.get('turns', 0)} turn(s), est. {cost_txt}):",
        ]
        if rv.get("text"):
            lines += ["  " + ln for ln in str(rv["text"]).splitlines()]
        if rv.get("error"):
            lines.append(f"  ({rv['error']})")
    lines += ["", DISCLAIMER]
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_reports(result: ScanResult, reports_dir: Path) -> dict[str, Path]:
    """Write ``reports/<as_of>/signals.{csv,json}`` and ``report.txt``."""
    out = reports_dir / (result.as_of or "undated")
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "signals.json", "csv": out / "signals.csv", "txt": out / "report.txt"}
    _atomic_write(paths["json"], to_json(result))
    _atomic_write(paths["csv"], to_csv(result))
    _atomic_write(paths["txt"], render_table(result))
    return paths
