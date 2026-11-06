"""End-to-end scan: load data -> benchmark regime -> per-symbol analysis (sequential, low
memory) -> breadth -> ranking. Pure orchestration; no I/O besides the data store."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from tsx_div_ta import __version__
from tsx_div_ta.analysis import benchmark_frame, build_analysis
from tsx_div_ta.clock import Clock
from tsx_div_ta.config import AppConfig
from tsx_div_ta.data.store import DataStore, LoadResult
from tsx_div_ta.indicators import TA_BACKEND, TA_VERSION
from tsx_div_ta.regime import MarketState, Regime, compute_breadth
from tsx_div_ta.scoring import ScoreCard, rank, score_frame
from tsx_div_ta.signals import Signal, emit_signals

log = logging.getLogger(__name__)

STATUS_OK = "OK"
STATUS_SHORT = "INSUFFICIENT_HISTORY"
STATUS_FAILED = "FAILED"
STATUS_ERROR = "ERROR"
BENCHMARK_CHART_BARS = 252


@dataclass
class SymbolSummary:
    symbol: str
    status: str
    date: str | None = None
    close: float | None = None
    stock_regime: str = Regime.UNKNOWN.value
    score: float | None = None
    outlook: str | None = None
    confidence: float | None = None
    drivers: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    ttm_yield: float | None = None
    fwd_yield: float | None = None
    yield_pct: float | None = None
    div_growth: float | None = None
    rsi: float | None = None
    adx: float | None = None
    atr_pct: float | None = None
    dist_sma_slow: float | None = None
    stop_long: float | None = None
    stop_short: float | None = None
    above_sma_slow: bool | None = None
    signals: list[str] = field(default_factory=list)
    new_signals: list[str] = field(default_factory=list)
    bars: int = 0
    stale: bool = False
    quality_issues: list[str] = field(default_factory=list)
    message: str | None = None
    scorecard: ScoreCard | None = None

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        d = asdict(self)
        d.pop("scorecard")
        if detail and self.scorecard is not None:
            d["factors"] = [asdict(f) for f in self.scorecard.factors]
        return d


@dataclass
class ScanResult:
    generated_at: str
    as_of: str | None
    market: MarketState
    rows: list[SymbolSummary]
    signals: list[Signal]
    failures: dict[str, str]
    top_bullish: list[str]
    top_bearish: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] | None = None
    # Last year of benchmark close/SMAs/regime for charts; not part of the JSON report.
    benchmark: pd.DataFrame | None = None

    @property
    def exit_code(self) -> int:
        """0 = all good, 1 = partial (some symbols failed), 2 = nothing analysable."""
        if not any(r.status in (STATUS_OK, STATUS_SHORT) for r in self.rows):
            return 2
        return 1 if self.failures else 0

    @property
    def signal_keys(self) -> list[str]:
        return [s.key for s in self.signals]


def _f(v: Any, nd: int = 4) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def truncate(frames: dict[str, pd.DataFrame], as_of: date | None) -> dict[str, pd.DataFrame]:
    if as_of is None:
        return frames
    ts = pd.Timestamp(as_of)
    return {s: f[f.index <= ts] for s, f in frames.items()}


def analyze_symbol(
    symbol: str,
    raw: pd.DataFrame,
    cfg: AppConfig,
    bench_close: pd.Series | None,
    market_labels: pd.Series | None,
    seen: set[str],
) -> tuple[SymbolSummary, list[Signal]]:
    if len(raw) < cfg.data.min_bars:
        last = raw.index[-1].strftime("%Y-%m-%d") if len(raw) else None
        return SymbolSummary(
            symbol,
            STATUS_SHORT,
            date=last,
            bars=len(raw),
            close=_f(raw["close"].iloc[-1]) if len(raw) else None,
            message=f"{len(raw)} bars < min_bars {cfg.data.min_bars}",
        ), []
    d = build_analysis(raw, cfg, bench_close, market_labels)
    sigs = emit_signals(symbol, d, cfg.signals)
    card = score_frame(d, sigs, cfg.scoring)
    r = d.iloc[-1]
    atr, raw_close, adj_close = float(r["atr"]), float(r["raw_close"]), float(r["close"])
    # ATR is computed on adjusted prices; express the stop in today's raw price terms.
    atr_raw = atr * raw_close / adj_close if adj_close else math.nan
    k = cfg.signals.atr_stop_mult
    sma_ok = math.isfinite(float(r["sma_slow"]))
    summary = SymbolSummary(
        symbol=symbol,
        status=STATUS_OK,
        date=d.index[-1].strftime("%Y-%m-%d"),
        close=_f(raw_close),
        stock_regime=str(r["stock_regime"]),
        score=card.score,
        outlook=card.outlook,
        confidence=card.confidence,
        drivers=card.drivers(),
        components=card.components,
        ttm_yield=_f(r["ttm_yield"]),
        fwd_yield=_f(r["fwd_yield"]),
        yield_pct=_f(r["yield_pct"]),
        div_growth=_f(r["div_growth"]),
        rsi=_f(r["rsi"], 2),
        adx=_f(r["adx"], 2),
        atr_pct=_f(r["atr_pct"]),
        dist_sma_slow=_f(r["dist_sma_slow"]),
        stop_long=_f(raw_close - k * atr_raw, 2),
        stop_short=_f(raw_close + k * atr_raw, 2),
        above_sma_slow=bool(r["close"] > r["sma_slow"]) if sma_ok else None,
        signals=[s.name for s in sigs],
        new_signals=[s.name for s in sigs if s.key not in seen],
        bars=len(d),
        scorecard=card,
    )
    return summary, sigs


def run_scan(
    cfg: AppConfig,
    store: DataStore,
    clock: Clock,
    symbols: Sequence[str] | None = None,
    offline: bool = False,
    as_of: date | None = None,
    seen: set[str] | None = None,
) -> ScanResult:
    universe = list(dict.fromkeys(symbols or cfg.universe.symbols))
    bench_sym = cfg.universe.benchmark
    load: LoadResult = store.load(
        [*universe, bench_sym] if bench_sym not in universe else universe, offline=offline
    )
    frames = truncate(load.frames, as_of)
    failures = dict(load.failures)
    seen = seen or set()

    bench = benchmark_frame(frames.get(bench_sym), cfg)
    bench_close = None if bench is None else bench["close"]
    labels = None if bench is None else bench["regime"].astype(object)
    if labels is None:
        log.warning(
            "benchmark %s unavailable: market regime UNKNOWN, relative strength disabled", bench_sym
        )

    rows: list[SymbolSummary] = []
    signals: list[Signal] = []
    for sym in universe:
        raw = frames.get(sym)
        if raw is None or raw.empty:
            rows.append(SymbolSummary(sym, STATUS_FAILED, message=failures.get(sym, "no data")))
            failures.setdefault(sym, "no data")
            continue
        try:
            summary, sigs = analyze_symbol(sym, raw, cfg, bench_close, labels, seen)
        except Exception as exc:  # one bad symbol must never abort the scan
            log.exception("%s: analysis failed", sym)
            failures[sym] = f"analysis error: {type(exc).__name__}: {exc}"
            rows.append(SymbolSummary(sym, STATUS_ERROR, message=failures[sym]))
            continue
        summary.stale = sym in load.stale
        q = load.quality.get(sym)
        summary.quality_issues = list(q.issues) if q else []
        rows.append(summary)
        signals.extend(sigs)

    ok_rows = [r for r in rows if r.status == STATUS_OK]
    breadth = compute_breadth(r.above_sma_slow for r in ok_rows)
    last_label = (
        str(labels.iloc[-1]) if labels is not None and len(labels) else Regime.UNKNOWN.value
    )
    m_date = labels.index[-1].strftime("%Y-%m-%d") if labels is not None and len(labels) else None
    market = MarketState.build(bench_sym, m_date, last_label, breadth, cfg.regime)
    bulls, bears = rank(
        [(r.symbol, r.score) for r in ok_rows if r.score is not None], cfg.scoring.top_n
    )
    rows.sort(key=lambda r: (r.score is None, -(r.score or 0.0), r.symbol))
    data_dates = [r.date for r in ok_rows if r.date]
    return ScanResult(
        generated_at=clock.now().isoformat(timespec="seconds"),
        as_of=max(data_dates) if data_dates else m_date,
        market=market,
        rows=rows,
        signals=sorted(signals, key=lambda s: (s.symbol, -s.strength, s.name)),
        failures=failures,
        top_bullish=bulls,
        top_bearish=bears,
        meta={
            "version": __version__,
            "ta_backend": f"{TA_BACKEND} {TA_VERSION}",
            "universe_size": len(universe),
            "stale": sorted(load.stale),
            "requested_as_of": as_of.isoformat() if as_of else None,
        },
        benchmark=None
        if bench is None
        else bench[["close", "sma_fast", "sma_slow", "regime"]].tail(BENCHMARK_CHART_BARS).copy(),
    )
