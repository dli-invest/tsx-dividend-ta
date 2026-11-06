"""Historical sanity check of each rule: forward returns after the rule fired.

Signals are produced by the exact same no-look-ahead code path as ``scan``; only the
*evaluation* looks forward (that is its purpose). Counter-regime scaling/gating is ignored -
every event is counted. This is descriptive statistics, not a backtest (no costs, overlapping
windows, survivorship bias in a hand-picked universe)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tsx_div_ta.analysis import benchmark_regime, build_analysis
from tsx_div_ta.config import AppConfig
from tsx_div_ta.signals import RULES, Direction, rule_events

EVAL_COLUMNS = ("rule", "direction", "horizon", "count", "hit_rate", "mean_pct", "median_pct")


def evaluate(
    frames: dict[str, pd.DataFrame], cfg: AppConfig, horizons: Sequence[int], symbols: Sequence[str]
) -> pd.DataFrame:
    bench_close, labels = benchmark_regime(frames.get(cfg.universe.benchmark), cfg)
    returns: dict[tuple[str, int], list[np.ndarray]] = {
        (r.name, h): [] for r in RULES for h in horizons
    }
    for sym in symbols:
        raw = frames.get(sym)
        if raw is None or len(raw) < cfg.data.min_bars:
            continue
        d = build_analysis(raw, cfg, bench_close, labels)
        ev = rule_events(d, cfg.signals)
        close = d["close"]
        for h in horizons:
            fwd = (close.shift(-h) / close - 1).to_numpy() * 100
            for r in RULES:
                mask = ev[r.name].to_numpy() & ~np.isnan(fwd)
                if mask.any():
                    returns[(r.name, h)].append(fwd[mask])
    rows = []
    for r in RULES:
        for h in horizons:
            parts = returns[(r.name, h)]
            vals = np.concatenate(parts) if parts else np.array([])
            hits = vals < 0 if r.direction is Direction.BEARISH else vals > 0
            rows.append(
                {
                    "rule": r.name,
                    "direction": r.direction.value,
                    "horizon": h,
                    "count": int(vals.size),
                    "hit_rate": round(float(hits.mean()), 4) if vals.size else None,
                    "mean_pct": round(float(vals.mean()), 3) if vals.size else None,
                    "median_pct": round(float(np.median(vals)), 3) if vals.size else None,
                }
            )
    return pd.DataFrame(rows, columns=list(EVAL_COLUMNS))
