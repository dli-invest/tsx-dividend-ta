"""Incremental data updates: cache + fetcher + validation, one symbol's failure never aborts
the rest."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tsx_div_ta.clock import Clock
from tsx_div_ta.config import DataConfig
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.fetcher import Fetcher
from tsx_div_ta.data.validate import QualityReport, clean_ohlcv, drop_partial_bar

log = logging.getLogger(__name__)

ADJ_TOLERANCE = 1e-4


@dataclass
class LoadResult:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    quality: dict[str, QualityReport] = field(default_factory=dict)
    full_refreshed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)


def history_start(today: date, years: int) -> date:
    return (pd.Timestamp(today) - pd.DateOffset(years=years)).date()


def incremental_start(last: pd.Timestamp, overlap_days: int) -> date:
    return (pd.Timestamp(last) - pd.offsets.BDay(overlap_days)).date()


def needs_full_refresh(cached: pd.DataFrame, new: pd.DataFrame) -> str | None:
    """Return a reason string if the incremental window shows the cached adjusted history is
    no longer valid (new dividend/split, or Yahoo revised the adjustment factor)."""
    if new.empty:
        return None
    overlap = cached.index.intersection(new.index)
    if overlap.empty:
        return "no overlap between cache and new data"
    fresh = new.loc[new.index.difference(cached.index)]
    if (fresh["dividends"] > 0).any():
        return "new dividend"
    if (fresh["splits"].fillna(0) != 0).any():
        return "new split"
    c, n = cached.loc[overlap], new.loc[overlap]
    if not np.allclose(c["dividends"], n["dividends"]) or not np.allclose(
        c["splits"].fillna(0), n["splits"].fillna(0)
    ):
        return "dividend/split revised in overlap"
    f_old = (c["adj_close"] / c["close"]).to_numpy()
    f_new = (n["adj_close"] / n["close"]).to_numpy()
    if np.nanmax(np.abs(f_new / f_old - 1)) > ADJ_TOLERANCE:
        return "adjustment factor revised"
    return None


class DataStore:
    def __init__(
        self, cache: ParquetCache, fetcher: Fetcher | None, clock: Clock, cfg: DataConfig
    ) -> None:
        self.cache = cache
        self.fetcher = fetcher
        self.clock = clock
        self.cfg = cfg

    def _finalize(
        self, sym: str, raw: pd.DataFrame, result: LoadResult, hist0: date
    ) -> pd.DataFrame:
        df, rep = clean_ohlcv(raw, sym)
        df = drop_partial_bar(df, self.clock.now(), self.cfg.close_buffer_time())
        df = df[df.index >= pd.Timestamp(hist0)]
        result.quality[sym] = rep
        for issue in rep.issues:
            log.info("%s data quality: %s", sym, issue)
        return df

    def load(self, symbols: Sequence[str], offline: bool = False, full: bool = False) -> LoadResult:
        result = LoadResult()
        today = self.clock.now().date()
        hist0 = history_start(today, self.cfg.history_years)
        cached: dict[str, pd.DataFrame | None] = {s: self.cache.load(s) for s in symbols}

        if offline or self.fetcher is None:
            for s in symbols:
                c = cached[s]
                if c is None or c.empty:
                    result.failures[s] = "not in cache (offline mode)"
                    continue
                df = self._finalize(s, c, result, hist0)
                if df.empty:
                    result.failures[s] = "cached data empty after cleaning"
                else:
                    result.frames[s] = df
            return result

        full_syms = [s for s in symbols if full or cached[s] is None or cached[s].empty]  # type: ignore[union-attr]
        incr = [s for s in symbols if s not in full_syms]
        updated: dict[str, pd.DataFrame] = {}

        if incr:
            last_dates = [cached[s].index.max() for s in incr]  # type: ignore[union-attr]
            start = incremental_start(min(last_dates), self.cfg.overlap_days)
            res = self.fetcher.fetch(incr, start)
            for s in incr:
                old = cached[s]
                assert old is not None
                new = res.frames.get(s)
                if new is None:
                    log.warning(
                        "%s: refresh failed (%s); using cached data to %s",
                        s,
                        res.errors.get(s),
                        old.index.max().date(),
                    )
                    result.stale.append(s)
                    updated[s] = old
                    continue
                new_clean, _ = clean_ohlcv(new, s)
                new_clean = drop_partial_bar(
                    new_clean, self.clock.now(), self.cfg.close_buffer_time()
                )
                reason = needs_full_refresh(old, new_clean)
                if reason:
                    log.info("%s: full refresh (%s)", s, reason)
                    full_syms.append(s)
                    continue
                merged = pd.concat([old, new_clean])
                updated[s] = merged[~merged.index.duplicated(keep="last")].sort_index()

        if full_syms:
            res = self.fetcher.fetch(full_syms, hist0)
            for s in full_syms:
                new = res.frames.get(s)
                if new is not None:
                    updated[s] = new
                    result.full_refreshed.append(s)
                elif cached[s] is not None and not cached[s].empty:  # type: ignore[union-attr]
                    log.warning(
                        "%s: full refresh failed (%s); using cached data", s, res.errors.get(s)
                    )
                    result.stale.append(s)
                    updated[s] = cached[s]
                else:
                    result.failures[s] = res.errors.get(s, "no data returned")

        for s in symbols:
            if s not in updated:
                continue
            df = self._finalize(s, updated[s], result, hist0)
            if df.empty:
                result.failures[s] = "no usable rows after cleaning"
                continue
            self.cache.save(s, df)
            result.frames[s] = df
        return result
