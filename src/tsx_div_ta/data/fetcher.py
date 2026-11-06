"""Market-data fetchers. ``YFinanceFetcher`` wraps the unofficial Yahoo Finance API politely:
small batches, delays, bounded retries with exponential backoff + jitter, no threads."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import pandas as pd

log = logging.getLogger(__name__)

Downloader = Callable[[list[str], date], pd.DataFrame]

COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}
RATE_LIMIT_ERRORS = {"YFRateLimitError"}


@dataclass
class FetchResult:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class Fetcher(Protocol):
    def fetch(self, symbols: Sequence[str], start: date) -> FetchResult:
        """Daily bars from ``start`` (inclusive) to the latest available, per symbol."""
        ...


def _standardize_one(sub: pd.DataFrame) -> pd.DataFrame | None:
    sub = sub.rename(columns=COLUMN_MAP)
    keep = [c for c in COLUMN_MAP.values() if c in sub.columns]
    sub = sub[keep]
    if "close" not in sub.columns:
        return None
    sub = sub.dropna(subset=["close"])
    return sub if not sub.empty else None


def standardize(raw: pd.DataFrame | None, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split a ``yf.download`` frame into per-symbol frames with lowercase column names.

    Handles both column layouts (ticker/field and field/ticker) and single-level frames."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw
        if "Close" in set(frame.columns.get_level_values(0)):
            frame = frame.swaplevel(0, 1, axis=1)
        tickers = set(frame.columns.get_level_values(0))
        for sym in symbols:
            if sym in tickers:
                sub = _standardize_one(frame[sym].copy())
                if sub is not None:
                    out[sym] = sub
    elif len(symbols) == 1:
        sub = _standardize_one(raw.copy())
        if sub is not None:
            out[symbols[0]] = sub
    return out


def _yf_download(symbols: list[str], start: date) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf  # lazy: offline runs never import yfinance

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return yf.download(
        symbols,
        start=start.isoformat(),
        interval="1d",
        auto_adjust=False,  # explicit: the default has changed across yfinance versions
        actions=True,
        group_by="ticker",
        threads=False,
        progress=False,
    )


class YFinanceFetcher:
    def __init__(
        self,
        batch_size: int = 10,
        batch_delay_s: float = 1.0,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
        downloader: Downloader | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.batch_delay_s = batch_delay_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._download = downloader or _yf_download
        self._sleep = sleep
        self._rng = rng or random.Random()

    def backoff(self, attempt: int, rate_limited: bool) -> float:
        """Delay before retry number ``attempt + 1`` (attempt is 0-based)."""
        delay = self.backoff_base_s * (2**attempt) + self._rng.uniform(0, self.backoff_base_s)
        return delay * (3 if rate_limited else 1)

    def fetch(self, symbols: Sequence[str], start: date) -> FetchResult:
        result = FetchResult()
        syms = list(dict.fromkeys(symbols))
        batches = [syms[i : i + self.batch_size] for i in range(0, len(syms), self.batch_size)]
        for n, batch in enumerate(batches):
            if n > 0 and self.batch_delay_s > 0:
                self._sleep(self.batch_delay_s)
            frames, err = self._fetch_batch(batch, start)
            result.frames.update(frames)
            for sym in batch:
                if sym not in frames:
                    result.errors[sym] = err or "no data returned (invalid or delisted symbol?)"
        return result

    def _fetch_batch(
        self, batch: list[str], start: date
    ) -> tuple[dict[str, pd.DataFrame], str | None]:
        last_err: str | None = None
        for attempt in range(self.max_retries):
            rate_limited = False
            try:
                frames = standardize(self._download(batch, start), batch)
                if frames:
                    return frames, None
                last_err = "empty response"
            except Exception as exc:  # unofficial API: any library/network error is retryable
                rate_limited = type(exc).__name__ in RATE_LIMIT_ERRORS
                last_err = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "download failed for %s (attempt %d/%d): %s",
                    batch,
                    attempt + 1,
                    self.max_retries,
                    last_err,
                )
            if attempt < self.max_retries - 1:
                self._sleep(self.backoff(attempt, rate_limited))
        return {}, last_err
