"""Per-symbol Parquet cache with atomic writes and corrupt-file recovery."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def safe_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "_", symbol)


class ParquetCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, symbol: str) -> Path:
        return self.directory / f"{safe_name(symbol)}.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self.path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p, use_threads=False)  # tiny files: no thread pool
        except Exception as exc:  # corrupt/truncated file: recover by re-fetching
            log.warning("cache file %s unreadable (%s); deleting and re-fetching", p, exc)
            p.unlink(missing_ok=True)
            return None
        df.index = pd.DatetimeIndex(df.index)
        return df

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        p = self.path(symbol)
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        os.replace(tmp, p)
