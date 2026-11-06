"""Injectable clock so tests never depend on the real time of day."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

TORONTO = ZoneInfo("America/Toronto")


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as a timezone-aware datetime in America/Toronto."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(TORONTO)


@dataclass(frozen=True)
class FixedClock:
    """Always returns ``fixed`` (naive datetimes are interpreted as Toronto local time)."""

    fixed: datetime

    def now(self) -> datetime:
        if self.fixed.tzinfo is None:
            return self.fixed.replace(tzinfo=TORONTO)
        return self.fixed.astimezone(TORONTO)
