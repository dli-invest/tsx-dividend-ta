"""Remembers which signals were reported last run so new ones can be flagged ``NEW``."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)


def load_seen(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k) for k in data.get("signal_keys", [])}
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("state file %s unreadable (%s); treating all signals as new", path, exc)
        return set()


def save_seen(path: Path, keys: Iterable[str], as_of: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"as_of": as_of, "signal_keys": sorted(set(keys))}, indent=1), encoding="utf-8"
    )
    os.replace(tmp, path)
