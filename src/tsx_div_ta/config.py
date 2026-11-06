"""Configuration: TOML file -> validated, frozen dataclasses.

Every key is optional (defaults below mirror ``config.toml``). Unknown sections or keys are
rejected so that a typo can never silently change the bot's behaviour.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from datetime import time
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised for any invalid configuration value."""


REGIMES: tuple[str, ...] = ("BULL", "BEAR", "NEUTRAL")
COMPONENTS: tuple[str, ...] = (
    "trend",
    "momentum",
    "relative_strength",
    "volume",
    "dividend",
    "risk",
    "signals",
)

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "RY.TO",
    "TD.TO",
    "BNS.TO",
    "BMO.TO",
    "CM.TO",
    "NA.TO",
    "MFC.TO",
    "SLF.TO",
    "GWO.TO",
    "POW.TO",
    "ENB.TO",
    "TRP.TO",
    "PPL.TO",
    "FTS.TO",
    "EMA.TO",
    "H.TO",
    "CU.TO",
    "BCE.TO",
    "T.TO",
    "RCI-B.TO",
    "CNQ.TO",
    "SU.TO",
    "REI-UN.TO",
    "CAR-UN.TO",
)

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL": {
        "trend": 0.25,
        "momentum": 0.20,
        "relative_strength": 0.15,
        "volume": 0.10,
        "dividend": 0.10,
        "risk": 0.10,
        "signals": 0.10,
    },
    "BEAR": {
        "trend": 0.30,
        "momentum": 0.10,
        "relative_strength": 0.15,
        "volume": 0.05,
        "dividend": 0.10,
        "risk": 0.20,
        "signals": 0.10,
    },
    "NEUTRAL": {
        "trend": 0.25,
        "momentum": 0.15,
        "relative_strength": 0.15,
        "volume": 0.10,
        "dividend": 0.15,
        "risk": 0.10,
        "signals": 0.10,
    },
}

_SYMBOL_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-]*$")


def _check_types(obj: Any) -> None:
    """Validate (and coerce int->float) each field against the type of its default value."""
    for f in fields(obj):
        default = f.default
        if default is MISSING:  # factory-built fields (e.g. weights) validate themselves
            continue
        value = getattr(obj, f.name)
        name = f"{type(obj).__name__}.{f.name}"
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{name} must be true/false, got {value!r}")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name} must be an integer, got {value!r}")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name} must be a number, got {value!r}")
            object.__setattr__(obj, f.name, float(value))
        elif isinstance(default, str):
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{name} must be a non-empty string, got {value!r}")
        elif isinstance(default, tuple) and (
            not isinstance(value, tuple) or not all(isinstance(v, str) for v in value)
        ):
            raise ConfigError(f"{name} must be a list of strings, got {value!r}")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _check_symbol(sym: str, where: str) -> None:
    _require(
        bool(_SYMBOL_RE.match(sym)), f"{where}: invalid symbol {sym!r} (expected e.g. 'RY.TO')"
    )


@dataclass(frozen=True)
class UniverseConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    benchmark: str = "XIC.TO"

    def __post_init__(self) -> None:
        _check_types(self)
        _require(len(self.symbols) > 0, "universe.symbols must not be empty")
        for s in self.symbols:
            _check_symbol(s, "universe.symbols")
        dupes = sorted({s for s in self.symbols if self.symbols.count(s) > 1})
        _require(not dupes, f"universe.symbols has duplicates: {dupes}")
        _check_symbol(self.benchmark, "universe.benchmark")


@dataclass(frozen=True)
class DataConfig:
    history_years: int = 5
    min_bars: int = 260
    batch_size: int = 10
    batch_delay_s: float = 1.0
    max_retries: int = 3
    backoff_base_s: float = 2.0
    overlap_days: int = 5
    market_close_buffer: str = "16:30"
    cache_dir: str = "cache"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    state_file: str = "state/last_run.json"

    def __post_init__(self) -> None:
        _check_types(self)
        _require(1 <= self.history_years <= 30, "data.history_years must be in 1..30")
        _require(self.min_bars >= 1, "data.min_bars must be >= 1")
        _require(self.batch_size >= 1, "data.batch_size must be >= 1")
        _require(self.batch_delay_s >= 0, "data.batch_delay_s must be >= 0")
        _require(self.max_retries >= 1, "data.max_retries must be >= 1")
        _require(self.backoff_base_s >= 0, "data.backoff_base_s must be >= 0")
        _require(self.overlap_days >= 1, "data.overlap_days must be >= 1")
        self.close_buffer_time()  # validates format

    def close_buffer_time(self) -> time:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", self.market_close_buffer)
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            raise ConfigError(
                f"data.market_close_buffer must be HH:MM, got {self.market_close_buffer!r}"
            )
        return time(int(m.group(1)), int(m.group(2)))


@dataclass(frozen=True)
class IndicatorConfig:
    sma_fast: int = 50
    sma_slow: int = 200
    ema: int = 21
    sma_slope_days: int = 20
    rsi: int = 14
    roc: int = 63
    atr: int = 14
    bb_length: int = 20
    bb_std: float = 2.0
    adx: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    supertrend_length: int = 10
    supertrend_mult: float = 3.0
    cmf: int = 20
    volume_sma: int = 20
    donchian: int = 55
    high_low_window: int = 252
    rs_window: int = 63

    def __post_init__(self) -> None:
        _check_types(self)
        for f in fields(self):
            _require(getattr(self, f.name) > 0, f"indicators.{f.name} must be > 0")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, int) and f.name != "bb_std":
                _require(v >= 2 or f.name == "sma_slope_days", f"indicators.{f.name} must be >= 2")
        _require(self.sma_fast < self.sma_slow, "indicators.sma_fast must be < sma_slow")
        _require(self.macd_fast < self.macd_slow, "indicators.macd_fast must be < macd_slow")


@dataclass(frozen=True)
class RegimeConfig:
    confirm_bars: int = 3
    breadth_bull: float = 0.60
    breadth_bear: float = 0.40

    def __post_init__(self) -> None:
        _check_types(self)
        _require(self.confirm_bars >= 1, "regime.confirm_bars must be >= 1")
        _require(
            0.0 <= self.breadth_bear < self.breadth_bull <= 1.0,
            "regime requires 0 <= breadth_bear < breadth_bull <= 1",
        )


@dataclass(frozen=True)
class SignalConfig:
    lookback_bars: int = 5
    rsi_pullback: float = 40.0
    rsi_bear_ceiling_low: float = 55.0
    rsi_bear_ceiling_high: float = 60.0
    rsi_oversold: float = 25.0
    adx_trend: float = 25.0
    adx_supertrend: float = 20.0
    breakout_rvol: float = 1.5
    breakout_rvol_bear: float = 2.0
    atr_proximity: float = 1.0
    counter_regime_scale: float = 0.6
    gate_counter_regime: bool = False
    yield_percentile: float = 0.80
    yield_window_years: int = 5
    div_cut_pct: float = 0.10
    div_cut_lookback_days: int = 365
    special_div_mult: float = 2.0
    vol_expansion_mult: float = 1.5
    atr_stop_mult: float = 2.0

    def __post_init__(self) -> None:
        _check_types(self)
        _require(self.lookback_bars >= 1, "signals.lookback_bars must be >= 1")
        for name in (
            "rsi_pullback",
            "rsi_bear_ceiling_low",
            "rsi_bear_ceiling_high",
            "rsi_oversold",
            "adx_trend",
            "adx_supertrend",
        ):
            v = getattr(self, name)
            _require(0.0 <= v <= 100.0, f"signals.{name} must be in [0, 100]")
        _require(
            self.rsi_bear_ceiling_low <= self.rsi_bear_ceiling_high,
            "signals.rsi_bear_ceiling_low must be <= rsi_bear_ceiling_high",
        )
        _require(
            self.breakout_rvol > 0 and self.breakout_rvol_bear > 0, "breakout_rvol must be > 0"
        )
        _require(self.atr_proximity > 0, "signals.atr_proximity must be > 0")
        _require(0.0 <= self.counter_regime_scale <= 1.0, "signals.counter_regime_scale in [0, 1]")
        _require(0.0 < self.yield_percentile < 1.0, "signals.yield_percentile must be in (0, 1)")
        _require(self.yield_window_years >= 1, "signals.yield_window_years must be >= 1")
        _require(0.0 < self.div_cut_pct < 1.0, "signals.div_cut_pct must be in (0, 1)")
        _require(self.div_cut_lookback_days >= 1, "signals.div_cut_lookback_days must be >= 1")
        _require(self.special_div_mult > 1.0, "signals.special_div_mult must be > 1")
        _require(self.vol_expansion_mult > 1.0, "signals.vol_expansion_mult must be > 1")
        _require(self.atr_stop_mult > 0, "signals.atr_stop_mult must be > 0")


@dataclass(frozen=True)
class ScoringConfig:
    top_n: int = 5
    weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
    )

    def __post_init__(self) -> None:
        _check_types(self)
        _require(self.top_n >= 1, "scoring.top_n must be >= 1")
        _require(isinstance(self.weights, dict), "scoring.weights must be a table")
        unknown = set(self.weights) - set(REGIMES)
        _require(not unknown, f"scoring.weights: unknown regime(s) {sorted(unknown)}")
        merged: dict[str, dict[str, float]] = {}
        for regime in REGIMES:
            given = self.weights.get(regime, {})
            _require(isinstance(given, dict), f"scoring.weights.{regime} must be a table")
            bad = set(given) - set(COMPONENTS)
            _require(not bad, f"scoring.weights.{regime}: unknown component(s) {sorted(bad)}")
            w = dict(DEFAULT_WEIGHTS[regime])
            for k, v in given.items():
                _require(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0,
                    f"scoring.weights.{regime}.{k} must be a number >= 0",
                )
                w[k] = float(v)
            _require(sum(w.values()) > 0, f"scoring.weights.{regime} must not all be zero")
            merged[regime] = w
        object.__setattr__(self, "weights", merged)


@dataclass(frozen=True)
class ReviewConfig:
    enabled: bool = False
    model: str = "claude-haiku-5-5"
    max_tokens: int = 2000  # thinking tokens count toward this cap
    max_turns: int = 4
    max_symbols: int = 12
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        _check_types(self)
        _require(64 <= self.max_tokens <= 8000, "review.max_tokens must be in 64..8000")
        _require(1 <= self.max_turns <= 10, "review.max_turns must be in 1..10")
        _require(1 <= self.max_symbols <= 100, "review.max_symbols must be in 1..100")
        _require(self.timeout_s > 0, "review.timeout_s must be > 0")


@dataclass(frozen=True)
class AppConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    base_dir: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        need = self.indicators.sma_slow + self.indicators.sma_slope_days
        _require(
            self.data.min_bars >= need,
            f"data.min_bars ({self.data.min_bars}) must be >= indicators.sma_slow + "
            f"sma_slope_days ({need}) so the regime can be computed",
        )

    def path(self, relative: str) -> Path:
        """Resolve a configured path relative to the config file's directory."""
        p = Path(relative).expanduser()
        return p if p.is_absolute() else self.base_dir / p


_SECTIONS: dict[str, type] = {
    "universe": UniverseConfig,
    "data": DataConfig,
    "indicators": IndicatorConfig,
    "regime": RegimeConfig,
    "signals": SignalConfig,
    "scoring": ScoringConfig,
    "review": ReviewConfig,
}


def _build(cls: type, raw: Any, name: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{name}]: {unknown}")
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()}
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any], base_dir: Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig` from an already-parsed TOML dict."""
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"Unknown config section(s): {unknown}")
    parts = {name: _build(cls, raw[name], name) for name, cls in _SECTIONS.items() if name in raw}
    return AppConfig(**parts, base_dir=base_dir or Path.cwd())


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from ``path``; ``None`` returns pure defaults rooted at the CWD."""
    if path is None:
        return AppConfig()
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {p}: {exc}") from exc
    return config_from_dict(raw, base_dir=p.parent)
