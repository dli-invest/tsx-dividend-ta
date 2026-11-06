from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest

from tsx_div_ta.config import (
    COMPONENTS,
    REGIMES,
    AppConfig,
    ConfigError,
    DataConfig,
    IndicatorConfig,
    RegimeConfig,
    ReviewConfig,
    ScoringConfig,
    SignalConfig,
    UniverseConfig,
    config_from_dict,
    load_config,
)

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"


def test_defaults_load() -> None:
    cfg = AppConfig()
    assert cfg.universe.benchmark == "XIC.TO"
    assert cfg.indicators.sma_slow == 200
    assert cfg.data.close_buffer_time() == time(16, 30)
    assert set(cfg.scoring.weights) == set(REGIMES)
    for w in cfg.scoring.weights.values():
        assert set(w) == set(COMPONENTS)
    assert cfg.review.enabled is False


def test_project_config_file_matches_defaults() -> None:
    """The shipped config.toml must parse and agree with the code defaults."""
    cfg = load_config(PROJECT_CONFIG)
    d = AppConfig()
    assert cfg.universe == d.universe
    assert cfg.data == d.data
    assert cfg.indicators == d.indicators
    assert cfg.regime == d.regime
    assert cfg.signals == d.signals
    assert cfg.scoring == d.scoring
    assert cfg.review == d.review
    assert cfg.notify == d.notify
    assert cfg.base_dir == PROJECT_CONFIG.parent


def test_load_none_returns_defaults() -> None:
    assert load_config(None).universe == UniverseConfig()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text("[data\nx=1")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(p)


def test_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text('[data]\ncache_dir = "mycache"\n')
    cfg = load_config(p)
    assert cfg.path(cfg.data.cache_dir) == tmp_path / "mycache"
    assert cfg.path("/abs/x") == Path("/abs/x")


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"bogus": {}}, "Unknown config section"),
        ({"data": {"histroy_years": 5}}, "Unknown key"),
        ({"data": []}, "must be a table"),
        ({"data": {"history_years": "5"}}, "must be an integer"),
        ({"data": {"history_years": True}}, "must be an integer"),
        ({"data": {"history_years": 0}}, "history_years"),
        ({"data": {"batch_size": 0}}, "batch_size"),
        ({"data": {"batch_delay_s": -1}}, "batch_delay_s"),
        ({"data": {"max_retries": 0}}, "max_retries"),
        ({"data": {"backoff_base_s": -0.5}}, "backoff_base_s"),
        ({"data": {"overlap_days": 0}}, "overlap_days"),
        ({"data": {"min_bars": 0}}, "min_bars"),
        ({"data": {"market_close_buffer": "25:00"}}, "HH:MM"),
        ({"data": {"market_close_buffer": "4pm"}}, "HH:MM"),
        ({"data": {"cache_dir": ""}}, "non-empty string"),
        ({"data": {"min_bars": 100}}, "must be >= indicators.sma_slow"),
        ({"universe": {"symbols": []}}, "must not be empty"),
        ({"universe": {"symbols": ["RY.TO", "RY.TO"]}}, "duplicates"),
        ({"universe": {"symbols": ["ry.to"]}}, "invalid symbol"),
        ({"universe": {"symbols": ["RY TO"]}}, "invalid symbol"),
        ({"universe": {"symbols": [1]}}, "list of strings"),
        ({"universe": {"benchmark": "bad sym"}}, "invalid symbol"),
        ({"indicators": {"sma_fast": 200, "sma_slow": 50}}, "sma_fast must be < sma_slow"),
        ({"indicators": {"macd_fast": 30}}, "macd_fast must be < macd_slow"),
        ({"indicators": {"rsi": -14}}, "must be > 0"),
        ({"indicators": {"rsi": 1}}, ">= 2"),
        ({"indicators": {"bb_std": "x"}}, "must be a number"),
        ({"regime": {"confirm_bars": 0}}, "confirm_bars"),
        ({"regime": {"breadth_bull": 0.3, "breadth_bear": 0.4}}, "breadth_bear < breadth_bull"),
        ({"signals": {"lookback_bars": 0}}, "lookback_bars"),
        ({"signals": {"rsi_pullback": 140}}, r"\[0, 100\]"),
        ({"signals": {"rsi_bear_ceiling_low": 65}}, "ceiling_low must be <="),
        ({"signals": {"breakout_rvol": 0}}, "breakout_rvol"),
        ({"signals": {"atr_proximity": 0}}, "atr_proximity"),
        ({"signals": {"counter_regime_scale": 1.5}}, "counter_regime_scale"),
        ({"signals": {"gate_counter_regime": "yes"}}, "true/false"),
        ({"signals": {"yield_percentile": 1.0}}, "yield_percentile"),
        ({"signals": {"yield_window_years": 0}}, "yield_window_years"),
        ({"signals": {"div_cut_pct": 0}}, "div_cut_pct"),
        ({"signals": {"div_cut_lookback_days": 0}}, "div_cut_lookback_days"),
        ({"signals": {"special_div_mult": 1.0}}, "special_div_mult"),
        ({"signals": {"vol_expansion_mult": 1.0}}, "vol_expansion_mult"),
        ({"signals": {"atr_stop_mult": 0}}, "atr_stop_mult"),
        ({"scoring": {"top_n": 0}}, "top_n"),
        ({"scoring": {"weights": {"SIDEWAYS": {}}}}, "unknown regime"),
        ({"scoring": {"weights": {"BULL": {"vibes": 1.0}}}}, "unknown component"),
        ({"scoring": {"weights": {"BULL": {"trend": -1}}}}, ">= 0"),
        ({"scoring": {"weights": {"BULL": {"trend": True}}}}, ">= 0"),
        ({"scoring": {"weights": {"BULL": []}}}, "must be a table"),
        ({"scoring": {"weights": {"BULL": dict.fromkeys(COMPONENTS, 0.0)}}}, "all be zero"),
        ({"review": {"max_tokens": 10}}, "max_tokens"),
        ({"review": {"max_turns": 0}}, "max_turns"),
        ({"review": {"max_symbols": 0}}, "max_symbols"),
        ({"review": {"timeout_s": 0}}, "timeout_s"),
        ({"review": {"model": ""}}, "non-empty string"),
        ({"notify": {"discord_webhook_env": "https://discord.com/x"}}, "environment variable name"),
        ({"notify": {"attach_files": "yes"}}, "true/false"),
    ],
)
def test_invalid_values_rejected(raw: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        config_from_dict(raw)


def test_partial_weights_merge_with_defaults() -> None:
    cfg = ScoringConfig(weights={"BULL": {"trend": 0.9}})
    assert cfg.weights["BULL"]["trend"] == 0.9
    assert cfg.weights["BULL"]["momentum"] == 0.20
    assert cfg.weights["BEAR"]["trend"] == 0.30


def test_weights_must_be_table() -> None:
    with pytest.raises(ConfigError, match="must be a table"):
        ScoringConfig(weights=[])  # type: ignore[arg-type]


def test_int_coerced_to_float() -> None:
    cfg = SignalConfig(rsi_pullback=40)
    assert isinstance(cfg.rsi_pullback, float)
    assert isinstance(IndicatorConfig(bb_std=2).bb_std, float)


def test_sections_construct_individually() -> None:
    assert DataConfig().history_years == 5
    assert RegimeConfig().confirm_bars == 3
    assert ReviewConfig().model.startswith("claude-")
