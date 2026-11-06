"""Daily chart (PNG) for the Discord report. Optional: needs ``pip install "tsx-div-ta[plot]"``.

Two stacked panels, each with its own single axis (never a dual axis):

1. Composite score per symbol as horizontal bars, sorted, coloured by polarity with a
   diverging blue (bullish, >= 20) / red (bearish, <= -20) pair and a neutral gray between.
   Bars grow from the zero baseline, so direction is also encoded by position.
2. Benchmark close with SMA50 / SMA200 over the last year. Price is drawn in a neutral ink
   so blue keeps meaning "bullish" across the figure.

Colours are dark-surface steps from the reference data-viz palette; the set was checked for
contrast on the surface (>= 3:1), normal-vision separation (OKLab dE >= 15) and colour-vision
deficiency separation (simulated dE >= 8, gray/red 7.7 with position + legend as secondary
encoding).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tsx_div_ta.pipeline import ScanResult

log = logging.getLogger(__name__)

SURFACE = "#1a1a19"
TEXT_PRIMARY = "#f0efec"
TEXT_SECONDARY = "#c3c2b7"
GRID = "#383835"
BULL = "#3987e5"
BEAR = "#e66767"
NEUTRAL = "#6b6a65"
PRICE = "#c3c2b7"
SMA_FAST = "#d95926"
SMA_SLOW = "#199e70"
OUTLOOK_THRESHOLD = 20.0
LABELLED_EXTREMES = 3
DPI = 110


def _polarity_color(score: float) -> str:
    if score >= OUTLOOK_THRESHOLD:
        return BULL
    if score <= -OUTLOOK_THRESHOLD:
        return BEAR
    return NEUTRAL


def spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Positions for end-labels: same order as ``values``, nudged apart so no two are closer
    than ``min_gap`` (labels that would collide are spread, never stacked on each other)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    prev: float | None = None
    for i in order:
        y = values[i] if prev is None else max(values[i], prev + min_gap)
        placed[i] = y
        prev = y
    if len(values) > 1:  # re-centre so the group stays around the true values
        shift = (sum(placed) - sum(values)) / len(values)
        placed = [p - shift for p in placed]
    return placed


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def render_chart(result: ScanResult, path: Path) -> Path | None:
    """Write the chart to ``path``. Returns None (and logs) if matplotlib is missing or there
    is nothing to plot; never raises for those cases."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless, no GUI toolkit
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        log.warning('chart skipped: matplotlib not installed (pip install "tsx-div-ta[plot]")')
        return None

    rows = sorted(
        (r for r in result.rows if r.status == "OK" and r.score is not None),
        key=lambda r: (r.score or 0.0, r.symbol),
    )
    bench = result.benchmark
    has_bench = bench is not None and not bench.empty and bench["close"].notna().any()
    if not rows and not has_bench:
        return None

    n_panels = 2 if (rows and has_bench) else 1
    score_h = max(3.0, 0.24 * len(rows) + 1.2) if rows else 0.0
    heights = [h for h in (score_h, 3.2 if has_bench else 0.0) if h]
    fig, grid = plt.subplots(
        n_panels,
        1,
        figsize=(8.0, sum(heights) + 0.9),
        gridspec_kw={"height_ratios": heights},
        facecolor=SURFACE,
        squeeze=False,
    )
    axes = [row[0] for row in grid]
    fig.suptitle(
        f"TSX dividend TA · {result.as_of or ''}",
        color=TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    panel = 0

    if rows:
        ax = axes[panel]
        panel += 1
        _style_axes(ax)
        scores = [float(r.score or 0.0) for r in rows]
        ys = list(range(len(rows)))
        # height < 1 leaves a surface gap between adjacent bars
        ax.barh(ys, scores, height=0.72, color=[_polarity_color(s) for s in scores], zorder=2)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, zorder=3)
        ax.set_yticks(ys, [r.symbol for r in rows])
        ax.set_xlim(-100, 100)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(
            "Composite score (−100 bearish … +100 bullish)", color=TEXT_SECONDARY, fontsize=9
        )
        label_idx = set(range(min(LABELLED_EXTREMES, len(rows)))) | set(
            range(max(0, len(rows) - LABELLED_EXTREMES), len(rows))
        )
        for i in sorted(label_idx):
            s = scores[i]
            ax.text(
                s + (2 if s >= 0 else -2),
                i,
                f"{s:+.0f}",
                va="center",
                ha="left" if s >= 0 else "right",
                color=TEXT_PRIMARY,
                fontsize=8,
            )
        m = result.market
        breadth = "n/a" if m.breadth is None else f"{m.breadth:.0%}"
        ax.set_title(
            f"Scores · market {m.regime} · breadth above SMA200 {breadth}",
            color=TEXT_SECONDARY,
            fontsize=10,
            loc="left",
        )
        ax.legend(
            handles=[
                Patch(color=BULL, label="Bullish (≥ +20)"),
                Patch(color=NEUTRAL, label="Neutral"),
                Patch(color=BEAR, label="Bearish (≤ −20)"),
            ],
            loc="lower right",
            fontsize=8,
            frameon=False,
            labelcolor=TEXT_PRIMARY,
        )

    if has_bench:
        assert bench is not None
        ax = axes[panel]
        _style_axes(ax)
        series = [
            ("close", PRICE, f"{result.market.benchmark} (adjusted)"),
            ("sma_fast", SMA_FAST, "SMA50"),
            ("sma_slow", SMA_SLOW, "SMA200"),
        ]
        ends: list[tuple[float, str]] = []
        for col, color, label in series:
            s = bench[col].dropna()
            if s.empty:
                continue
            ax.plot(s.index, s.to_numpy(), color=color, linewidth=1.6, label=label,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)  # fmt: skip
            ends.append((float(s.iloc[-1]), f"{label.split(' ')[0]} {s.iloc[-1]:.2f}"))
        lo, hi = ax.get_ylim()
        for y, text in zip(
            spread_labels([y for y, _ in ends], 0.07 * (hi - lo)), ends, strict=True
        ):
            ax.annotate(
                text[1],
                xy=(1.0, y),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=TEXT_PRIMARY,
                fontsize=8,
                annotation_clip=False,
            )
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.0)
        regime = str(bench["regime"].iloc[-1]) if "regime" in bench else result.market.regime
        ax.set_title(
            f"{result.market.benchmark} · 1 year · regime {regime}",
            color=TEXT_SECONDARY,
            fontsize=10,
            loc="left",
        )
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)

    fig.text(
        0.02,
        0.005,
        "Mechanical signals - not financial advice.",
        color=TEXT_SECONDARY,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.02, 0.93, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path
