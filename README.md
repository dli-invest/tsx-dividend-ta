# tsx-div-ta

A low-resource technical-analysis bot for **core Canadian (TSX) dividend stocks**, built to run in
WSL2. It downloads daily prices and dividends, works out the market regime (bull / bear / neutral),
computes indicators with **pandas-ta**, and produces explainable bullish and bearish signals plus a
detailed composite score. An optional AI review by Claude summarises each report for a fraction of a
cent.

> **Not financial advice.** Output is mechanical, informational and can be wrong. Data comes from
> Yahoo Finance via `yfinance` (unofficial, may be delayed or incorrect).

```
TSX dividend TA scan  |  data as of 2026-11-05  |  generated 2026-11-06T00:30:08-05:00
Market regime (XIC.TO): BULL   breadth above SMA200: 79% (BULL, confirms)

Symbol         Close Trend     Score Outlook         Conf  TTM%  Fwd% YPct  ATR%  vs200  Signals (* = new)
FTS.TO         85.79 BULL      +68.2 STRONG_BULLISH  100%  2.98  2.98    0  1.43  +11.3  golden_cross
MFC.TO         61.63 BULL      +37.2 BULLISH         100%  3.07  3.12    4  1.94  +11.8  breakout, macd_bull_cross
T.TO           13.19 BEAR       -7.9 NEUTRAL          95% 10.93  5.70   92  2.26  -13.1  yield_trap, dividend_cut
SLF.TO        102.83 BULL      -35.3 BEARISH         100%  3.66  3.73   11  2.40   +1.6  vol_expansion, relative_weakness, ...
```

## Measured footprint (WSL2, 16 cores, live Yahoo data, 24 stocks + benchmark)

| Run | Wall time | Peak RSS |
|---|---|---|
| Cold cache (5-year download of everything) | 11.7 s | 211 MB |
| Warm cache via `run.sh` (incremental download) | 9.1 s | 228 MB |
| `scan --offline` (compute only) | 3.6 s | 190 MB |

The bot uses one process and no worker threads (yfinance `threads=False`, single-threaded Parquet
I/O), runs under `nice -n 10`, and uses 0% CPU between scheduled runs. Most of the wall time is
polite network pacing. The cache for 25 symbols is about 1.6 MB.

## Install (WSL2 Ubuntu, Python 3.12)

The project targets **CPython 3.12**, which has security support until October 2028. Ubuntu 22.04
ships 3.10, so get 3.12 with [uv](https://docs.astral.sh/uv/). It installs into your home directory
and needs no sudo:

```bash
python3 -m venv ~/.uv-bootstrap && ~/.uv-bootstrap/bin/pip install uv
```

```bash
~/.uv-bootstrap/bin/uv python install 3.12
```

```bash
cd ~/claude_space/tsx-div-ta && ~/.uv-bootstrap/bin/uv venv --python 3.12 --seed .venv
```

```bash
.venv/bin/pip install -e ".[ai]"
```

Use `".[dev,ai]"` to also install the test and lint tools. All versions are pinned exactly in
`pyproject.toml`.

## Usage

```bash
.venv/bin/python -m tsx_div_ta scan
```

| Command | What it does |
|---|---|
| `scan` | update data incrementally, analyse, print the table, and write `reports/<date>/signals.{json,csv}` and `report.txt` |
| `scan --offline` | analyse cached data only (no network) |
| `scan --as-of 2026-06-30` | analyse as if it were that date (reproducible; doesn't touch state) |
| `scan --format json\|csv` | print JSON/CSV instead of the table (files are written either way) |
| `scan --symbols RY.TO,TD.TO` | override the universe |
| `scan --review` / `--no-review` | add or skip the Claude review |
| `update-cache [--full]` | refresh cached data only |
| `evaluate --horizons 20,60` | historical forward returns after each rule fired (descriptive, not a backtest) |

Exit codes: **0** success, **1** partial (some symbols failed or are stale), **2** fatal (bad config
or no data at all).

## Configuration

Everything lives in [`config.toml`](config.toml), and every key is optional. Unknown keys are
rejected, so a typo can't silently change behaviour. Main sections:

| Section | Controls |
|---|---|
| `[universe]` | tickers and the benchmark (default `XIC.TO`) |
| `[data]` | download batching, retries, cache, and history length |
| `[indicators]` | indicator lengths |
| `[regime]` | hysteresis and breadth thresholds |
| `[signals]` | every rule threshold |
| `[scoring.weights.*]` | weights per market regime |
| `[review]` | the AI review settings |

**Tickers:** Yahoo symbols use the `.TO` suffix, and share classes and units use a hyphen
(`RCI-B.TO`, `REI-UN.TO`).

## How it works

1. **Data.** Daily bars come from `yf.download(..., auto_adjust=False, actions=True, threads=False)`
   in batches of 10, with retries and exponential backoff.
   - **Caching:** data is cached per symbol as Parquet. Each run re-fetches only the last few days,
     and it re-downloads the full history when a new dividend or split changes Yahoo's adjusted
     prices.
   - **Partial bars:** today's bar is ignored until 16:30 ET.
   - **Validation:** bars are checked for gaps, duplicates, high < low, zero-volume runs, and moves
     larger than 40%.
2. **Prices.** Indicators use **dividend-adjusted** OHLC, so ex-dividend drops don't look like
   breakdowns. Yields use **raw** prices and payments.
3. **Indicators.** All indicators come from pandas-ta, called only from `indicators.py`:
   - SMA 50/200 and EMA 21
   - RSI, ROC, MACD and ADX/DMI
   - ATR, Bollinger Bands, Supertrend and Donchian channels
   - OBV, CMF and relative volume
   - 52-week range
   - relative strength vs the benchmark

   Tests check RSI, ATR, EMA, MACD, Bollinger Bands, Donchian channels, OBV and CMF against
   independent reference formulas.
4. **Regimes.**
   - **BULL:** close > SMA200, SMA50 > SMA200 and the SMA200 is rising.
   - **BEAR:** the mirror image.
   - **NEUTRAL:** anything else.

   A regime has to hold for 3 closes before it switches. The market regime comes from `XIC.TO`.
   Breadth (the % of stocks above their SMA200) is reported as confirming or diverging.
5. **Signals.** There are 17 rules (glossary below). Event rules report if they fired within the last
   5 bars, and state rules only if true today. A signal that goes against the market regime is scaled
   to ×0.6 strength. Set `gate_counter_regime = true` to drop those playbook signals instead.
6. **Score.** See *Scoring* below.
7. **Look-ahead safety.** Every value at bar *t* uses data up to *t* only. Tests check this with
   fixed cut-offs, random property-based cut-offs, and an end-to-end `--as-of` run compared with a
   scan of truncated data.

### Signal glossary

| Signal | Dir. | Definition |
|---|---|---|
| `golden_cross` | bull | SMA50 crosses above SMA200 |
| `pullback_uptrend` | bull | close > SMA200, RSI crosses back above 40, close within 1 ATR of EMA21/SMA50 |
| `macd_bull_cross` | bull | MACD crosses above its signal line, both > 0 |
| `breakout` | bull | close > the prior 55-day high on relative volume ≥ 1.5 (≥ 2.0 in a BEAR market) |
| `bb_reversal` | bull | low touches the lower Bollinger band, close back inside, close > SMA200 |
| `supertrend_up` | bull | Supertrend flips up with ADX > 20 |
| `death_cross` | bear | SMA50 crosses below SMA200 |
| `trend_breakdown` | bear | close crosses below SMA200, ADX > 25, −DI > +DI |
| `failed_rally` | bear | downtrend, RSI peaks in 55–60 then turns down, close < SMA50 |
| `macd_bear_cross` | bear | MACD crosses below its signal line, both < 0 |
| `relative_weakness` | bear | relative strength vs XIC at a new 63-day low |
| `supertrend_down` | bear | Supertrend flips down |
| `oversold_bounce` | watch | prior RSI < 25 and close > the prior high (counter-trend, low confidence) |
| `high_yield` | bull | TTM yield ≥ 80th percentile of its own 5-year history **and no cut in the past year** |
| `yield_trap` | watch | high yield while the stock is in a BEAR trend |
| `dividend_cut` | bear | regular dividend cut > 10% or suspended within 365 days |
| `vol_expansion` | watch | ATR% > 1.5× its 1-year median |

**Dividend metrics.** Two yields are reported:
- **TTM** is the sum of the last N regular payments, where N (payments per year) is inferred from the
  payment gaps. This is robust to pay-date jitter, unlike a fixed 365-day window.
- **Fwd** is the latest payment × N, so it reflects a cut straight away.

Special dividends (more than 2× the recent median) are excluded. A suspension is detected when no
payment arrives within max(2× the usual gap, 120 days).

### Scoring

The score is in **[-100, +100]** and is built from **7 components**, each the mean of its factors.
Every factor is in [-1, 1].

| Component | Factors |
|---|---|
| trend | price vs SMA200, SMA50/200 gap, SMA200 slope, Supertrend, ADX signed by DI |
| momentum | RSI, MACD histogram / ATR, 3-month ROC |
| relative_strength | 3-month out/under-performance vs XIC, position in the 63-day relative-strength range |
| volume | CMF, OBV slope, today's direction × relative volume |
| dividend | yield percentile (excluded right after a cut), dividend growth, cut safety |
| risk | volatility vs normal, drawdown from the 52-week high, overextension from SMA200 |
| signals | bullish minus bearish signal strength |

**Weights** depend on the market regime. For example, trend and risk count more in a BEAR market.

**Points:** each factor's contribution to the score is reported in points, and the points add up
exactly to the score. The console shows the top 3 drivers for each stock, and the JSON has every
factor.

**Confidence** is the share of factors with data. Missing factors (no benchmark, short history) are
excluded rather than guessed.

**Outlook:**
- ≥ 50: STRONG_BULLISH
- ≥ 20: BULLISH
- between -20 and +20: NEUTRAL
- ≤ -20: BEARISH
- ≤ -50: STRONG_BEARISH

## AI review (optional, low cost)

`scan --review` (or `[review] enabled = true`) sends a **compact JSON digest** of the report to Claude
and appends a short review to the table and JSON. The review covers:
- the market backdrop
- notable setups
- red flags such as yield traps, cuts, counter-regime signals, low confidence and stale data
- what to watch next

**Cost controls.** Defaults are `claude-haiku-5-5` at `effort: low`, with up to 2,000 output tokens
per call.
- **Turn cap:** the agent loop makes at most 4 API calls, and tools are switched off on the last one
  so it has to answer.
- **Two read-only tools:** `get_symbol_detail` and `get_rule_definition`. The model calls them only
  when it needs more detail.
- **Logged usage:** token usage and an estimated USD cost are recorded with every review.

For a 24-stock scan the digest is about 4.5k characters. A review uses roughly 2k input tokens
with no tool calls, up to about 15k in the worst case (4 turns, 3 tool calls). That's an estimated
**$0.001–$0.003 per review** at Haiku 5.5 prices ($0.10 / $0.50 per million tokens). The actual
tokens and cost are recorded in each report. Set `model = "claude-sonnet-5-5"` or
`"claude-opus-5-5"` for a deeper (pricier) review.

**Credentials.** The review uses the env var named by `api_key_env` (default `ANTHROPIC_API_KEY`).
Otherwise the SDK resolves `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile. The review never
changes signals or scores and never fails a scan. Problems such as no credentials, network errors or
a refusal show up as a status in the report.

## Scheduling on WSL

WSL shuts its VM down when idle, so cron inside WSL is unreliable. Use **Windows Task Scheduler** to
start `run.sh`. It runs at low priority, takes a lock so runs never overlap, uses Toronto time, and
appends to `logs/run.log`.

Run this in a Windows Command Prompt. It schedules weekdays at 15:00 **Windows local time**, which is
17:00 ET if your PC is on Mountain time. Adjust `/ST` for your timezone so it runs after the 16:30 ET
buffer.

```bash
schtasks /Create /TN "tsx-div-ta" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:00 /TR "wsl.exe -d Ubuntu -- /home/david/claude_space/tsx-div-ta/run.sh"
```

**Optional resource caps** go in `%UserProfile%\.wslconfig` on Windows (applies to all of WSL):

```
[wsl2]
memory=4GB
processors=2
```

## Development

```bash
.venv/bin/python -m pytest
```

That runs the full suite offline, with coverage enforced at ≥ 90% (currently 99%). An autouse
fixture blocks all network access. Other commands:

| Command | What it runs |
|---|---|
| `.venv/bin/python -m pytest -m network` | live Yahoo tests, if you add any (deselected by default) |
| `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests` | lint and formatting checks |
| `.venv/bin/mypy` | type checks |
| `UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_cli.py -k golden` | refresh the golden report after an intentional output change |

The 303 tests cover:
- config validation
- data cleaning, caching and incremental refresh
- indicators vs reference formulas
- every signal rule at, above and below its threshold
- regimes, dividends and scoring
- golden JSON output
- the CLI and its exit codes
- the AI review loop (with a fake client)
- randomized property-based tests
- a 30-symbol × 5-year resource-budget test

### Why pandas-ta-classic?

The original `pandas-ta` on PyPI is now `0.4.71b0`, a beta that requires Python ≥ 3.12 and
`numba==0.61.2`, a large JIT dependency that conflicts with the low-resource goal. The bot uses the
maintained community fork **`pandas-ta-classic` 0.8.32**: same API, pure pandas/numpy, NumPy 2
compatible. `indicators.py` falls back to `import pandas_ta` automatically, and the reference tests
would catch any maths differences if you swap backends.

## Known limitations

- Yahoo/yfinance is unofficial: symbols, adjustments or rate limits can change without notice.
- A change in payment frequency (e.g. quarterly → monthly) can look like a dividend cut for one
  payment.
- `evaluate` is descriptive: no costs, overlapping windows, and survivorship bias in a hand-picked
  universe.
- Signals ignore fundamentals, news and earnings. The AI review is told to say so rather than invent
  them.
