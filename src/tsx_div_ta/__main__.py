"""Command-line interface.

Exit codes: 0 success, 1 partial (some symbols failed), 2 fatal (bad config / no data),
3 report produced but Discord delivery failed (or webhook check failed)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from tsx_div_ta import DISCLAIMER, __version__
from tsx_div_ta.clock import Clock, SystemClock
from tsx_div_ta.config import AppConfig, ConfigError, load_config
from tsx_div_ta.data.cache import ParquetCache
from tsx_div_ta.data.fetcher import Fetcher, YFinanceFetcher
from tsx_div_ta.data.store import DataStore

log = logging.getLogger("tsx_div_ta")

DEFAULT_CONFIG = "config.toml"


def _symbols(text: str) -> list[str]:
    syms = [s.strip().upper() for s in text.split(",") if s.strip()]
    if not syms:
        raise argparse.ArgumentTypeError("expected a comma-separated list of symbols")
    return syms


def _date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def _horizons(text: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected e.g. 20,60") from exc
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("horizons must be positive integers")
    return vals


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=None, help=f"TOML config (default: ./{DEFAULT_CONFIG} if present)"
    )
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    common.add_argument(
        "--symbols", type=_symbols, default=None, help="override universe, e.g. RY.TO,TD.TO"
    )

    p = argparse.ArgumentParser(prog="tsx-div-ta", description=DISCLAIMER)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", parents=[common], help="update data, analyse, report")
    scan.add_argument("--offline", action="store_true", help="use cached data only")
    scan.add_argument("--as-of", type=_date, default=None, help="analyse as if on YYYY-MM-DD")
    scan.add_argument("--format", choices=("table", "csv", "json"), default="table")
    scan.add_argument("--no-save", action="store_true", help="don't write reports or state")
    rv = scan.add_mutually_exclusive_group()
    rv.add_argument("--review", action="store_true", help="add an AI review (needs API key)")
    rv.add_argument("--no-review", action="store_true", help="skip AI review even if enabled")
    scan.add_argument(
        "--notify", action="store_true", help="post the report to Discord (webhook from env)"
    )

    sub.add_parser(
        "check-webhook",
        parents=[common],
        help="verify the Discord webhook from env without posting anything",
    )

    upd = sub.add_parser("update-cache", parents=[common], help="refresh cached market data")
    upd.add_argument("--full", action="store_true", help="re-download full history")

    ev = sub.add_parser("evaluate", parents=[common], help="historical forward returns per rule")
    ev.add_argument("--horizons", type=_horizons, default=[20, 60])
    ev.add_argument("--offline", action="store_true")
    return p


def setup_logging(logs_dir: Path, verbose: bool) -> None:
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            logs_dir / "bot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as exc:  # pragma: no cover - unwritable log dir shouldn't stop analysis
        print(f"warning: cannot write logs to {logs_dir}: {exc}", file=sys.stderr)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(sh)


def _resolve_config(path: str | None) -> AppConfig:
    if path is None:
        return load_config(DEFAULT_CONFIG) if Path(DEFAULT_CONFIG).is_file() else load_config(None)
    return load_config(path)


def main(
    argv: Sequence[str] | None = None,
    fetcher: Fetcher | None = None,
    clock: Clock | None = None,
    review_client: Any = None,
    notifier: Any = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = _resolve_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.path(cfg.data.logs_dir), args.verbose)
    if args.command == "check-webhook":
        return _cmd_check_webhook(cfg, notifier)
    clock = clock or SystemClock()
    offline = bool(getattr(args, "offline", False))
    if fetcher is None and not offline:
        fetcher = YFinanceFetcher(
            cfg.data.batch_size,
            cfg.data.batch_delay_s,
            cfg.data.max_retries,
            cfg.data.backoff_base_s,
        )
    store = DataStore(
        ParquetCache(cfg.path(cfg.data.cache_dir)), None if offline else fetcher, clock, cfg.data
    )
    symbols = args.symbols or list(cfg.universe.symbols)

    if args.command == "update-cache":
        return _cmd_update(store, symbols, cfg, args.full)
    if args.command == "evaluate":
        return _cmd_evaluate(store, symbols, cfg, args.horizons, offline, clock)
    return _cmd_scan(store, symbols, cfg, args, clock, review_client, notifier)


def _make_notifier(cfg: AppConfig, notifier: Any) -> Any:
    from tsx_div_ta.notify import notifier_from_env

    return notifier or notifier_from_env(cfg.notify.discord_webhook_env, dict(os.environ))


def _cmd_check_webhook(cfg: AppConfig, notifier: Any) -> int:
    from tsx_div_ta.notify import NotifyError

    try:
        status = _make_notifier(cfg, notifier).check()
    except NotifyError as exc:
        print(f"webhook check failed: {exc}", file=sys.stderr)
        return 3
    print(f"Discord webhook OK (HTTP {status}); nothing was posted")
    return 0


def _cmd_update(store: DataStore, symbols: list[str], cfg: AppConfig, full: bool) -> int:
    syms = list(dict.fromkeys([*symbols, cfg.universe.benchmark]))
    res = store.load(syms, full=full)
    print(
        f"updated {len(res.frames)}/{len(syms)} symbols "
        f"(full refresh: {len(res.full_refreshed)}, stale: {len(res.stale)})"
    )
    for s, err in sorted(res.failures.items()):
        print(f"  FAILED {s}: {err}")
    if not res.frames:
        return 2
    return 1 if res.failures else 0


def _cmd_evaluate(
    store: DataStore,
    symbols: list[str],
    cfg: AppConfig,
    horizons: list[int],
    offline: bool,
    clock: Clock,
) -> int:
    from tsx_div_ta.evaluate import evaluate

    syms = list(dict.fromkeys([*symbols, cfg.universe.benchmark]))
    res = store.load(syms, offline=offline)
    if not res.frames:
        print("no data available", file=sys.stderr)
        return 2
    table = evaluate(res.frames, cfg, horizons, symbols)
    out_dir = cfg.path(cfg.data.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evaluate_{clock.now().date().isoformat()}.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"\nwritten: {path}\n{DISCLAIMER}")
    return 1 if res.failures else 0


def _cmd_scan(
    store: DataStore,
    symbols: list[str],
    cfg: AppConfig,
    args: argparse.Namespace,
    clock: Clock,
    review_client: Any,
    notifier: Any = None,
) -> int:
    from tsx_div_ta.pipeline import run_scan
    from tsx_div_ta.report import render_table, to_csv, to_json, write_reports
    from tsx_div_ta.state import load_seen, save_seen

    state_path = cfg.path(cfg.data.state_file)
    historical = args.as_of is not None
    seen = set() if historical else load_seen(state_path)
    result = run_scan(cfg, store, clock, symbols, offline=args.offline, as_of=args.as_of, seen=seen)

    if (args.review or cfg.review.enabled) and not args.no_review and result.exit_code < 2:
        from tsx_div_ta.review import review_report

        result.review = review_report(result, cfg.review, client=review_client)

    paths: dict[str, Path] = {}
    if not args.no_save:
        paths = write_reports(result, cfg.path(cfg.data.reports_dir))
        log.info("reports written: %s", ", ".join(str(p) for p in paths.values()))

    render = {"table": render_table, "csv": to_csv, "json": to_json}[args.format]
    sys.stdout.write(render(result))
    sys.stdout.flush()

    delivered = True
    if args.notify:
        delivered = _notify(cfg, result, paths, notifier)
    # Only remember signals as "seen" once they were actually delivered.
    if not args.no_save and not historical and result.exit_code < 2 and delivered:
        save_seen(state_path, result.signal_keys, result.as_of)
    return result.exit_code if delivered else 3


def _notify(cfg: AppConfig, result: Any, paths: dict[str, Path], notifier: Any) -> bool:
    from tsx_div_ta.notify import NotifyError

    files = [paths[k] for k in ("txt", "csv") if k in paths] if cfg.notify.attach_files else []
    try:
        sender = _make_notifier(cfg, notifier)
    except NotifyError as exc:
        log.error("Discord notification failed: %s", exc)
        return False
    with tempfile.TemporaryDirectory(prefix="tsx-div-ta-") as tmp:
        image = _render_chart(cfg, result, paths, Path(tmp)) if cfg.notify.chart else None
        try:
            sender.send(result, files, image=image)
        except NotifyError as exc:
            log.error("Discord notification failed: %s", exc)
            return False
    return True


def _render_chart(cfg: AppConfig, result: Any, paths: dict[str, Path], tmp: Path) -> Path | None:
    """Chart next to the saved reports (or in a temp dir). A chart problem never blocks the
    text report from being delivered."""
    from tsx_div_ta.chart import render_chart

    target = (paths["json"].parent if "json" in paths else tmp) / "chart.png"
    try:
        return render_chart(result, target)
    except Exception as exc:  # plotting must never stop the notification
        log.warning("chart skipped: %s: %s", type(exc).__name__, exc)
        return None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
