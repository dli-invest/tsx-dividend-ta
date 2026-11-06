#!/usr/bin/env bash
# Scheduled entry point (Windows Task Scheduler -> wsl.exe, cron, or a systemd timer).
# Low priority, never overlapping, Toronto time, output appended to logs/run.log.
# Extra arguments are passed to `scan`, e.g.:  ./run.sh --review
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export TZ="America/Toronto"
mkdir -p logs

exec 9>"$PROJECT_DIR/.run.lock"
if ! flock -n 9; then
  echo "$(date -Is) another run is in progress; exiting" >>logs/run.log
  exit 0
fi

{
  echo "=== $(date -Is) scan start"
  status=0
  nice -n 10 "$PROJECT_DIR/.venv/bin/python" -m tsx_div_ta scan --config "$PROJECT_DIR/config.toml" "$@" || status=$?
  echo "=== $(date -Is) scan end (exit $status)"
} >>logs/run.log 2>&1

exit "${status:-0}"
