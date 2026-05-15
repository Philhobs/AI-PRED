"""Cron-able health check for tools/daily_scheduler.py.

Reads data/scheduler_logs/scheduler.log (the cumulative summary written
by daily_scheduler after each chain run). Exits 0 if today's most-recent
entry is present and all steps ended OK; exits 1 otherwise. Suitable
for wiring into a system-level alert via cron / launchd / cronitor / etc.

Default behaviour (no args) checks today's entry. On weekends or holidays
when the scheduler doesn't fire by design, the check is configurable to
not flag a missing entry.

Exit codes:
  0  today's chain ran and all 5 steps returned OK
  1  today's chain ran but at least one step had a non-zero rc
  2  no entry for today (scheduler likely didn't run)
  3  log file missing entirely

Run:
  python -m tools.scheduler_health_check                      # check today
  python -m tools.scheduler_health_check --date 2026-05-15
  python -m tools.scheduler_health_check --skip-weekends      # don't flag sat/sun
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_LOG_PATH = _PROJECT_ROOT / "data" / "scheduler_logs" / "scheduler.log"


def _latest_entry_for_date(target: date) -> str | None:
    """Return the most recent log line for `target`, or None."""
    if not _LOG_PATH.exists():
        return None
    target_str = target.isoformat()
    last: str | None = None
    for line in _LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        # Each line starts with an ISO datetime then a space then 'date=YYYY-MM-DD'.
        if f"date={target_str}" in line:
            last = line
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--date", default=None,
        help="ISO date to check (default: today's local date).",
    )
    parser.add_argument(
        "--skip-weekends", action="store_true",
        help="Don't flag a missing entry on Saturday or Sunday (the scheduler "
             "is wired for mon-fri only by default).",
    )
    args = parser.parse_args()

    target = (date.fromisoformat(args.date) if args.date
              else datetime.now().date())

    if args.skip_weekends and target.weekday() >= 5:
        print(f"[health-check] {target} is a weekend; skipping check.")
        return 0

    if not _LOG_PATH.exists():
        print(f"[health-check] FAIL: log file missing at {_LOG_PATH}")
        return 3

    entry = _latest_entry_for_date(target)
    if entry is None:
        print(f"[health-check] FAIL: no scheduler entry for {target}")
        print(f"  log path: {_LOG_PATH}")
        return 2

    if "ERR(" in entry:
        print(f"[health-check] WARN: at least one step failed on {target}")
        print(f"  {entry}")
        return 1

    print(f"[health-check] OK: {target} chain ran cleanly")
    print(f"  {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
