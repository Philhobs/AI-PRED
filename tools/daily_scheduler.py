"""APScheduler daemon for the daily operational chain (Phase 2.13).

Runs the five-step operational pipeline on weekdays at the configured local
time (default 23:00). Each step is fail-soft: a failure logs an error and
the chain continues — the system never goes dark just because one upstream
source had a bad day.

Chain:
  1. bash tools/run_refresh.sh           — ingest fresh OHLCV + feature parquets
  2. python -m tools.daily_inference     — forward predictions for new dates
  3. python -m tools.rolling_score       — score matured predictions, update log
  4. python -m tools.macro_overlay       — daily macro_risk_score + gross_scale
  5. python -m tools.daily_report        — print operator view to stdout

Each step's stdout/stderr is captured to a per-step log under
data/scheduler_logs/YYYY-MM-DD/<step>.log. A scheduler-level summary line
is appended to data/scheduler_logs/scheduler.log every run.

Run as a foreground daemon:
  python -m tools.daily_scheduler

Run the chain once (testing, manual catch-up):
  python -m tools.daily_scheduler --run-once

Customize the schedule:
  python -m tools.daily_scheduler --time 22:30
  python -m tools.daily_scheduler --time 23:00 --tz America/New_York
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

_LOG = logging.getLogger("daily_scheduler")
_PROJECT_ROOT = Path(__file__).parent.parent
_LOG_DIR = _PROJECT_ROOT / "data" / "scheduler_logs"

# (step_label, command). bash commands run as a single shell string; Python
# module invocations run via sys.executable for portability across venvs.
def _steps() -> list[tuple[str, list[str]]]:
    py = sys.executable
    return [
        ("run_refresh",     ["bash", str(_PROJECT_ROOT / "tools" / "run_refresh.sh")]),
        ("daily_inference", [py, "-m", "tools.daily_inference"]),
        ("rolling_score",   [py, "-m", "tools.rolling_score"]),
        ("macro_overlay",   [py, "-m", "tools.macro_overlay"]),
        ("daily_report",    [py, "-m", "tools.daily_report"]),
    ]


def _run_chain() -> dict:
    """Execute the five-step chain once. Returns per-step exit-code summary."""
    today = datetime.now().strftime("%Y-%m-%d")
    day_log_dir = _LOG_DIR / today
    day_log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}
    started = time.time()
    _LOG.info("[chain] START  date=%s", today)

    for label, cmd in _steps():
        step_log = day_log_dir / f"{label}.log"
        t0 = time.time()
        _LOG.info("[chain] step %s …", label)
        try:
            with step_log.open("w") as f:
                rc = subprocess.run(
                    cmd, cwd=_PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT,
                    # 4-hour wallclock cap per step. run_refresh.sh has
                    # genuine multi-hour steps (graph_features takes ~70 min
                    # of multicore compute on a typical day) and the chain
                    # fires at 23:00 weeknights, so a 4-hour cap still
                    # finishes well before market open.
                    timeout=4 * 3600,
                ).returncode
        except subprocess.TimeoutExpired:
            rc = 124  # standard "timeout" exit code
        except Exception as e:  # noqa: BLE001 — fail-soft per project convention
            _LOG.warning("[chain] step %s raised %s", label, e)
            rc = -1
        elapsed = time.time() - t0
        results[label] = rc
        marker = "OK " if rc == 0 else "ERR"
        _LOG.info("[chain] %s  %s  rc=%d  %.1fs  log=%s",
                  marker, label, rc, elapsed, step_log)

    total = time.time() - started
    summary = (f"date={today} total={total:.1f}s  " +
               " ".join(f"{k}={'OK' if rc == 0 else f'ERR({rc})'}"
                        for k, rc in results.items()))
    _LOG.info("[chain] DONE  %s", summary)

    # Append to the cumulative scheduler log
    cumlog = _LOG_DIR / "scheduler.log"
    cumlog.parent.mkdir(parents=True, exist_ok=True)
    with cumlog.open("a") as f:
        f.write(f"{datetime.now().isoformat()}  {summary}\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run-once", action="store_true",
        help="Run the chain immediately and exit; skip the scheduler loop. "
             "Useful for manual catch-up runs and for verifying the chain wiring.",
    )
    parser.add_argument(
        "--time", default="23:00",
        help="Daily run time HH:MM in --tz timezone (default 23:00).",
    )
    parser.add_argument(
        "--tz", default=None,
        help="Timezone for the schedule (default: system local). "
             "Use IANA names, e.g. 'America/New_York' for US market close.",
    )
    args = parser.parse_args()

    # Logging: human-readable, stdout-only by default. Operator can redirect
    # to a file via shell if they want persistence beyond the per-step logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.run_once:
        results = _run_chain()
        return 0 if all(rc == 0 for rc in results.values()) else 1

    try:
        hour, minute = (int(p) for p in args.time.split(":", 1))
    except ValueError:
        _LOG.error("--time must be HH:MM, got %r", args.time)
        return 2

    trigger = CronTrigger(
        day_of_week="mon-fri", hour=hour, minute=minute,
        timezone=args.tz,  # None ⇒ system local
    )
    sched = BlockingScheduler()
    sched.add_job(_run_chain, trigger=trigger, name="daily_chain", max_instances=1)

    _LOG.info("[scheduler] starting — daily_chain at mon-fri %02d:%02d %s",
              hour, minute, args.tz or "(system local)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        _LOG.info("[scheduler] interrupted, shutting down")
        sched.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
