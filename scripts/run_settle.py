#!/usr/bin/env python3
"""Post-game settlement and P&L report runner.

Runs the evening/next-morning settlement sequence:
  1. settle_results.py  — score yesterday's picks against actual NBA stats
  2. pnl_report.py      — print running P&L summary

Logs stdout + stderr to logs/settle_YYYY-MM-DD.log.

Scheduled by Windows Task Scheduler to run at 6:05 AM PT daily (before the
morning run, so settled results are ready for review alongside today's picks).

NBA API has game results by ~2 AM PT, so 6:05 AM is safe.
"""

import io
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Reconfigure stdout/stderr to UTF-8 so player names with accents don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT    = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
PYTHON  = sys.executable
SEASON  = "2025-26"

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"settle_{date.today()}.log"

def run(args: list[str], log_file) -> int:
    print(f"\n>>> {' '.join(args)}", flush=True)
    log_file.write(f"\n>>> {' '.join(args)}\n")
    log_file.flush()

    result = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    log_file.write(result.stdout or "")
    log_file.flush()
    print(result.stdout or "", end="", flush=True)
    return result.returncode

with open(log_path, "a", encoding="utf-8") as lf:
    lf.write(f"\n{'='*60}\n  Settlement Run — {date.today()}\n{'='*60}\n")

    # Step 1: settle yesterday's entries
    rc1 = run([PYTHON, "scripts/settle_results.py", "--season", SEASON], lf)
    if rc1 != 0:
        lf.write(f"\n[ERROR] settle_results.py failed (rc={rc1}).\n")

    # Step 2: print full P&L summary (all time, daily breakdown)
    rc2 = run([PYTHON, "scripts/pnl_report.py", "--by-day",
               "--start", "2026-03-14"], lf)
    if rc2 != 0:
        lf.write(f"\n[ERROR] pnl_report.py failed (rc={rc2}).\n")
    else:
        lf.write("\n[OK] Settlement complete.\n")

print(f"\nLog written to: {log_path}")
sys.exit(max(rc1, rc2))
