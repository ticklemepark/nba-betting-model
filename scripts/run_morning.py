#!/usr/bin/env python3
"""Morning paper-trading pipeline runner.

Runs the full morning sequence for a given date (default: today):
  1. build_today_game_features.py  — fetch schedule, injuries, compute ELO/features
  2. daily_pipeline.py             — fetch Underdog lines, screen picks, build entries, log to DB

Logs stdout + stderr to logs/morning_YYYY-MM-DD.log.

Scheduled by Windows Task Scheduler to run at 6:30 AM PT daily.
"""

import io
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

# Force UTF-8 for subprocesses so log files capture non-ASCII cleanly
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Reconfigure stdout/stderr to UTF-8 so player names with accents don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
LOG_DIR   = ROOT / "logs"
PYTHON    = sys.executable
BANKROLL  = 1000.0    # paper bankroll (dollars)
MIN_EDGE  = 0.04      # 4 % minimum edge threshold
KELLY     = 0.25      # quarter-Kelly
MAX_ENTRIES = 15

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"morning_{date.today()}.log"

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
    lf.write(f"\n{'='*60}\n  Morning Run — {date.today()}\n{'='*60}\n")

    # Step 1: build today's game features + injury report
    rc1 = run([PYTHON, "scripts/build_today_game_features.py", "--force"], lf)
    if rc1 != 0:
        lf.write(f"\n[ERROR] build_today_game_features.py failed (rc={rc1}). Continuing...\n")

    # Step 2: run the main pipeline (logs picks to DB)
    rc2 = run([
        PYTHON, "scripts/daily_pipeline.py",
        "--bankroll",    str(BANKROLL),
        "--min-edge",    str(MIN_EDGE),
        "--kelly",       str(KELLY),
        "--max-entries", str(MAX_ENTRIES),
    ], lf)

    if rc2 != 0:
        lf.write(f"\n[ERROR] daily_pipeline.py failed (rc={rc2}).\n")
    else:
        lf.write("\n[OK] Morning run complete.\n")

print(f"\nLog written to: {log_path}")
sys.exit(max(rc1, rc2))
