"""Run every (candidate × mode) combination and write all plan JSONs.

Each run is a subprocess so agent_tools.py imports cleanly with the right
env vars (WITH_MEMORY and CANDIDATE are read at module import time, and
the @cognee.agent_memory decorator freezes session_id at decoration).

Candidates live under data/candidates/*.json; modes are naive / grounded.
Output lands in output/{mode}_plan_{candidate}.json.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CANDIDATES = ["dev_rao", "maria_cruz", "arjun_mehta", "priya_sharma"]
MODES = [
    ("naive", "examples.demos.recruiting_distill_memory.run_naive"),
    ("grounded", "examples.demos.recruiting_distill_memory.run_grounded"),
]


def _run(mode: str, module: str, candidate: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"[{mode}] candidate={candidate}")
    print("=" * 70)
    env = {
        "RECRUITING_CANDIDATE": candidate,
    }
    # Inherit parent env (API keys, etc.) plus per-run overrides.
    import os
    merged = {**os.environ, **env}
    cp = subprocess.run(
        [sys.executable, "-m", module],
        env=merged,
        cwd=str(HERE.parent.parent.parent),  # repo root
    )
    if cp.returncode != 0:
        raise SystemExit(f"{mode} run for {candidate} failed (exit {cp.returncode})")


def main() -> None:
    for candidate in CANDIDATES:
        for mode, module in MODES:
            _run(mode, module, candidate)
    print("\nAll runs complete. Check output/ for plan JSONs, then run")
    print("  python -m examples.demos.recruiting_distill_memory.check_rule_compliance")


if __name__ == "__main__":
    main()
