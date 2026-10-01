"""Orchestrator: check setup, then run each step folder's main.py in order.

    uv run python examples/cookbooks/inbox_qa_skill/run.py --check      # setup only, no work
    uv run python examples/cookbooks/inbox_qa_skill/run.py              # all steps
    uv run python examples/cookbooks/inbox_qa_skill/run.py --no-email --file my.txt \
        --question "What did we decide?"

Each step runs as its own process, the same command a person or an agent would type, so
a step that works here works alone. Exit codes: 0 done, 2 setup missing, otherwise the
failing step's exit code.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def missing_setup(need_gmail: bool) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    import cognee  # loads .env, so an LLM_API_KEY set there is seen

    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if need_gmail and not (HERE / "1_ingest_email" / "credentials.json").exists():
        missing.append("Gmail OAuth client not found at 1_ingest_email/credentials.json.")
    return missing


def run_step(folder: str, *args: str) -> None:
    print(f"[run] {folder}", flush=True)
    code = subprocess.call([sys.executable, str(HERE / folder / "main.py"), *args])
    if code:
        print(f"[run] FAILED: {folder} exited {code}. Run it alone to see why.")
        sys.exit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--no-email", action="store_true", help="skip the Gmail step")
    parser.add_argument("--file", type=Path, help="text file to remember (default: notes.txt)")
    parser.add_argument("--question", help="what to ask (default: about the newest email)")
    args = parser.parse_args()

    missing = missing_setup(need_gmail=not args.no_email)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    if not args.no_email:
        run_step("1_ingest_email")
    run_step("2_ingest_file", *([str(args.file)] if args.file else []))
    run_step("3_answer", *([args.question] if args.question else []))
