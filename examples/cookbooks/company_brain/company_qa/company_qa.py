"""Company brain: check setup, remember your company's sources, then answer across them.

    uv run python examples/cookbooks/company_brain/company_qa/company_qa.py    # the sample
    uv run python examples/cookbooks/company_brain/company_qa/company_qa.py \
        --database postgresql://user:pw@host/hr --tickets ~/exports/tickets.json \
        --docs ~/Documents/company --ask "Who owns the Atlas fix?"

With no source given it runs on the sample company from setup.py, so it works with only
LLM_API_KEY. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing, 1 a script
failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.ask import ask
from scripts.ingest import ingest
from scripts.ui import open_ui
from setup import write_sample

SAMPLE = Path(__file__).parent / "sample"
SAMPLE_TABLES = ["employee_profiles", "project_profiles", "customer_profiles"]
SAMPLE_QUESTION = (
    "Who is handling Brightline Retail's open high-priority ticket, which team are they "
    "on, and what fix was decided for it?"
)


def no_sources(args: argparse.Namespace) -> bool:
    """True when no source of your own is given, so the sample stands in."""
    return not (args.database or args.tickets or args.docs)


def use_sample(args: argparse.Namespace) -> None:
    """Point every source at the sample company that setup.py writes."""
    args.database = f"sqlite:///{SAMPLE / 'company.db'}"
    args.tables = ",".join(SAMPLE_TABLES)
    args.tickets = SAMPLE / "tickets.json"
    args.docs = SAMPLE / "docs"
    args.ask = args.ask or SAMPLE_QUESTION


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if args.sample:  # setup.py writes the sample just before the run
        return missing
    if args.tickets and not args.tickets.expanduser().is_file():
        missing.append(f"Ticket export not found: {args.tickets}")
    if args.docs and not args.docs.expanduser().is_dir():
        missing.append(f"Docs folder not found: {args.docs}")
    return missing


async def run(args: argparse.Namespace) -> None:
    tables = args.tables.split(",") if args.tables else None
    await ingest(args.database, tables, args.tickets, args.docs)
    if args.ask:
        await ask(args.ask)
    if args.ui:
        await open_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    parser.add_argument("--database", help="SQLAlchemy URL, e.g. postgresql://user:pw@host/db")
    parser.add_argument("--tables", help="comma-separated tables or views (default: all)")
    parser.add_argument("--tickets", type=Path, help="a JSON or CSV ticket export")
    parser.add_argument("--docs", type=Path, help="a folder of documents")
    parser.add_argument("--ask", help="a question to answer once the sources are remembered")
    parser.add_argument("--ui", action="store_true", help="browse the graph afterwards")
    args = parser.parse_args()
    if not args.sample and no_sources(args):
        print("[setup] No source given, so this runs on the sample company.")
        args.sample = True
    if args.sample:
        use_sample(args)

    missing = missing_setup(args)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    if args.sample:
        write_sample()
        print("[setup] Wrote the sample from setup.py.")
    asyncio.run(run(args))
