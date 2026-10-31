"""TrustRAG: Verifiable Financial Document Intelligence on Cognee.

Ingests complex financial documents (regulatory notices, credit agreements, 10-K filings),
constructs a domain knowledge graph in Cognee, and answers questions with atomic claim
decomposition, natural language inference (NLI) verification, and trust score gating.

Usage:
    # Run end-to-end on synthetic sample financial documents:
    uv run python examples/cookbooks/trustrag/trustrag.py

    # Check setup and prerequisites:
    uv run python examples/cookbooks/trustrag/trustrag.py --check

    # Point at your own financial documents and ask a question:
    uv run python examples/cookbooks/trustrag/trustrag.py --docs path/to/filings --ask "What is the liability cap under Section 11?"

Exit codes:
    0: Success
    1: Ingestion or verification error
    2: Missing configuration or setup prerequisites
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")

COOKBOOK_DIR = Path(__file__).parent
REPO_ROOT = COOKBOOK_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COOKBOOK_DIR))

try:
    from examples.cookbooks.trustrag.config import default_config
    from examples.cookbooks.trustrag.scripts.ask import ask
    from examples.cookbooks.trustrag.scripts.clear import clear_dataset
    from examples.cookbooks.trustrag.scripts.ingest import ingest
    from examples.cookbooks.trustrag.setup import write_sample
except ImportError:
    from config import default_config
    from scripts.ask import ask
    from scripts.clear import clear_dataset
    from scripts.ingest import ingest
    from setup import write_sample

SAMPLE_DIR = COOKBOOK_DIR / "sample"
DEFAULT_QUESTION = (
    "What are the annual stress testing parameters under Regulatory Notice 24-99, "
    "and what was Acme Financial's total revenue in fiscal year 2024?"
)


def missing_setup(args: argparse.Namespace) -> list[str]:
    """Check configuration and prerequisites. Returns list of missing items."""
    missing: list[str] = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (specify it in your environment or .env).")

    if not args.sample and args.docs:
        docs_path = Path(args.docs).expanduser().resolve()
        if not docs_path.exists():
            missing.append(f"Docs folder not found: {docs_path}")

    return missing


def use_sample(args: argparse.Namespace) -> None:
    """Prepare and use synthetic sample financial documents."""
    if not SAMPLE_DIR.exists() or not list(SAMPLE_DIR.glob("*.txt")):
        print("[setup] Generating synthetic sample financial documents...")
        write_sample(SAMPLE_DIR)
    args.docs = SAMPLE_DIR
    args.ask = args.ask or DEFAULT_QUESTION


async def run(args: argparse.Namespace) -> None:
    ds = args.dataset or default_config.dataset_name

    if args.clear:
        print(f"[setup] Clearing dataset '{ds}' before ingestion...")
        await clear_dataset(dataset_name=ds, force=True)

    if args.docs:
        print(f"[ingest] Ingesting documents from {args.docs} into dataset '{ds}'...")
        await ingest(docs_folder=Path(args.docs), dataset_name=ds)

    if args.ask:
        await ask(question=args.ask, dataset_name=ds, verbose=args.verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check prerequisites and exit")
    parser.add_argument("--sample", action="store_true", help="use synthetic sample documents")
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        help="clear dataset before remembering (default: on for sample, off otherwise)",
    )
    parser.add_argument("--docs", type=Path, help="folder of documents to ingest")
    parser.add_argument("--ask", help="question to ask once documents are remembered")
    parser.add_argument("--dataset", help="target Cognee dataset name")
    parser.add_argument("--verbose", "-v", action="store_true", help="show verbose evidence details")
    args = parser.parse_args()

    if args.check:
        missing = missing_setup(args)
        if missing:
            print("[check] Setup missing prerequisites:", file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            sys.exit(2)
        else:
            print("[check] All prerequisites configured.")
            sys.exit(0)

    # If no source was given, default to synthetic sample
    if not args.sample and not args.docs:
        print("[setup] No documents given; running on synthetic sample.")
        args.sample = True

    if args.sample:
        use_sample(args)

    if args.clear is None:
        args.clear = args.sample

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
