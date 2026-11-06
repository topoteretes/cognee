"""Ingest financial documents into Cognee using the domain graph model.

Extracts financial entities (Organization, Person, RegulatoryFiling, Regulation,
FinancialInstrument) and establishes relationships using Cognee's cognification pipeline.

Features:
- Robust document-level error propagation and summary statistics.
- Strict exit codes: 0 for full success, 1 for errors.
- Never silently ignores ingestion or cognification failures.

Run alone: uv run python examples/cookbooks/trustrag/scripts/ingest.py [--docs FOLDER] [--dataset NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee

# Add cookbook root to sys.path so models and config are importable
COOKBOOK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(COOKBOOK_DIR))

from config import default_config
from models import EXTRACTION_PROMPT, FinancialGraph

logger = logging.getLogger(__name__)


async def ingest_documents(
    doc_paths: list[Path],
    dataset_name: str | None = None,
) -> tuple[int, int, list[str]]:
    """Remember and cognify documents into Cognee.

    Returns:
        (succeeded_count, failed_count, failure_reasons)
    """
    ds = dataset_name or default_config.dataset_name
    succeeded = 0
    failed = 0
    errors: list[str] = []

    for path in doc_paths:
        expanded = path.expanduser().resolve()
        if not expanded.exists():
            failed += 1
            err = f"File not found: {expanded}"
            errors.append(err)
            print(f"[ingest] ERROR: {err}", file=sys.stderr)
            continue

        try:
            print(f"[ingest] Ingesting {expanded.name} into dataset '{ds}'...")
            await cognee.remember(
                str(expanded),
                dataset_name=ds,
                node_set=["financial_docs"],
                graph_model=FinancialGraph,
                custom_prompt=EXTRACTION_PROMPT,
            )
            succeeded += 1
            print(f"[ingest] Successfully remembered and cognified {expanded.name}")
        except Exception as e:
            failed += 1
            err = f"Failed to ingest {expanded.name}: {e}"
            errors.append(err)
            print(f"[ingest] ERROR: {err}", file=sys.stderr)

    return succeeded, failed, errors


async def ingest(
    docs_folder: Path | None = None,
    doc_files: list[Path] | None = None,
    dataset_name: str | None = None,
) -> None:
    """Ingest files from a folder or file list into Cognee."""
    paths: list[Path] = []
    if doc_files:
        paths.extend(doc_files)
    if docs_folder:
        folder = docs_folder.expanduser().resolve()
        if folder.is_dir():
            for p in sorted(folder.glob("*")):
                if p.is_file() and not p.name.startswith("."):
                    paths.append(p)
        elif folder.is_file():
            paths.append(folder)

    if not paths:
        sample_dir = COOKBOOK_DIR / "sample"
        if sample_dir.is_dir():
            paths = sorted(sample_dir.glob("*.txt"))

    if not paths:
        print("[ingest] No documents found to ingest. Run setup.py first or specify --docs.", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Found {len(paths)} document(s) for ingestion.")
    succeeded, failed, errors = await ingest_documents(paths, dataset_name=dataset_name)

    print("\n--- Ingestion Summary ---")
    print(f"Total documents: {len(paths)}")
    print(f"Succeeded:       {succeeded}")
    print(f"Failed:          {failed}")

    if failed > 0:
        print(f"[ingest] Encountered {failed} failure(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=Path, help="folder of documents or single file to ingest")
    parser.add_argument("--files", type=Path, nargs="+", help="individual files to ingest")
    parser.add_argument("--dataset", help="target Cognee dataset name")
    args = parser.parse_args()

    asyncio.run(ingest(docs_folder=args.docs, doc_files=args.files, dataset_name=args.dataset))

