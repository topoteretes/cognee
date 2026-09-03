"""
Build a benchmark corpus (memories.json) from a directory of documents.

The percentile benchmark eats a JSON array of ``{"title", "content"}`` objects
(see ``cognee/tests/utils/mock_ingestion``). Existing corpora — the 50 small
documents, War and Peace — were produced ad hoc; this script makes the step
reproducible for document sets that arrive as files on disk.

Usage:

    # Technical datasheets (164 PDFs) -> corpus
    python build_corpus.py --from-dir ~/Downloads/datasheets --output datasheets.json

    # Text/markdown corpora work the same way
    python build_corpus.py --from-dir ./notes --output notes.json

Supported inputs: ``.pdf`` (via pypdf), ``.txt``, ``.md``, ``.markdown``.

Title uniqueness is a correctness requirement, not cosmetics. The mock replay
in ``mock_ingestion.install_mocks`` matches a chunk to its recorded response by
testing ``title in chunk_text``. If one title is a substring of another, the
wrong knowledge graph is replayed and the benchmark silently measures the wrong
thing. This script therefore verifies pairwise non-containment and, when a
clash exists, disambiguates with a terminal ``[doc-NNNN]`` marker — unique to
one document, so no disambiguated title can be contained in any other.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | PDF_SUFFIXES

# Collapse runs of whitespace introduced by PDF layout extraction. Keeping
# paragraph breaks matters: the chunker splits on them, so flattening every
# newline would change chunk boundaries and inflate per-chunk size.
_WS_RUN = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF, page by page, with pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - environment guard
        sys.exit("pypdf is required for PDF input: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # a single broken page must not kill the corpus
            print(f"  warn: {path.name}: page extraction failed ({type(exc).__name__})")
            pages.append("")
    return "\n\n".join(pages)


def normalise(text: str) -> str:
    """Tidy extracted text without disturbing paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def title_from_path(path: Path) -> str:
    """Filename stem -> a human-readable title."""
    stem = path.stem.replace("_", " ").replace("-", " ")
    return _WS_RUN.sub(" ", stem).strip()


def find_containment_clashes(titles: list[str]) -> list[tuple[int, int]]:
    """Return (i, j) pairs where titles[i] is contained in titles[j].

    O(n^2), which is fine for the file-backed corpora this script builds
    (hundreds of documents). The arXiv builder, which works at 100k+ scale,
    sidesteps the problem entirely by minting guaranteed-unique titles.
    """
    clashes = []
    for i, a in enumerate(titles):
        for j, b in enumerate(titles):
            if i != j and a in b:
                clashes.append((i, j))
    return clashes


def build(paths: list[Path], min_chars: int) -> list[dict]:
    memories: list[dict] = []
    skipped_empty: list[str] = []

    for path in paths:
        if path.suffix.lower() in PDF_SUFFIXES:
            raw = extract_pdf_text(path)
        else:
            raw = path.read_text(encoding="utf-8", errors="replace")

        content = normalise(raw)
        if len(content) < min_chars:
            # Scanned/image-only PDFs extract to ~nothing. Including them would
            # add documents that produce no chunks and quietly shrink the corpus.
            skipped_empty.append(f"{path.name} ({len(content)} chars)")
            continue

        memories.append({"title": title_from_path(path), "content": content})

    if skipped_empty:
        print(f"\nskipped {len(skipped_empty)} document(s) below --min-chars:")
        for name in skipped_empty[:20]:
            print(f"  - {name}")
        if len(skipped_empty) > 20:
            print(f"  ... and {len(skipped_empty) - 20} more")

    # Disambiguate only where a real containment clash exists, so titles stay
    # readable in the report for the common case.
    titles = [m["title"] for m in memories]
    clashes = find_containment_clashes(titles)
    if clashes:
        involved = sorted({i for pair in clashes for i in pair})
        print(
            f"\n{len(clashes)} title-containment clash(es); disambiguating {len(involved)} title(s)"
        )
        for idx in involved:
            memories[idx]["title"] = f"{memories[idx]['title']} [doc-{idx:04d}]"

        remaining = find_containment_clashes([m["title"] for m in memories])
        if remaining:
            sys.exit(f"error: {len(remaining)} clash(es) survived disambiguation")

    return memories


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-dir", type=Path, required=True, help="Directory of source documents"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Destination memories.json"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip documents whose extracted text is shorter than this (default: 200)",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    args = parser.parse_args()

    if not args.from_dir.is_dir():
        sys.exit(f"error: {args.from_dir} is not a directory")

    globber = args.from_dir.rglob if args.recursive else args.from_dir.glob
    paths = sorted(p for p in globber("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not paths:
        sys.exit(
            f"error: no supported documents ({', '.join(sorted(SUPPORTED))}) in {args.from_dir}"
        )

    print(f"reading {len(paths)} document(s) from {args.from_dir}")
    memories = build(paths, args.min_chars)
    if not memories:
        sys.exit("error: no documents survived extraction")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")

    total_chars = sum(len(m["content"]) for m in memories)
    print(
        f"\nwrote {args.output} — {len(memories)} memories, "
        f"{total_chars:,} chars (~{total_chars / 4 / 1e6:.1f}M tokens), "
        f"{args.output.stat().st_size / 1e6:.1f} MB on disk"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
