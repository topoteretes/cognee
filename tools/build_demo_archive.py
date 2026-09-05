"""Build the COGX archive bundled with `cognee-cli demo`.

Maintainer tool, run once when the demo content changes:

    LLM_API_KEY=... uv run python tools/build_demo_archive.py

It cognifies the bundled quickstart sample (`cognee/cli/samples/quickstart.txt`)
in a throwaway store, exports the resulting graph as a COGX archive into
`cognee/cli/samples/demo_graph/`, and then trims the archive down to its
graph records:

- `documents.jsonl` is removed. Document records would make the import call
  `add()`, whose pipeline demands a working LLM connection — the whole point
  of the demo is to need no API key. The chunk text survives regardless: the
  exporter also writes every DocumentChunk as a raw node, which is exactly
  what CHUNKS_LEXICAL retrieval reads.
- The manifest's counts are rewritten to match.

The resulting archive is committed; `cognee-cli demo` restores it graph-only
(no embeddings) via `remember(COGXArchiveSource(...), index_vectors=False)`.

Requires a configured LLM (and embedding) provider — this script performs the
one real cognify so end users never have to.
"""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "cognee" / "cli" / "samples"
QUICKSTART = SAMPLES_DIR / "quickstart.txt"
ARCHIVE_DIR = SAMPLES_DIR / "demo_graph"
DATASET_NAME = "demo_archive_seed"


async def build() -> None:
    # Isolate the build in a scratch store so it never touches (or reads)
    # the developer's own cognee databases.
    scratch = Path(tempfile.mkdtemp(prefix="cognee_demo_build_"))
    print(f"Scratch store: {scratch}")

    import cognee
    from cognee import config

    config.system_root_directory(str(scratch / "system"))
    config.data_root_directory(str(scratch / "data"))

    text = QUICKSTART.read_text(encoding="utf-8")
    print("Cognifying the quickstart sample (this makes real LLM calls)...")
    # Small chunks, one per paragraph-ish: CHUNKS_LEXICAL returns whole chunks
    # as answers and the demo prints them in full, so a chunk must BE a
    # displayable answer — with one document-sized chunk every query returns
    # the same intro text and the facts never fit on screen.
    await cognee.remember(text, dataset_name=DATASET_NAME, self_improvement=False, chunk_size=60)

    print(f"Exporting COGX archive to {ARCHIVE_DIR} ...")
    if ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)
    await cognee.export(DATASET_NAME, format="cogx", destination=ARCHIVE_DIR)

    # Trim to graph records only (see module docstring).
    removed = {}
    for file_name, kind in (("documents.jsonl", "document"), ("episodes.jsonl", "episode")):
        path = ARCHIVE_DIR / file_name
        if path.exists():
            removed[kind] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
            path.unlink()

    # The exporter records absolute build-machine paths in raw_data_location;
    # null them so the bundled archive neither ships the maintainer's
    # filesystem layout nor imports dead file:// provenance on user machines.
    # Exported nodes are flat dicts; handle a nested properties dict too so a
    # future export-format change cannot silently re-leak the paths.
    nodes_path = ARCHIVE_DIR / "nodes.jsonl"
    if nodes_path.exists():
        sanitized = []
        for line in nodes_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            node = json.loads(line)
            for record in (node, node.get("properties")):
                if isinstance(record, dict) and "raw_data_location" in record:
                    record["raw_data_location"] = None
            sanitized.append(json.dumps(node, default=str))
        nodes_path.write_text("\n".join(sanitized) + "\n", encoding="utf-8")

    manifest_path = ARCHIVE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for kind in removed:
        manifest.get("counts", {}).pop(kind, None)
    manifest.setdefault("notes", []).append(
        "Bundled cognee-cli demo archive: trimmed to graph records so the "
        "import is zero-LLM and key-free (built by tools/build_demo_archive.py)."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Removed record kinds: {removed or 'none'}")
    print(f"Final counts: {manifest.get('counts')}")
    for path in sorted(ARCHIVE_DIR.iterdir()):
        print(f"  {path.name}  {path.stat().st_size} bytes")

    shutil.rmtree(scratch, ignore_errors=True)
    print("Done. Commit the archive directory.")


if __name__ == "__main__":
    if not QUICKSTART.exists():
        sys.exit(f"Quickstart sample not found: {QUICKSTART}")
    asyncio.run(build())
