"""Write enola-shaped snapshot directories for tests of the code graph and code docs tasks.

A snapshot directory as ``enola --generate`` leaves it, reduced to what cognee
reads: ``facts.jsonl``, ``docs.jsonl`` (enola >= 0.4.32: the whole comment of
each documented symbol and module, one JSON line per fact) and ``receipt.json``,
which vouches for ``docs.jsonl`` through ``output_hashes`` the way enola does.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cognee.tasks.code_graph.enola import DOCS_ARTIFACT


def enola_fact_id(repo: str | None, kind: str, name: str, file: str) -> str:
    """A 32-hex fact id. Deterministic per (repo, kind, name, file), like enola's; not its exact hash."""
    key = "\x00".join((repo or "", kind, name, file))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def doc_line(
    repo: str | None,
    kind: str,
    name: str,
    file: str,
    doc: str,
    line: int | None = None,
    fact_id: str | None = None,
) -> dict[str, Any]:
    """One docs.jsonl line; ``fact_id`` defaults to enola_fact_id of the same fields."""
    entry: dict[str, Any] = {
        "id": fact_id or enola_fact_id(repo, kind, name, file),
        "kind": kind,
        "name": name,
        "file": file,
        "doc": doc,
    }
    if repo is not None:
        entry["repo"] = repo
    if line is not None:
        entry["line"] = line
    return entry


def sha256_label(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def write_enola_snapshot(
    directory: str | Path,
    facts: Iterable[dict[str, Any]] | None = None,
    docs: Iterable[dict[str, Any]] | None = None,
    snapshot_id: str | None = "snapshot-1",
    vouch_docs: bool = True,
    receipt: bool = True,
) -> Path:
    """Write facts.jsonl, docs.jsonl and receipt.json into ``directory`` (created).

    ``docs`` None writes no docs.jsonl (a snapshot from enola before 0.4.32, or
    a configuration without the docs renderer); an empty iterable writes an
    empty file, which a vouching receipt makes "nothing documented". The
    receipt carries ``snapshot_id``, ``format_version`` 1, ``fact_count`` and
    ``output_hashes`` of the files written, docs.jsonl's left out when
    ``vouch_docs`` is False (an earlier run's file the receipt does not list).
    ``receipt`` False writes none (a historical writer).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fact_list = list(facts or [])
    facts_bytes = "".join(json.dumps(fact) + "\n" for fact in fact_list).encode("utf-8")
    (directory / "facts.jsonl").write_bytes(facts_bytes)
    output_hashes = {"facts.jsonl": sha256_label(facts_bytes)}
    renderers = []
    if docs is not None:
        docs_bytes = "".join(json.dumps(entry) + "\n" for entry in docs).encode("utf-8")
        (directory / DOCS_ARTIFACT).write_bytes(docs_bytes)
        if vouch_docs:
            output_hashes[DOCS_ARTIFACT] = sha256_label(docs_bytes)
            renderers.append("docs")
    if receipt:
        content = {
            "format_version": 1,
            "fact_count": len(fact_list),
            "renderers": renderers,
            "output_hashes": output_hashes,
        }
        if snapshot_id is not None:
            content["snapshot_id"] = snapshot_id
        (directory / "receipt.json").write_text(json.dumps(content), encoding="utf-8")
    return directory
