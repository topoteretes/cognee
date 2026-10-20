"""Turn the docstrings Enola extracted into CodeDocuments and DocumentChunks.

The extraction task of the code docs chain (the docs tail of every code route,
and the memify pipeline). The text comes from Enola alone: ``docs.jsonl`` in
the snapshot directory (enola >= 0.4.32) holds the whole comment of every
documented symbol and module -- a Python docstring, a Go doc comment, a JSDoc
block, Javadoc, KDoc, Rust ``///``, Doxygen, ... -- one line per fact, joined to
the code graph by the fact's ``id`` (the ``enola_id`` of its CodeSymbol or
CodeModule). This task reads the dataset's code graph for those nodes, reads the
docs of the snapshot each repository's graph was built from, and emits one
``CodeDocument`` per file (for a module doc, per package directory) plus one
``DocumentChunk`` per documented symbol or module, its text headed
``file:line`` and ``name (kind)``.

The chunks are plain ``DocumentChunk`` objects on purpose: every existing
retriever (recall, GRAPH_COMPLETION, HYBRID, CHUNKS) already searches the
``DocumentChunk_text`` collection, so a docstring becomes searchable memory
without any retrieval change, and the ``describes`` edge the link task adds
(``add_code_doc_edges``) carries a hit on to the CodeSymbol -- and from there
to file, line, callers and flows -- in one hop.

Ids are deterministic. A document's id is a function of (repo, file) and a
chunk's of (document, text content), so the same docs yield identical ids,
text and order: the store writes nothing on an unchanged re-run, and the link
task can tell a changed docstring (new id) from an unchanged one and sweep the
old chunk.

Docs are read only from the snapshot the code graph holds
(``snapshot_docs``): its receipt must vouch for ``docs.jsonl`` and its
snapshot id must equal the repository node's ``last_snapshot_id``. A
repository whose snapshot docs are not available this run -- a clone or
staging directory gone from disk, a snapshot from an older enola or without
the docs renderer, a snapshot another run wrote over -- is left out: the run
emits nothing for it and the link task's sweep leaves its stored docs alone by
the same rule, so they are never traded for nothing.

This task never emits a CodeSymbol or CodeModule DataPoint: writing one back
would overwrite the code graph's node. The link task writes the edges to the
existing code nodes explicitly, by id.

Reading and joining a repository's docs (a file read, a hash, a JSON parse per
line, a token count per docstring) is synchronous and runs in a worker thread,
one repository at a time, so the API's event loop keeps serving while a large
repository is extracted.
"""

import asyncio
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.external_metadata import normalize_external_metadata
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_docs.models import CodeDocsBatch, CodeDocument, code_document_id
from cognee.tasks.code_graph.enola import (
    EnolaDocsUnavailable,
    is_enola_id,
    read_enola_docs,
    read_enola_fact_docs,
)

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")

CUT_TYPE = "code_docstring"
CHUNKER_ID = "code_docs"
# Chunk budget when cognify's automatic size cannot be resolved (no embedding
# engine configured, as in unit tests).
FALLBACK_MAX_CHUNK_TOKENS = 1024

# The one graph read: the repositories (their snapshot's place and identity)
# and the code nodes a docs.jsonl line can name.
_READ_TYPES = ["CodeRepository", "CodeSymbol", "CodeModule"]
# The node type a docs.jsonl line's kind names.
_NODE_TYPE_OF_KIND = {"symbol": "CodeSymbol", "module": "CodeModule"}
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


@dataclass
class _Docstring:
    """One docstring to chunk: its 2-line header, the text and the chunk metadata."""

    file_path: str
    # The document's name: the file, or for the repository root's module doc
    # "<repo>/" (enola names that package and its directory ".").
    document_name: str
    sort_key: tuple
    header: str
    text: str
    metadata: dict[str, Any]
    node_set_names: list[str]
    language: str | None
    # Counted on the document; a module doc is not a symbol.
    is_symbol: bool


# --- graph rows -------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Backend containers (UUIDs, models, JSON-in-string values) as plain JSON values."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return _plain(json.loads(value))
        except ValueError:
            pass
    return value


def node_properties(raw: Any) -> dict[str, Any]:
    """A stored node's properties as one flat dict, whichever adapter answered.

    Adapters return the properties flattened, or nested under a ``properties``
    key that may itself be JSON text, with UUIDs and JSON-in-string values on
    some backends; this is the one reading of a graph row the code docs tasks
    share (the link task's sweep reads stored nodes the same way).
    """
    properties = _plain(raw)
    if not isinstance(properties, dict):
        return {}
    nested = properties.get("properties")
    if isinstance(nested, dict):
        properties = {
            **nested,
            **{key: value for key, value in properties.items() if key != "properties"},
        }
    return properties


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _fact_properties(properties: Mapping[str, Any]) -> dict[str, Any]:
    fact_properties = properties.get("fact_properties")
    return fact_properties if isinstance(fact_properties, dict) else {}


def _text_field(properties: Mapping[str, Any], key: str) -> str | None:
    """A node's string field, else the same key of its fact properties (as enola wrote it)."""
    for value in (properties.get(key), _fact_properties(properties).get(key)):
        if isinstance(value, str) and value:
            return value
    return None


def _node_set_names(properties: Mapping[str, Any]) -> list[str]:
    """The node-set names a stored node carries (``belongs_to_set``, else ``source_node_set``)."""
    names: list[str] = []
    tags = properties.get("belongs_to_set")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                names.append(tag)
            elif isinstance(tag, dict) and isinstance(tag.get("name"), str) and tag["name"]:
                names.append(tag["name"])
    if not names and isinstance(properties.get("source_node_set"), str):
        names = [part.strip() for part in properties["source_node_set"].split(",") if part.strip()]
    return list(dict.fromkeys(names))


# --- the snapshot's docs ----------------------------------------------------


def snapshot_directory(
    repository: Mapping[str, Any], snapshot_dir: str | os.PathLike | None = None
) -> Path:
    """Where a repository's snapshot is read: ``snapshot_dir``, else ``<path>/.enola``.

    The code routes pass the snapshot they just loaded. Without it, the
    repository node's ``path`` is where an in-place route left the snapshot;
    a code file's staging directory is gone once its run ends, and a secondary
    repository of a multi-repo snapshot records only its label, so those are
    unavailable (read_enola_docs says so). Raises EnolaDocsUnavailable when
    the node records no path.
    """
    if snapshot_dir is not None:
        return Path(snapshot_dir)
    path = repository.get("path")
    if not isinstance(path, str) or not path:
        raise EnolaDocsUnavailable("the repository node records no source directory")
    return Path(path) / ".enola"


def snapshot_docs(
    repository: Mapping[str, Any], snapshot_dir: str | os.PathLike | None = None
) -> tuple[dict, ...]:
    """The docs.jsonl lines of the snapshot this repository's code graph was built from.

    ``repository`` is a stored CodeRepository node's properties; the directory
    is snapshot_directory's. The docs are those of the code graph only when the
    directory's snapshot id equals the node's ``last_snapshot_id`` (stamped
    when the snapshot was loaded): a directory another enola run wrote over
    since -- a newer commit, another enola release -- describes other facts,
    and so does one from another repository.

    Raises EnolaDocsUnavailable, naming the reason, when there is no directory
    to read, when read_enola_docs finds no usable docs.jsonl in it, or when the
    snapshot ids differ. The extraction then leaves the repository out, and
    the link task's sweep, which follows the extraction's record
    (CodeDocsBatch), keeps its stored docs.
    """
    directory = snapshot_directory(repository, snapshot_dir)
    docs = read_enola_docs(directory)
    stored_id = repository.get("last_snapshot_id")
    if not isinstance(stored_id, str) or not stored_id:
        raise EnolaDocsUnavailable(
            f"the repository node records no snapshot id to match '{directory}' against"
        )
    if docs.snapshot_id != stored_id:
        raise EnolaDocsUnavailable(
            f"the snapshot in '{directory}' is {docs.snapshot_id}, "
            f"the code graph was built from {stored_id}"
        )
    return docs.lines


# --- declarations that share a fact id -------------------------------------

# enola's ``doc`` prop: the first paragraph on one line, cut at a word past
# this many bytes and ended with "..." (DocSummary, internal/facts/doc.go).
_DOC_SUMMARY_LIMIT = 300


def _doc_summary(text: str) -> str:
    """What enola's ``doc`` prop holds for a comment: its first paragraph, on one line, capped."""
    words: list[str] = []
    for text_line in text.split("\n"):
        fields = text_line.split()
        if not fields:
            if words:
                break
            continue
        words.extend(fields)
    summary = " ".join(words)
    encoded = summary.encode("utf-8")
    if len(encoded) <= _DOC_SUMMARY_LIMIT:
        return summary
    cut = encoded.rfind(b" ", 0, _DOC_SUMMARY_LIMIT + 1)
    if cut <= 0:
        cut = _DOC_SUMMARY_LIMIT
        while cut > 0 and encoded[cut] & 0xC0 == 0x80:
            cut -= 1
    return encoded[:cut].decode("utf-8").rstrip(" ") + "..."


def _own_docs(
    directory: Path, lines: list[dict], counts: Counter
) -> dict[tuple[str, int | None], str]:
    """The text to store for each line of an id that several lines share, where it is not theirs.

    Declarations that share a fact id (overloads, a property's getter and
    setter, singledispatch registrations) are each a docs.jsonl line, at their
    own line. enola 0.4.32 keeps the whole comments in a table keyed without
    the line, so every one of those lines can carry one declaration's text.
    facts.jsonl still has each declaration's own ``doc``, enola's summary of
    its comment (_doc_summary computes the same summary): a line whose text
    does not summarize to exactly that gets it instead -- shorter, but its
    own. ``lines`` are the lines the run emits, so only they are counted and
    looked up. Returns {(id, line): replacement}.
    """
    shared = {fact_id for fact_id, n in Counter(line["id"] for line in lines).items() if n > 1}
    if not shared:
        return {}
    own = read_enola_fact_docs(directory, shared)
    replacements: dict[tuple[str, int | None], str] = {}
    for line in lines:
        key = (line["id"], line.get("line"))
        expected = own.get(key) if line["id"] in shared else None
        if expected is None:
            continue
        if _doc_summary(line["doc"]) != expected:
            replacements[key] = expected
            counts["shared_id_own_doc"] += 1
    return replacements


# --- token budget -----------------------------------------------------------


def _count_words(text: str) -> int:
    return len(text.split())


def _token_counter() -> Callable[[str], int]:
    """The counter the TextChunker cuts with (the embedding tokenizer), else words."""
    try:
        from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

        tokenizer = get_embedding_engine().tokenizer
    except Exception:
        logger.debug("No embedding tokenizer available; counting words instead.", exc_info=True)
        return _count_words
    return tokenizer.count_tokens if tokenizer is not None else _count_words


async def _default_chunk_budget() -> int:
    """The chunk size cognify runs with, else FALLBACK_MAX_CHUNK_TOKENS."""
    try:
        from cognee.infrastructure.llm.utils import get_max_chunk_tokens

        budget = await get_max_chunk_tokens()
        if isinstance(budget, int) and budget > 0:
            return budget
    except Exception:
        logger.debug("Could not resolve the chunk size; using the fallback.", exc_info=True)
    return FALLBACK_MAX_CHUNK_TOKENS


# --- chunk text -------------------------------------------------------------


def _pack(
    items: list[str], room: int, count_tokens: Callable[[str], int], joiner: str
) -> list[str]:
    """Greedily join consecutive items while they fit ``room`` tokens.

    A line or paragraph break is allowed one token per join; a space merges
    into the word that follows it and is not charged.
    """
    separator = 0 if joiner == " " else 1
    groups: list[list[str]] = []
    size = 0
    for item in items:
        item_size = count_tokens(item)
        if not groups or size + separator + item_size > room:
            groups.append([item])
            size = item_size
        else:
            groups[-1].append(item)
            size += separator + item_size
    return [joiner.join(group) for group in groups]


def _fit_paragraph(paragraph: str, room: int, count_tokens: Callable[[str], int]) -> list[str]:
    """A paragraph in pieces that fit ``room``: whole when it can be, else by lines, else words."""
    if count_tokens(paragraph) <= room:
        return [paragraph]
    pieces: list[str] = []
    for block in _pack(paragraph.splitlines(), room, count_tokens, "\n"):
        if count_tokens(block) <= room:
            pieces.append(block)
        else:
            pieces.extend(_pack(block.split(), room, count_tokens, " "))
    return pieces


def _chunk_texts(
    header: str, docstring: str, budget: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """The chunk texts of one docstring.

    One text when header + docstring fit the budget. Otherwise the docstring
    is split on blank lines and the paragraphs are packed into as few chunks
    as fit, each repeating the header; a paragraph is only split inside when
    it alone exceeds the budget.
    """
    text = f"{header}\n\n{docstring}"
    if count_tokens(text) <= budget:
        return [text]
    prefix = f"{header}\n\n"
    room = max(budget - count_tokens(prefix), 1)
    parts: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(docstring):
        paragraph = paragraph.strip("\n")
        if paragraph.strip():
            parts.extend(_fit_paragraph(paragraph, room, count_tokens))
    return [prefix + body for body in _pack(parts, room, count_tokens, "\n\n")] or [text]


def _header(file_path: str, line: int | None, name: str, kind: str) -> str:
    """``file:line`` (a module: its directory) over ``name (kind)``."""
    location = f"{file_path}:{line}" if line is not None else file_path
    return f"{location}\n{name} ({kind})"


# --- extraction -------------------------------------------------------------


def _docstring(
    repo: str,
    line: Mapping[str, Any],
    node_id: str,
    properties: Mapping[str, Any],
    text: str | None = None,
) -> _Docstring:
    """The docstring a docs.jsonl line gives the code node it names (``text`` overrides its doc)."""
    file_path, name = line["file"], line["name"]
    text = (text if text is not None else line["doc"]).strip("\n")
    language = _text_field(properties, "language")
    if line["kind"] == "module":
        if file_path in (".", ""):
            # The package at the repository root: enola names it and its
            # directory "."; the header and the document say which repository.
            location, display_name = f"{repo}/", repo
        else:
            location, display_name = file_path, name
        return _Docstring(
            file_path=file_path,
            document_name=location,
            sort_key=(0, name, text),
            header=_header(location, None, display_name, "module"),
            text=text,
            metadata={
                "repo": repo,
                "file": file_path,
                "line": None,
                "end_line": None,
                "symbol": name,
                "symbol_id": None,
                "symbol_kind": "module",
                "language": language,
                "enola_id": line["id"],
                "module_id": node_id,
            },
            node_set_names=_node_set_names(properties),
            language=language,
            is_symbol=False,
        )
    number = line.get("line")
    kind = _text_field(properties, "symbol_kind") or "symbol"
    return _Docstring(
        file_path=file_path,
        document_name=file_path,
        sort_key=(number or 0, name, text),
        header=_header(file_path, number, name, kind),
        text=text,
        metadata={
            "repo": repo,
            "file": file_path,
            "line": number,
            "end_line": _as_int(properties.get("end_line")),
            "symbol": name,
            "symbol_id": node_id,
            "symbol_kind": kind,
            "language": language,
            "enola_id": line["id"],
        },
        node_set_names=_node_set_names(properties),
        language=language,
        is_symbol=True,
    )


def _extract_repository(
    repo: str,
    repository: Mapping[str, Any],
    snapshot_dir: str | os.PathLike | None,
    code_nodes: Mapping[str, tuple[str, dict[str, Any]]],
    include_modules: bool,
    budget: int,
    count_tokens: Callable[[str], int],
    known_node_sets: Mapping[str, NodeSet],
) -> tuple[list[CodeDocument], list[DocumentChunk], Counter, dict[str, NodeSet]]:
    """One repository's documents and chunks: files sorted, docstrings by (line, name).

    Synchronous on purpose: extract_code_docs runs it in a worker thread, so
    it shares nothing mutable with the caller. ``code_nodes`` maps the
    repository's code nodes by their enola id. ``known_node_sets`` (the
    NodeSet objects earlier repositories built, by name) is read only; the
    returned dict is it plus the names this repository added, for the caller
    to merge. Raises EnolaDocsUnavailable (from snapshot_docs).
    """
    directory = snapshot_directory(repository, snapshot_dir)
    counts: Counter = Counter()
    # Another repository's lines of a multi-repo snapshot are left out: every
    # repository's directory receives every repository's lines.
    lines = [
        line
        for line in snapshot_docs(repository, directory)
        if line.get("repo") is None or line.get("repo") == repo
    ]
    joined: list[tuple[dict, tuple[str, dict[str, Any]]]] = []
    for line in lines:
        counts["lines"] += 1
        if line["kind"] == "module" and not include_modules:
            counts["modules_skipped"] += 1
            continue
        node = code_nodes.get(line["id"])
        if node is None or node[1].get("type") != _NODE_TYPE_OF_KIND[line["kind"]]:
            # A fact the code graph does not hold under that id: one the
            # mapping collapses into a same-named fact of the same kind, or
            # one it does not map.
            counts["unmatched"] += 1
            continue
        joined.append((line, node))
    own_docs = _own_docs(directory, [line for line, _node in joined], counts)
    by_file: dict[str, list[_Docstring]] = {}
    for line, node in joined:
        docstring = _docstring(repo, line, *node, own_docs.get((line["id"], line.get("line"))))
        counts["modules" if line["kind"] == "module" else "symbols"] += 1
        by_file.setdefault(docstring.file_path, []).append(docstring)

    node_sets = dict(known_node_sets)

    def sets_for(names: list[str]) -> list[NodeSet] | None:
        if not names:
            return None
        for name in names:
            # Same construction as classify_documents.update_node_set, so the
            # id matches the NodeSet node the code graph (or an add()) created.
            node_sets.setdefault(name, NodeSet(name=name))
        return [node_sets[name] for name in names]

    documents: list[CodeDocument] = []
    chunks: list[DocumentChunk] = []
    for file_path in sorted(by_file):
        docstrings = sorted(by_file[file_path], key=lambda doc: doc.sort_key)
        document_set_names = list(
            dict.fromkeys(name for doc in docstrings for name in doc.node_set_names)
        )
        language = next((doc.language for doc in docstrings if doc.language), None)
        document_name = docstrings[0].document_name
        document = CodeDocument(
            id=code_document_id(repo, file_path),
            name=document_name,
            raw_data_location=file_path,
            external_metadata=normalize_external_metadata(
                {"code_doc": {"repo": repo, "file": file_path, "language": language}}
            ),
            repo=repo,
            file_path=file_path,
            language=language,
            symbol_count=sum(1 for doc in docstrings if doc.is_symbol),
            belongs_to_set=sets_for(document_set_names),
            source_node_set=", ".join(document_set_names) or None,
        )
        documents.append(document)

        # Chunk identity is content-derived; the occurrence counter keeps two
        # identical texts in one document distinct (see chunk_id module).
        occurrences: dict[str, int] = {}
        chunk_index = 0
        for doc in docstrings:
            for text in _chunk_texts(doc.header, doc.text, budget, count_tokens):
                content_hash = chunk_content_hash(text)
                occurrence = occurrences.get(content_hash, 0)
                occurrences[content_hash] = occurrence + 1
                chunks.append(
                    DocumentChunk(
                        id=content_chunk_id(document.id, content_hash, occurrence),
                        text=text,
                        chunk_size=count_tokens(text),
                        chunk_index=chunk_index,
                        cut_type=CUT_TYPE,
                        content_hash=content_hash,
                        max_chunk_tokens=budget,
                        chunker_id=CHUNKER_ID,
                        is_part_of=document,
                        contains=[],
                        importance_weight=document.importance_weight,
                        document_id=str(document.id),
                        document_name=document_name,
                        external_metadata=normalize_external_metadata({"code_doc": doc.metadata}),
                        belongs_to_set=sets_for(doc.node_set_names),
                        source_node_set=", ".join(doc.node_set_names) or None,
                    )
                )
                chunk_index += 1
        counts["chunks"] += chunk_index
    return documents, chunks, counts, node_sets


@task_summary("Extracted {n} code doc data point(s)")
async def extract_code_docs(
    data=None,
    ctx: Optional["PipelineContext"] = None,
    repo: str | None = None,
    include_modules: bool = True,
    max_chunk_tokens: int | None = None,
    snapshot_dir: str | os.PathLike | None = None,
) -> list[DataPoint]:
    """Read the dataset's code graph and return its Enola docs as documents and chunks.

    ``data`` is whatever memify seeds the extraction with (``[{}]`` or a memory
    fragment) and is ignored: the input is the code graph itself, read through
    the graph engine, and the snapshot each repository's graph was built from.
    ``repo`` limits the run to one Enola repository label (None: every
    CodeRepository in the dataset); ``include_modules`` adds a chunk per module
    (package) doc; ``max_chunk_tokens`` overrides cognify's chunk budget.

    ``snapshot_dir`` is the snapshot directory the caller just loaded (the
    code routes pass it); without it each repository's docs are read from
    ``<CodeRepository.path>/.enola``. Either way a repository is read only when
    that snapshot is the one its code graph holds (``snapshot_docs``); one
    whose docs are not available is left out with a warning, and the link
    task's sweep keeps its stored docs. A multi-repo snapshot has one directory
    for all its repositories: each reads its own lines from it.

    Returns the CodeDocuments first, then their DocumentChunks, in a
    deterministic order: repositories by name, files by path, docstrings by
    (line, name); a module doc has its package directory's document to itself.
    The result is a CodeDocsBatch whose ``read_snapshots`` records each
    repository whose docs were read (one with no documented fact included)
    with the snapshot id read. The link task's sweep covers those whose graph
    still holds that snapshot, keeps the stored docs of every other repository
    that still has its CodeRepository node, and sweeps those of a repository
    whose node is gone (forgotten or renamed).

    The graph is read on the event loop; each repository's docs are then read
    and joined in a worker thread (``asyncio.to_thread``), repositories one
    after another, so a concurrent request is served while a large repository
    is extracted.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_filtered_graph_data([{"type": _READ_TYPES}])

    repositories: dict[str, dict[str, Any]] = {}
    code_nodes_by_repo: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}
    for item in nodes or []:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        raw_id, raw_properties = item
        properties = node_properties(raw_properties)
        node_type = properties.get("type")
        if node_type == "CodeRepository":
            if isinstance(properties.get("name"), str) and properties["name"]:
                repositories[properties["name"]] = properties
        elif node_type in ("CodeSymbol", "CodeModule"):
            node_repo, enola_id = properties.get("repo"), properties.get("enola_id")
            if isinstance(node_repo, str) and is_enola_id(enola_id):
                code_nodes_by_repo.setdefault(node_repo, {})[enola_id] = (str(raw_id), properties)

    selected = sorted(name for name in repositories if repo is None or name == repo)
    if not selected:
        logger.info(
            "No code graph repository%s in the dataset; nothing to document.",
            f" named {repo!r}" if repo is not None else "",
        )
        return CodeDocsBatch.of([], {})

    budget = max_chunk_tokens if max_chunk_tokens and max_chunk_tokens > 0 else None
    if budget is None:
        budget = await _default_chunk_budget()
    count_tokens = _token_counter()
    node_sets: dict[str, NodeSet] = {}
    dataset = getattr(getattr(ctx, "dataset", None), "name", None)

    documents: list[DataPoint] = []
    chunks: list[DataPoint] = []
    read: dict[str, str | None] = {}
    for name in selected:
        try:
            repo_documents, repo_chunks, counts, repo_node_sets = await asyncio.to_thread(
                _extract_repository,
                name,
                repositories[name],
                snapshot_dir,
                code_nodes_by_repo.get(name, {}),
                include_modules,
                budget,
                count_tokens,
                node_sets,
            )
        except EnolaDocsUnavailable as reason:
            logger.warning(
                "Code docs of repository %r%s are kept as stored, nothing is derived or swept "
                "for it this run: %s.",
                name,
                f" (dataset {dataset!r})" if dataset else "",
                reason,
            )
            continue
        # One NodeSet object per name across the run: the next repository
        # reuses what this one built, and add_data_points stores each once.
        node_sets.update(repo_node_sets)
        documents.extend(repo_documents)
        chunks.extend(repo_chunks)
        # snapshot_docs read this snapshot: the one the node recorded.
        read[name] = repositories[name].get("last_snapshot_id")
        logger.info(
            "Code docs of repository %r%s: %d docs.jsonl line(s); %d symbol and %d module "
            "doc(s) joined to the code graph, %d naming a fact the graph does not hold, %d "
            "module doc(s) left out, %d declaration(s) given their own first paragraph in "
            "place of a text another declaration of their id carried; %d document(s) and %d "
            "chunk(s) emitted.",
            name,
            f" (dataset {dataset!r})" if dataset else "",
            counts["lines"],
            counts["symbols"],
            counts["modules"],
            counts["unmatched"],
            counts["modules_skipped"],
            counts["shared_id_own_doc"],
            len(repo_documents),
            counts["chunks"],
        )
    return CodeDocsBatch.of(documents + chunks, read)
