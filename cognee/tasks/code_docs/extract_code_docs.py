"""Turn the code graph's documented symbols into DocumentChunks.

The extraction task of the code docs memify pipeline. It reads the dataset's
code graph (CodeRepository, CodeSymbol and CodeModule nodes), reads each
documented symbol's docstring from the repository source on the server
(Python via ``ast``; Enola's ``doc`` prop when the source cannot be read) and
emits one ``CodeDocument`` per source file plus one ``DocumentChunk`` per
documented symbol, its text starting with ``file:line``.

Coverage is what the Python reader finds in the source, plus the Enola
fallback for symbols whose facts carry a ``doc`` prop: the first paragraph of
the doc comment, cut at 300 bytes, which enola 0.4.30 (the pinned release)
emits for Python, Go and TypeScript/JavaScript symbols; snapshots written by
older releases carry none. On this repository the fallback supplied the
frontend's TS/JS docstrings and the few Python symbols the source matcher did
not place. A full docstring for another language needs a reader in
``docstrings.READERS``.

The chunks are plain ``DocumentChunk`` objects on purpose: every existing
retriever (recall, GRAPH_COMPLETION, HYBRID, CHUNKS) already searches the
``DocumentChunk_text`` collection, so a docstring becomes searchable memory
without any retrieval change, and the ``describes`` edge the link task adds
(``add_code_doc_edges``) carries a hit on to the CodeSymbol -- and from there
to file, line, callers and flows -- in one hop.

Ids are deterministic. A document's id is a function of (repo, file path)
and a chunk's of (document, text content), so the same graph over the same
sources yields identical ids, text and order: re-running the pipeline
rewrites the same nodes, and the link task can tell a changed docstring (new
id) from an unchanged one and sweep the old chunk.

This task never emits a CodeSymbol or CodeModule DataPoint: writing one back
would overwrite the code graph's node. The link task writes the edges to the
existing code nodes explicitly, by id.
"""

import json
import os
import posixpath
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.external_metadata import normalize_external_metadata
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_docs.docstrings import (
    STATUS_MISSING,
    STATUS_PARSE_ERROR,
    FileDocs,
    language_of,
    module_name_of,
    read_file_docs,
    symbol_dotted_name,
)
from cognee.tasks.code_docs.models import CodeDocument, code_document_id
from cognee.tasks.code_graph.extract_code_graph import fact_node_id

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")

CUT_TYPE = "code_docstring"
CHUNKER_ID = "code_docs"
DOC_SOURCE_SOURCE = "source"
DOC_SOURCE_ENOLA = "enola"
# Chunk budget when cognify's automatic size cannot be resolved (no embedding
# engine configured, as in unit tests).
FALLBACK_MAX_CHUNK_TOKENS = 1024

_CODE_GRAPH_TYPES = ["CodeRepository", "CodeSymbol", "CodeModule"]
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


@dataclass
class _Docstring:
    """One docstring to chunk: its 2-3 line header, the text and the chunk metadata."""

    header: str
    text: str
    metadata: dict[str, Any]
    node_set_names: list[str]
    # Counted on the document; the module docstring is not a symbol.
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


def _enola_doc(properties: Mapping[str, Any]) -> str | None:
    """Enola's ``doc`` prop: the top-level field newer loads write, else the prop."""
    for value in (properties.get("doc"), _fact_properties(properties).get("doc")):
        if isinstance(value, str) and value.strip():
            return value.strip()
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


def _header(file_path: str, line: int, name: str, kind: str, signature: str | None) -> str:
    header = f"{file_path}:{line}\n{name} ({kind})"
    return f"{header}\n{signature}" if signature else header


# --- extraction -------------------------------------------------------------


def _file_docstrings(
    repo: str,
    repo_path: str | None,
    file_path: str,
    symbols: list[tuple[str, dict[str, Any]]],
    module_ids: set[str],
    include_modules: bool,
    counts: Counter,
) -> tuple[FileDocs, str | None, list[_Docstring]]:
    """The docstrings of one source file, module docstring first, symbols in source order."""
    has_reader = language_of(file_path) is not None
    if repo_path is None:
        file_docs = FileDocs(status=STATUS_MISSING, language=language_of(file_path))
    else:
        file_docs = read_file_docs(repo_path, file_path)
    if not has_reader:
        counts["files_unsupported"] += 1
    elif file_docs.status == STATUS_PARSE_ERROR:
        counts["parse_errors"] += 1
    elif not file_docs.readable:
        counts["files_unreadable"] += 1

    language = file_docs.language
    file_set_names: list[str] = []
    for _node_id, properties in symbols:
        language = language or _fact_properties(properties).get("language")
        file_set_names.extend(_node_set_names(properties))
    file_set_names = list(dict.fromkeys(file_set_names))
    if not isinstance(language, str) or not language:
        language = None

    docstrings: list[_Docstring] = []
    if include_modules and file_docs.module_doc:
        module_name = module_name_of(file_path)
        module_node_id = str(fact_node_id(repo, "module", posixpath.dirname(file_path) or "."))
        docstrings.append(
            _Docstring(
                header=_header(file_path, 1, module_name, "module", None),
                text=file_docs.module_doc,
                metadata={
                    "repo": repo,
                    "file": file_path,
                    "line": 1,
                    "end_line": None,
                    "symbol": module_name,
                    "symbol_id": None,
                    "symbol_kind": "module",
                    "language": language,
                    "doc_source": DOC_SOURCE_SOURCE,
                    # The CodeModule of the file's directory, when the graph holds it.
                    "module_id": module_node_id if module_node_id in module_ids else None,
                },
                node_set_names=file_set_names,
                is_symbol=False,
            )
        )

    for node_id, properties in symbols:
        counts["symbols"] += 1
        name = properties["name"]
        line = _as_int(properties.get("line"))
        kind = properties.get("symbol_kind") or _fact_properties(properties).get("symbol_kind")
        found = (
            file_docs.find(line, symbol_dotted_name(name, file_path))
            if file_docs.readable
            else None
        )
        if found is not None and found.docstring:
            text, doc_source = found.docstring, DOC_SOURCE_SOURCE
            doc_line, end_line, signature = found.line, found.end_line, found.signature
            kind = kind or found.kind
            counts["documented_from_source"] += 1
        else:
            text = _enola_doc(properties)
            if text is None:
                # Why there is nothing to store, so the summary line does not
                # report a TypeScript interface or a variable as a def that
                # lacks a docstring.
                if not has_reader:
                    counts["symbols_unsupported_language"] += 1
                elif not file_docs.readable:
                    counts["symbols_in_unreadable_files"] += 1
                elif found is None:
                    counts["symbols_not_defs"] += 1
                else:
                    counts["undocumented"] += 1
                continue
            if line is None:
                counts["symbols_without_line"] += 1
                continue
            doc_source = DOC_SOURCE_ENOLA
            doc_line, end_line, signature = line, _as_int(properties.get("end_line")), None
            counts["documented_from_enola"] += 1
        kind = kind if isinstance(kind, str) and kind else "symbol"
        docstrings.append(
            _Docstring(
                header=_header(file_path, doc_line, name, kind, signature),
                text=text,
                metadata={
                    "repo": repo,
                    "file": file_path,
                    "line": doc_line,
                    "end_line": end_line,
                    "symbol": name,
                    "symbol_id": node_id,
                    "symbol_kind": kind,
                    "language": language,
                    "doc_source": doc_source,
                },
                node_set_names=_node_set_names(properties),
                is_symbol=True,
            )
        )
    return file_docs, language, docstrings


def _extract_repository(
    repo: str,
    repo_path: str | None,
    symbols: list[tuple[str, dict[str, Any]]],
    module_ids: set[str],
    include_modules: bool,
    budget: int,
    count_tokens: Callable[[str], int],
    node_sets: dict[str, NodeSet],
) -> tuple[list[CodeDocument], list[DocumentChunk], Counter]:
    """One repository's documents and chunks, files sorted, symbols by (line, name)."""
    counts: Counter = Counter()
    by_file: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for node_id, properties in symbols:
        file_path = properties.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            counts["symbols_without_file"] += 1
            continue
        by_file.setdefault(file_path, []).append((node_id, properties))

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
        counts["files"] += 1
        file_symbols = sorted(
            by_file[file_path],
            key=lambda item: (_as_int(item[1].get("line")) or 0, str(item[1].get("name"))),
        )
        file_docs, language, docstrings = _file_docstrings(
            repo, repo_path, file_path, file_symbols, module_ids, include_modules, counts
        )
        if not docstrings:
            continue

        document_set_names = list(
            dict.fromkeys(name for doc in docstrings for name in doc.node_set_names)
        )
        document = CodeDocument(
            id=code_document_id(repo, file_path),
            name=file_path,
            raw_data_location=os.path.join(repo_path, file_path)
            if repo_path is not None and file_docs.readable
            else file_path,
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
            # A module docstring is tagged like its document: the union of the
            # file's symbols' sets.
            set_names = doc.node_set_names if doc.is_symbol else document_set_names
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
                        document_name=file_path,
                        external_metadata=normalize_external_metadata({"code_doc": doc.metadata}),
                        belongs_to_set=sets_for(set_names),
                        source_node_set=", ".join(set_names) or None,
                    )
                )
                chunk_index += 1
        counts["chunks"] += chunk_index
    return documents, chunks, counts


@task_summary("Extracted {n} code doc data point(s)")
async def extract_code_docs(
    data=None,
    ctx: Optional["PipelineContext"] = None,
    repo: str | None = None,
    include_modules: bool = True,
    max_chunk_tokens: int | None = None,
) -> list[DataPoint]:
    """Read the dataset's code graph and return its docstrings as documents and chunks.

    ``data`` is whatever memify seeds the extraction with (``[{}]`` or a memory
    fragment) and is ignored: the input is the code graph itself, read through
    the graph engine. ``repo`` limits the run to one Enola repository label
    (None: every CodeRepository in the dataset); ``include_modules`` adds a
    chunk per file-level docstring; ``max_chunk_tokens`` overrides cognify's
    chunk budget.

    Returns the CodeDocuments first, then their DocumentChunks, in a
    deterministic order: repositories by name, files by path, symbols by
    (line, name), the module docstring first in its file.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_filtered_graph_data([{"type": _CODE_GRAPH_TYPES}])

    repositories: dict[str, dict[str, Any]] = {}
    symbols_by_repo: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    module_ids_by_repo: dict[str, set[str]] = {}
    for item in nodes or []:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        raw_id, raw_properties = item
        node_id = str(raw_id)
        properties = node_properties(raw_properties)
        node_type = properties.get("type")
        if node_type == "CodeRepository":
            if isinstance(properties.get("name"), str) and properties["name"]:
                repositories[properties["name"]] = properties
        elif node_type == "CodeSymbol":
            if isinstance(properties.get("repo"), str) and isinstance(properties.get("name"), str):
                symbols_by_repo.setdefault(properties["repo"], []).append((node_id, properties))
        elif node_type == "CodeModule" and isinstance(properties.get("repo"), str):
            module_ids_by_repo.setdefault(properties["repo"], set()).add(node_id)

    selected = sorted(name for name in repositories if repo is None or name == repo)
    if not selected:
        logger.info(
            "No code graph repository%s in the dataset; nothing to document.",
            f" named {repo!r}" if repo is not None else "",
        )
        return []

    budget = max_chunk_tokens if max_chunk_tokens and max_chunk_tokens > 0 else None
    if budget is None:
        budget = await _default_chunk_budget()
    count_tokens = _token_counter()
    node_sets: dict[str, NodeSet] = {}
    dataset = getattr(getattr(ctx, "dataset", None), "name", None)

    documents: list[DataPoint] = []
    chunks: list[DataPoint] = []
    for name in selected:
        repo_path = repositories[name].get("path")
        if not isinstance(repo_path, str) or not os.path.isdir(repo_path):
            logger.warning(
                "Repository %r has no readable source directory at %r; "
                "using the docs Enola extracted only.",
                name,
                repo_path,
            )
            repo_path = None
        repo_documents, repo_chunks, counts = _extract_repository(
            name,
            repo_path,
            symbols_by_repo.get(name, []),
            module_ids_by_repo.get(name, set()),
            include_modules,
            budget,
            count_tokens,
            node_sets,
        )
        documents.extend(repo_documents)
        chunks.extend(repo_chunks)
        logger.info(
            "Code docs of repository %r%s: %d file(s), %d symbol(s) seen; %d documented from "
            "source, %d from enola; without a docstring: %d def/class, %d not a def/class in "
            "the source (variables, stale lines), %d in %d file(s) with no reader for the "
            "language, %d in %d unreadable file(s) (%d parse error(s)); %d document(s) and "
            "%d chunk(s) emitted.",
            name,
            f" (dataset {dataset!r})" if dataset else "",
            counts["files"],
            counts["symbols"],
            counts["documented_from_source"],
            counts["documented_from_enola"],
            counts["undocumented"],
            counts["symbols_not_defs"],
            counts["symbols_unsupported_language"],
            counts["files_unsupported"],
            counts["symbols_in_unreadable_files"],
            counts["files_unreadable"] + counts["parse_errors"],
            counts["parse_errors"],
            len(repo_documents),
            counts["chunks"],
        )
    return documents + chunks
