"""Graph models of the code docstring pipeline.

A ``CodeDocument`` is one source file's docstrings, shaped as a ``Document``
so the standard cognify chain (``add_data_points``, optionally
``extract_graph_and_summarize``) treats its chunks like any other document's.
The chunks themselves are plain ``DocumentChunk`` objects, built by
``extract_code_docs`` rather than by a chunker: every retriever already
searches ``DocumentChunk_text``, so docstrings become searchable memory
without a retrieval change.

Ids are deterministic (``code_document_id``) so a re-run of the pipeline on
an unchanged repository rewrites the same nodes instead of adding copies.
"""

from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.data.processing.document_types import Document

# The edge from a docstring chunk to the CodeSymbol (or CodeModule) it documents.
DESCRIBES_RELATIONSHIP = "describes"

# Where a docstring came from: the source file, read by a language reader, or
# the ``doc`` prop enola extracted (the first paragraph, cut at 300 bytes).
DOC_SOURCE_SOURCE = "source"
DOC_SOURCE_ENOLA = "enola"


def code_document_id(repo: str, file_path: str) -> UUID:
    """The id of the CodeDocument for ``file_path`` of ``repo``: stable across runs."""
    return uuid5(NAMESPACE_OID, f"{repo}:code_doc:{file_path}")


class CodeDocument(Document):
    """One source file's docstrings, as a document the cognify chain can chunk and embed.

    ``repo`` and ``file_path`` hold the same values as the file's CodeSymbol
    nodes (Enola's repo label and the repo-relative path), which is how the
    link task finds the code graph again. ``name`` is the file path;
    ``raw_data_location`` is the absolute source path when the file was read,
    else the repo-relative path. The chunks are produced up front by
    ``extract_code_docs``, so ``Document.read`` is never called on one.
    """

    type: str = "code_doc"
    mime_type: str = "text/plain"
    repo: str
    file_path: str
    language: str | None = None
    # Documented symbols this document holds (the module docstring not counted).
    symbol_count: int = 0
    # DOC_SOURCE_SOURCE when the file was read and at least one docstring came
    # from it; DOC_SOURCE_ENOLA when every chunk carries enola's ``doc`` prop
    # (no reader for the language, or the source could not be read). A run
    # that cannot read a repository's source keeps the documents read from it
    # rather than replace them with enola's shorter docs (see extract_code_docs).
    doc_source: str = DOC_SOURCE_SOURCE
