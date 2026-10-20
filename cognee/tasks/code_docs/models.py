"""Graph models of the code docstring pipeline.

A ``CodeDocument`` is one source file's docstrings (for a module doc, one
package directory's), shaped as a ``Document`` so the standard cognify chain
(``add_data_points``, optionally ``extract_graph_and_summarize``) treats its
chunks like any other document's. The chunks themselves are plain
``DocumentChunk`` objects, built by ``extract_code_docs`` from Enola's
``docs.jsonl`` rather than by a chunker: every retriever already searches
``DocumentChunk_text``, so docstrings become searchable memory without a
retrieval change.

Ids are deterministic (``code_document_id``) so a re-run of the pipeline on
an unchanged repository rewrites the same nodes instead of adding copies.
"""

from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.data.processing.document_types import Document

# The edge from a docstring chunk to the CodeSymbol (or CodeModule) it documents.
DESCRIBES_RELATIONSHIP = "describes"


class CodeDocsBatch(list):
    """A code docs run's data points, with the repositories whose snapshot docs it read.

    extract_code_docs decides once, per repository, whether the docs of the
    snapshot its code graph holds could be read, and records the ones it read
    in ``read_repos``. The link task's sweep honours exactly that decision: a
    repository it did not read keeps its stored docs, whatever the snapshot
    directory or the repository node look like by the time the sweep runs
    (another run may write either in between). The pipeline hands a
    coroutine's result to the next task as the same object, so the set rides
    along; a task that builds a new list from the batch carries it over with
    ``carry`` (cognify_code_doc_chunks), and the store passes the batch itself
    on. A plain list has no ``read_repos``, and the sweep then checks the
    snapshots itself.
    """

    read_repos: frozenset[str] | None = None

    @classmethod
    def of(cls, points, read_repos) -> "CodeDocsBatch":
        batch = cls(points)
        batch.read_repos = frozenset(read_repos)
        return batch

    @classmethod
    def carry(cls, source, points):
        """``points`` with the read set of ``source``, when ``source`` has one; else ``points``."""
        read_repos = getattr(source, "read_repos", None)
        return points if read_repos is None else cls.of(points, read_repos)


def code_document_id(repo: str, file_path: str) -> UUID:
    """The id of the CodeDocument for ``file_path`` of ``repo``: stable across runs."""
    return uuid5(NAMESPACE_OID, f"{repo}:code_doc:{file_path}")


class CodeDocument(Document):
    """One source file's docstrings, as a document the cognify chain can chunk and embed.

    ``repo`` and ``file_path`` are Enola's repository label and the ``file`` of
    its docs.jsonl lines: the repo-relative path of a symbol's file, or the
    directory of a module (package). ``name`` and ``raw_data_location`` hold
    that path too: the text comes from the snapshot, not from reading the
    file. The chunks are produced up front by ``extract_code_docs``, so
    ``Document.read`` is never called on one.
    """

    type: str = "code_doc"
    mime_type: str = "text/plain"
    repo: str
    file_path: str
    language: str | None = None
    # Documented symbols this document holds (a module doc not counted).
    symbol_count: int = 0
