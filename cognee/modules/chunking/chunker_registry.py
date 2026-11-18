"""Chunkers a deployment adds by name, so ``DEFAULT_CHUNKER`` can select one.

``cognify(chunker=...)`` takes a class, which only Python callers can pass. A server
picks its default by name instead. Only registered names resolve: a name is never
treated as an import path.

Every call that names no chunker uses the default (``resolve_chunker``): ``cognify()``,
``remember()``, ``update()``, the CLI, the API, the task-list builders and cognee's own
pipelines. Register the chunker in every process that reads ``DEFAULT_CHUNKER``; an
unregistered name fails every write. A document cut by a registered chunker is also
updated by it.

For ``update()`` to replace only the chunks an edit touched, the chunker has to:

* declare its own ``chunker_id`` and stamp it on every chunk it yields
  (``DocumentChunk(chunker_id=self.chunker_id, ...)``). ``update()`` compares it with
  the id on the stored chunks, and rebuilds a document another chunker cut. A chunk
  yielded without it fails the run (``require_chunker_stamp``);
* cut without overlap or loss: its chunk texts, joined in order, are the document's
  text. A document whose chunks do not join back to its text is rebuilt in full on
  every update (``chunks_not_tiling``).

``update()`` runs the chunker over the edited region alone, from one chunk boundary
to another, with the ``max_chunk_tokens`` recorded on the chunks it replaces (the
current limit when none is recorded), so a cut should depend only on the text given.

Example: ``examples/guides/custom_chunker.py``.
"""

from cognee.exceptions import CogneeConfigurationError
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.TextChunker import TextChunker

registered_chunkers: dict[str, type[Chunker]] = {"TextChunker": TextChunker}


def register_chunker(name: str, chunker: type[Chunker]) -> None:
    """Make ``chunker`` selectable as ``name`` in ``DEFAULT_CHUNKER``.

    The class needs a ``chunker_id`` that no other class uses, registered under any
    name or built into cognee: it is how ``update()`` tells which chunker cut a stored
    document. A subclass that cuts differently from its parent declares a new one.
    Registering a class again under its name changes nothing; "TextChunker" always
    names TextChunker.
    """
    if not (isinstance(chunker, type) and issubclass(chunker, Chunker)):
        raise TypeError("A registered chunker must be a subclass of Chunker.")
    if not chunker.chunker_id:
        raise ValueError(f"{chunker.__name__} declares no chunker_id.")
    if name == "TextChunker" and chunker is not TextChunker:
        raise ValueError(
            '"TextChunker" names the built-in TextChunker; register under another name.'
        )
    for other in [*_builtin_chunkers(), *registered_chunkers.values()]:
        if other is not chunker and other.chunker_id == chunker.chunker_id:
            raise ValueError(
                f"{chunker.__name__} has the chunker_id of {other.__name__} "
                f"({chunker.chunker_id!r}); declare its own."
            )
    registered_chunkers[name] = chunker


def _builtin_chunkers() -> list[type[Chunker]]:
    """The chunkers cognee ships, whose ids stored documents may carry."""
    from cognee.modules.chunking.CsvChunker import CsvChunker
    from cognee.modules.chunking.JsonListChunker import JsonListChunker
    from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap

    chunkers = [TextChunker, CsvChunker, JsonListChunker, TextChunkerWithOverlap]
    try:
        from cognee.modules.chunking.LangchainChunker import LangchainChunker
    except ImportError:
        # Without langchain installed nothing in this process can subclass it.
        return chunkers
    return [*chunkers, LangchainChunker]


def resolve_chunker(chunker: type[Chunker] | None) -> type[Chunker]:
    """``chunker``, or the deployment's default (``get_default_chunker``) when None.

    Every public entry point that takes a chunker defaults to None and resolves it
    here, once, so a caller that names none gets the same chunker on every path. The
    helpers below an entry point take the resolved class and have no default.
    """
    return chunker if chunker is not None else get_default_chunker()


def stamps_every_chunk(chunker: type[Chunker]) -> bool:
    """Whether every chunk ``chunker`` cut is known to carry its ``chunker_id``.

    True for a chunker a deployment registered, which ``require_chunker_stamp`` holds
    to it. Not for TextChunker: chunks carry an id since cognee 1.5.4, so documents it
    cut before then have none, and it keeps updating them. Not for a chunker passed
    per call either, which may predate the id as well.
    """
    return chunker is not TextChunker and chunker in registered_chunkers.values()


def require_chunker_stamp(chunker: type[Chunker], chunk) -> None:
    """Raise when a registered chunker yields a chunk that does not carry its id.

    Only chunks being cut now are checked, never stored ones. Stored without the id,
    such a chunk could later be updated by another chunker with nothing to tell.
    """
    if stamps_every_chunk(chunker) and chunk.chunker_id != chunker.chunker_id:
        raise CogneeConfigurationError(
            message=f"{chunker.__name__} yielded a chunk with chunker_id "
            f"{chunk.chunker_id!r}, not its own ({chunker.chunker_id!r}).",
            remediation="Build every chunk with chunker_id=self.chunker_id.",
        )


def get_default_chunker() -> type[Chunker]:
    """The chunker a call uses when it names none (``DEFAULT_CHUNKER``, else TextChunker)."""
    from cognee.modules.cognify.config import get_cognify_config

    # A blank DEFAULT_CHUNKER= counts as unset.
    name = get_cognify_config().default_chunker or "TextChunker"
    if name not in registered_chunkers:
        raise CogneeConfigurationError(
            message=f"DEFAULT_CHUNKER names an unknown chunker {name!r}.",
            remediation="Register it with register_chunker() before the first cognify call, "
            f"or use one of: {', '.join(sorted(registered_chunkers))}.",
        )
    return registered_chunkers[name]
