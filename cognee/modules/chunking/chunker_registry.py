"""Chunkers a deployment adds by name, so ``DEFAULT_CHUNKER`` can select one.

``cognify(chunker=...)`` takes a class, which only Python callers can pass. A server
picks its default by name instead. Only registered names resolve: a name is never
treated as an import path.

``cognify()``, ``remember()`` and ``update()`` all use the default, so a document cut
by a registered chunker is also updated by it. For ``update()`` to replace only the
chunks an edit touched, the chunker has to:

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

    The class needs a ``chunker_id`` no other registered chunker uses: it is how
    ``update()`` tells which chunker cut a stored document. A subclass that cuts
    differently from its parent declares a new one.
    """
    if not (isinstance(chunker, type) and issubclass(chunker, Chunker)):
        raise TypeError("A registered chunker must be a subclass of Chunker.")
    if not chunker.chunker_id:
        raise ValueError(f"{chunker.__name__} declares no chunker_id.")
    for other_name, other in registered_chunkers.items():
        if other_name != name and other is not chunker and other.chunker_id == chunker.chunker_id:
            raise ValueError(
                f"{chunker.__name__} has the chunker_id of {other.__name__} "
                f"({chunker.chunker_id!r}); declare its own."
            )
    registered_chunkers[name] = chunker


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

    name = get_cognify_config().default_chunker
    if name not in registered_chunkers:
        raise CogneeConfigurationError(
            message=f"DEFAULT_CHUNKER names an unknown chunker {name!r}.",
            remediation="Register it with register_chunker() before the first cognify call, "
            f"or use one of: {', '.join(sorted(registered_chunkers))}.",
        )
    return registered_chunkers[name]
