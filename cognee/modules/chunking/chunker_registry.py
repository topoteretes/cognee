"""Chunkers a deployment adds by name, so ``DEFAULT_CHUNKER`` can select one.

``cognify(chunker=...)`` takes a class, which only Python callers can pass. A server
picks its default by name instead. Only registered names resolve: a name is never
treated as an import path.
"""

from cognee.exceptions import CogneeConfigurationError
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.TextChunker import TextChunker

registered_chunkers: dict[str, type[Chunker]] = {"TextChunker": TextChunker}


def register_chunker(name: str, chunker: type[Chunker]) -> None:
    """Make ``chunker`` selectable as ``name`` in ``DEFAULT_CHUNKER``."""
    if not (isinstance(chunker, type) and issubclass(chunker, Chunker)):
        raise TypeError("A registered chunker must be a subclass of Chunker.")
    registered_chunkers[name] = chunker


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
