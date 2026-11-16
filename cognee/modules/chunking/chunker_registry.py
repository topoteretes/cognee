"""Chunkers by name.

``cognify(chunker=...)`` takes a class, which only Python callers can pass. A name
is what a request body or an environment variable can carry: ``POST /cognify``
accepts one in ``chunker``, and ``DEFAULT_CHUNKER`` names the chunker a server uses
when a call gives none. A deployment adds its own with ``register_chunker``, the way
``use_retriever`` adds a retriever.

Only registered names resolve. A name is never treated as an import path, so a
request cannot make the server load a class its operator did not register.
"""

from importlib import import_module

from cognee.exceptions import CogneeConfigurationError
from cognee.modules.chunking.Chunker import Chunker

# Module path per built-in chunker; the class has the module's name. Imported on
# first use, because LangchainChunker needs an optional dependency.
BUILT_IN_CHUNKERS = {
    "TextChunker": "cognee.modules.chunking.TextChunker",
    "LangchainChunker": "cognee.modules.chunking.LangchainChunker",
    "CsvChunker": "cognee.modules.chunking.CsvChunker",
    "JsonListChunker": "cognee.modules.chunking.JsonListChunker",
}

registered_chunkers: dict[str, type[Chunker]] = {}


def register_chunker(name: str, chunker: type[Chunker]) -> None:
    """Make ``chunker`` selectable as ``name``. A built-in name cannot be replaced."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A chunker name must be a non-empty string.")
    if name in BUILT_IN_CHUNKERS:
        raise ValueError(f"{name!r} is a built-in chunker and cannot be replaced.")
    if not (isinstance(chunker, type) and issubclass(chunker, Chunker)):
        raise TypeError("A registered chunker must be a subclass of Chunker.")
    registered_chunkers[name] = chunker


def chunker_names() -> list[str]:
    """Every name ``get_chunker`` resolves, sorted."""
    return sorted({*BUILT_IN_CHUNKERS, *registered_chunkers})


def validate_chunker_name(name: str) -> str:
    """``name`` if it is registered; raises without importing the chunker."""
    if name not in registered_chunkers and name not in BUILT_IN_CHUNKERS:
        raise ValueError(f"Unknown chunker {name!r}. Available: {', '.join(chunker_names())}.")
    return name


def get_chunker(name: str) -> type[Chunker]:
    """The chunker class registered as ``name``."""
    validate_chunker_name(name)
    if name in registered_chunkers:
        return registered_chunkers[name]
    return getattr(import_module(BUILT_IN_CHUNKERS[name]), name)


def get_default_chunker() -> type[Chunker]:
    """The chunker a call uses when it names none (``DEFAULT_CHUNKER``, else TextChunker)."""
    from cognee.modules.cognify.config import get_cognify_config

    name = get_cognify_config().default_chunker
    try:
        return get_chunker(name)
    except ValueError as error:
        raise CogneeConfigurationError(
            message=f"DEFAULT_CHUNKER is set to an unknown chunker. {error}",
            remediation="Set DEFAULT_CHUNKER to an available name, or register the chunker "
            "with register_chunker() before the first cognify call.",
        ) from error
