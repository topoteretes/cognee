"""Base class for memory-migration sources.

A :class:`MemorySource` adapts an external memory system (or an export file
produced by one) into a stream of COGX records. ``cognee.remember()`` accepts
any MemorySource instance and routes it through the migration loader, so

    await cognee.remember(Mem0Source("mem0_export.json"))

works the same way as remembering text, with the provider specifics fully
encapsulated in the source object.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from cognee.modules.migration.cogx import COGXRecord

IMPORT_MODES = ("re-derive", "preserve", "hybrid")


class MemorySource(ABC):
    """An async stream of COGX records read from an external memory system.

    Args:
        mode: Import fidelity mode.
            - ``"re-derive"`` (default): ingest raw content (episodes,
              documents, memories) and run Cognee's own extraction (cognify).
              Costs LLM tokens; the source system's derived graph is ignored.
            - ``"preserve"``: map the source's already-extracted entities and
              facts directly into the graph with zero LLM calls; raw content
              is stored as data items but not cognified.
            - ``"hybrid"``: preserve the source graph and cognify raw content.
    """

    source_system: str = "unknown"

    # Whether ``records()`` may be called more than once, each call returning
    # a fresh iterator over the same records. True for all built-in sources
    # (they re-read a file on each call) and required for the streaming
    # preserve-mode import, which passes over the records twice (nodes, then
    # facts) to keep memory bounded. Set False on one-shot sources (e.g. live
    # API cursors) to force the buffered import path instead.
    replayable: bool = True

    def __init__(self, mode: str = "re-derive"):
        if mode not in IMPORT_MODES:
            raise ValueError(f"Unknown import mode {mode!r}. Expected one of {IMPORT_MODES}.")
        self.mode = mode

    @abstractmethod
    def records(self) -> AsyncIterator[COGXRecord]:
        """Yield COGX records from the source."""
