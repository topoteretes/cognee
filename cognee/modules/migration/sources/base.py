"""Base class for memory-migration sources.

A :class:`MemorySource` adapts an external memory system (or an export file
produced by one) into a stream of CMIF records. ``cognee.remember()`` accepts
any MemorySource instance and routes it through the migration loader, so

    await cognee.remember(Mem0Source("mem0_export.json"))

works the same way as remembering text, with the provider specifics fully
encapsulated in the source object.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from cognee.modules.migration.cmif import CMIFRecord

IMPORT_MODES = ("re-derive", "preserve", "hybrid")


class MemorySource(ABC):
    """An async stream of CMIF records read from an external memory system.

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

    def __init__(self, mode: str = "re-derive"):
        if mode not in IMPORT_MODES:
            raise ValueError(f"Unknown import mode {mode!r}. Expected one of {IMPORT_MODES}.")
        self.mode = mode

    @abstractmethod
    def records(self) -> AsyncIterator[CMIFRecord]:
        """Yield CMIF records from the source."""
