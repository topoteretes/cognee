"""OMP memory source: import an Open Memory Protocol export.

Reads the memories of any OMP-conformant export — a ``manifest.json`` plus
``memories.jsonl`` directory, a folder of markdown memories with front
matter, a single ``.jsonl``/``.json``/``.md`` file, or already-parsed
objects — and yields them as COGX records::

    await cognee.remember(OMPSource("team_memory_export"))
    await cognee.remember(OMPSource("~/.claude/memory"), mode="re-derive")

Memories cognee itself exported (``cognee.export(format="omp")``) come back
as the record kinds they were; every other memory becomes a
:class:`COGXMemory` whose body is cognified in ``re-derive`` mode (the
default) or stored as is in ``preserve`` mode. Links and evidence the export
does not resolve are kept and marked dangling (draft 3.5.5); checksums are
verified and reported on the import receipt, never used to reject (3.5.7).
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from cognee.modules.migration.cogx import COGXRecord
from cognee.modules.migration.omp import (
    OMPManifest,
    iter_omp_memories,
    mark_dangling,
    read_omp_manifest,
    record_from_memory,
)
from cognee.modules.migration.sources.base import MemorySource


class OMPSource(MemorySource):
    """Stream the memories of an OMP export as COGX records.

    ``external_system`` on every record is the export's ``origin`` (its
    manifest), falling back to ``"omp"``: with the memory's identifier that
    is the global key (draft 5.3), so re-importing the same export is
    idempotent and the same identifier from two stores stays distinct.
    """

    source_system = "omp"

    def __init__(self, source: Any, mode: str = "re-derive", verify_checksums: bool = True):
        super().__init__(mode=mode)
        if isinstance(source, str):
            source = Path(source).expanduser()
        self._source = source
        self.manifest: OMPManifest | None = read_omp_manifest(source)
        self.origin: str | None = self.manifest.origin if self.manifest else None
        self.verify_checksums = verify_checksums
        self._integrity_reported = False

    async def records(self) -> AsyncIterator[COGXRecord]:
        # Pass one keeps only the identifiers, so dangling marks are decided
        # against the whole export without holding every memory in memory.
        identifiers = {memory.identifier for memory in iter_omp_memories(self._source)}
        external_system = self.origin or self.source_system
        for memory in iter_omp_memories(self._source):
            # Verify as received: the dangling marks below are the receiver's
            # additions and would change the hashed form.
            if self.verify_checksums and self.receipt is not None and not self._integrity_reported:
                self.receipt.integrity_check(memory.identifier, memory.verify_checksum())
            mark_dangling(memory, identifiers)
            yield record_from_memory(memory, external_system=external_system)
        # Replays (the streaming import reads a source several times) must
        # not count the same verification again.
        self._integrity_reported = True
