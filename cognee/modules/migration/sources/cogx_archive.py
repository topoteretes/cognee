"""COGX archive source: re-import an archive produced by ``cognee.export()``.

This is the restore half of backup/restore and the receiving end of
Cognee-to-Cognee instance migration.
"""

from pathlib import Path
from typing import AsyncIterator, Union

from cognee.modules.migration.cogx import COGXRecord, read_archive, read_manifest
from cognee.modules.migration.sources.base import MemorySource


class COGXArchiveSource(MemorySource):
    """Stream records from a COGX archive directory.

    Defaults to ``mode="preserve"``: cognee-origin archives already carry an
    extracted graph (plus raw nodes for full fidelity), so the restore is
    zero-LLM — matching the default of ``cognee.push()`` and the remember
    router. Pass ``mode="hybrid"`` to additionally re-cognify the raw content,
    or ``mode="re-derive"`` for extraction-only imports.

    Raises ValueError when the archive was written by a newer major COGX
    version than this reader supports.
    """

    source_system = "cogx"

    def __init__(self, directory: Union[str, Path], mode: str = "preserve"):
        super().__init__(mode=mode)
        self.directory = Path(directory)
        # read_manifest validates the archive's cogx_version (raises ValueError
        # when the archive is ahead of this reader).
        manifest = read_manifest(self.directory)
        if manifest is not None:
            self.source_system = manifest.source_system

    async def records(self) -> AsyncIterator[COGXRecord]:
        for record in read_archive(self.directory):
            yield record
