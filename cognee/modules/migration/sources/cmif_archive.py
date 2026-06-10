"""CMIF archive source: re-import an archive produced by ``cognee.export()``.

This is the restore half of backup/restore and the receiving end of
Cognee-to-Cognee instance migration.
"""

from pathlib import Path
from typing import AsyncIterator, Union

from cognee.modules.migration.cmif import CMIFRecord, read_archive, read_manifest
from cognee.modules.migration.sources.base import MemorySource


class CMIFArchiveSource(MemorySource):
    source_system = "cmif"

    def __init__(self, directory: Union[str, Path], mode: str = "hybrid"):
        super().__init__(mode=mode)
        self.directory = Path(directory)
        manifest = read_manifest(self.directory)
        if manifest is not None:
            self.source_system = manifest.source_system

    async def records(self) -> AsyncIterator[CMIFRecord]:
        for record in read_archive(self.directory):
            yield record
