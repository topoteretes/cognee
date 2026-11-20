"""COGX archive source: re-import an archive produced by ``cognee.export()``.

This is the restore half of backup/restore and the receiving end of
Cognee-to-Cognee instance migration.
"""

from collections.abc import AsyncIterator
from pathlib import Path

from cognee.modules.migration.cogx import (
    CANONICAL_CHECKSUM_SINCE,
    COGXManifest,
    COGXRecord,
    cogx_version_tuple,
    read_archive,
    read_manifest,
    read_social_layer,
    verify_checksum,
)
from cognee.modules.migration.sources.base import MemorySource


class COGXArchiveSource(MemorySource):
    """Stream records from a COGX archive directory.

    Defaults to ``mode="preserve"``: cognee-origin archives already carry an
    extracted graph (plus raw nodes for full fidelity), so the restore is
    zero-LLM — matching the default of ``cognee.push()`` and the remember
    router. Pass ``mode="hybrid"`` to additionally re-cognify the raw content,
    or ``mode="re-derive"`` for extraction-only imports.

    Record checksums are verified on the first read and reported on the
    import receipt (OMP draft 3.5.3); a mismatch is reported, never a
    rejection (3.5.7). Archives before COGX 0.3 hashed a field order this
    reader no longer reproduces, so their checksums are not checked.

    Raises ValueError when the archive was written by a newer major COGX
    version than this reader supports.
    """

    source_system = "cogx"

    def __init__(
        self, directory: str | Path, mode: str = "preserve", verify_checksums: bool = True
    ):
        super().__init__(mode=mode)
        self.directory = Path(directory)
        # read_manifest validates the archive's cogx_version (raises ValueError
        # when the archive is ahead of this reader).
        manifest = read_manifest(self.directory)
        self.manifest: COGXManifest | None = manifest
        if manifest is not None:
            self.source_system = manifest.source_system
            # Cognee-origin archives carry the source store's stamped
            # data-migration revision; the import re-stamps to it (see
            # MemorySource.migration_revision). Absent on older archives
            # and external-system exports -> None.
            self.migration_revision = manifest.migration_revision
        # Present only when exported with include_permissions=True.
        self.social_layer = read_social_layer(self.directory)
        self.verify_checksums = verify_checksums
        self._checkable = (
            manifest is not None
            and cogx_version_tuple(manifest.cogx_version) >= CANONICAL_CHECKSUM_SINCE
        )
        self._integrity_reported = False

    async def records(self) -> AsyncIterator[COGXRecord]:
        check = (
            self.verify_checksums
            and self._checkable
            and self.receipt is not None
            and not self._integrity_reported
        )
        for record in read_archive(self.directory):
            if check and record.kind != "raw_node":
                self.receipt.integrity_check(record.external_id, verify_checksum(record))
            yield record
        # The streaming import reads an archive several times; verify once.
        self._integrity_reported = True
