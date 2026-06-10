"""Public migration API: move memory into and out of Cognee.

Importing from other memory systems (pass any source to ``cognee.remember``)::

    from cognee.migration import Mem0Source, ZepSource, LettaSource

    await cognee.remember(Mem0Source("mem0_export.json"))
    await cognee.remember(ZepSource("graphiti_dump.json", mode="hybrid"))

Exporting (``cognee.export``)::

    result = await cognee.export("main_dataset", format="graphml")

Restoring / Cognee-to-Cognee migration::

    await cognee.export("main_dataset", format="cmif", destination="backup_cmif")
    await cognee.remember(CMIFArchiveSource("backup_cmif"))
"""

from cognee.modules.migration import (
    CMIF_VERSION,
    CMIFArchiveSource,
    CMIFDocument,
    CMIFEntity,
    CMIFEpisode,
    CMIFFact,
    CMIFManifest,
    CMIFMemory,
    CMIFMemoryBlock,
    CMIFRecord,
    CMIFScope,
    CMIFTurn,
    EXPORT_FORMATS,
    ExportResult,
    GraphitiSource,
    IMPORT_MODES,
    LettaSource,
    Mem0Source,
    MemorySource,
    ZepSource,
    export_dataset,
    read_archive,
    read_manifest,
)

__all__ = [
    "CMIF_VERSION",
    "CMIFArchiveSource",
    "CMIFDocument",
    "CMIFEntity",
    "CMIFEpisode",
    "CMIFFact",
    "CMIFManifest",
    "CMIFMemory",
    "CMIFMemoryBlock",
    "CMIFRecord",
    "CMIFScope",
    "CMIFTurn",
    "EXPORT_FORMATS",
    "ExportResult",
    "GraphitiSource",
    "IMPORT_MODES",
    "LettaSource",
    "Mem0Source",
    "MemorySource",
    "ZepSource",
    "export_dataset",
    "read_archive",
    "read_manifest",
]
