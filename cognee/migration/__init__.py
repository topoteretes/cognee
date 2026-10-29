"""Public migration API: move memory into and out of Cognee.

Importing from other memory systems (pass any source to ``cognee.remember``)::

    from cognee.migration import Mem0Source, ZepSource, LettaSource

    await cognee.remember(Mem0Source("mem0_export.json"))
    await cognee.remember(ZepSource("graphiti_dump.json", mode="hybrid"))

Exporting (``cognee.export``)::

    snapshot = await cognee.export("main_dataset")     # GraphSnapshot: typed
    alice = snapshot.find(name="Alice")[0]             # real DataPoint objects
    result = await cognee.export("main_dataset", format="graphml")

Restoring / Cognee-to-Cognee migration::

    await cognee.export("main_dataset", format="cogx", destination="backup_cogx")
    await cognee.remember(COGXArchiveSource("backup_cogx"))
"""

from cognee.modules.migration import (
    COGX_VERSION,
    COGXArchiveSource,
    COGXDocument,
    COGXConfidence,
    COGXContributor,
    COGXEntity,
    COGXEvidence,
    COGXEpisode,
    COGXFact,
    COGXManifest,
    COGXMemory,
    COGXMemoryBlock,
    COGXOrigin,
    COGXRawNode,
    COGXRecord,
    COGXScope,
    COGXTurn,
    EXPORT_FORMATS,
    ExportResult,
    GraphEdge,
    GraphSnapshot,
    GraphitiSource,
    IMPORT_MODES,
    LettaSource,
    Mem0Source,
    MemorySource,
    ZepSource,
    build_snapshot,
    datapoint_registry,
    export_dataset,
    read_archive,
    read_manifest,
    rehydrate_node,
)

__all__ = [
    "COGX_VERSION",
    "EXPORT_FORMATS",
    "IMPORT_MODES",
    "COGXArchiveSource",
    "COGXConfidence",
    "COGXContributor",
    "COGXDocument",
    "COGXEntity",
    "COGXEpisode",
    "COGXEvidence",
    "COGXFact",
    "COGXManifest",
    "COGXMemory",
    "COGXMemoryBlock",
    "COGXOrigin",
    "COGXRawNode",
    "COGXRecord",
    "COGXScope",
    "COGXTurn",
    "ExportResult",
    "GraphEdge",
    "GraphSnapshot",
    "GraphitiSource",
    "LettaSource",
    "Mem0Source",
    "MemorySource",
    "ZepSource",
    "build_snapshot",
    "datapoint_registry",
    "export_dataset",
    "read_archive",
    "read_manifest",
    "rehydrate_node",
]
