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

Open Memory Protocol (draft v0.6) exchange with any conformant system::

    await cognee.export("main_dataset", format="omp", destination="main_omp")
    await cognee.remember(OMPSource("their_export"))        # jsonl, json or markdown
    result.receipt                                            # what happened per memory
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
    COGXLink,
    COGXManifest,
    COGXMemory,
    COGXMemoryBlock,
    COGXOrigin,
    COGXRawNode,
    COGXRecord,
    COGXScope,
    COGXTurn,
    COGXVersion,
    EXPORT_FORMATS,
    ExportResult,
    GraphEdge,
    GraphSnapshot,
    GraphitiSource,
    IMPORT_MODES,
    ImportReceipt,
    LangMemSource,
    LettaSource,
    Mem0Source,
    MemorySource,
    OMP_PROTOCOL_VERSION,
    OMPConformance,
    OMPManifest,
    OMPMemory,
    OMPSource,
    ZepSource,
    build_snapshot,
    datapoint_registry,
    export_dataset,
    iter_omp_memories,
    memory_from_record,
    read_archive,
    read_manifest,
    record_from_memory,
    rehydrate_node,
    write_omp_export,
)

__all__ = [
    "COGX_VERSION",
    "EXPORT_FORMATS",
    "IMPORT_MODES",
    "OMP_PROTOCOL_VERSION",
    "COGXArchiveSource",
    "COGXConfidence",
    "COGXContributor",
    "COGXDocument",
    "COGXEntity",
    "COGXEpisode",
    "COGXEvidence",
    "COGXFact",
    "COGXLink",
    "COGXManifest",
    "COGXMemory",
    "COGXMemoryBlock",
    "COGXOrigin",
    "COGXRawNode",
    "COGXRecord",
    "COGXScope",
    "COGXTurn",
    "COGXVersion",
    "ExportResult",
    "GraphEdge",
    "GraphSnapshot",
    "GraphitiSource",
    "ImportReceipt",
    "LangMemSource",
    "LettaSource",
    "Mem0Source",
    "MemorySource",
    "OMPConformance",
    "OMPManifest",
    "OMPMemory",
    "OMPSource",
    "ZepSource",
    "build_snapshot",
    "datapoint_registry",
    "export_dataset",
    "iter_omp_memories",
    "memory_from_record",
    "read_archive",
    "read_manifest",
    "record_from_memory",
    "rehydrate_node",
    "write_omp_export",
]
