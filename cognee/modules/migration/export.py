"""Export a Cognee dataset's knowledge graph to portable formats.

Formats:

- ``pydantic`` — a :class:`GraphSnapshot` of typed DataPoint instances (the
  Pydantic-native export: real ``Entity``/``DocumentChunk``/custom-model
  objects, losslessly serializable via ``model_dump_json``)
- ``cogx``    — a COGX archive directory (the canonical portable dump;
  re-importable via :class:`COGXArchiveSource`)
- ``json``    — full-fidelity nodes/edges JSON
- ``graphml`` — Gephi/yEd/NetworkX interop
- ``cypher`` — MERGE script loadable into any Neo4j-compatible database
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from cognee.modules.migration.cogx import (
    COGXArchiveWriter,
    COGXDocument,
    COGXEntity,
    COGXFact,
    parse_timestamp,
)
from cognee.modules.migration.formats import write_cypher, write_graphml, write_json
from cognee.modules.migration.snapshot import GraphSnapshot, build_snapshot
from cognee.shared.logging_utils import get_logger

logger = get_logger("migration.export")

EXPORT_FORMATS = ("pydantic", "cogx", "json", "graphml", "cypher")

_FORMAT_SUFFIX = {"json": ".json", "graphml": ".graphml", "cypher": ".cypher"}


@dataclass
class ExportResult:
    format: str
    destination: str
    dataset_name: str
    dataset_id: str
    num_nodes: int
    num_edges: int

    def __repr__(self):
        return (
            f"ExportResult(format={self.format!r}, destination={self.destination!r}, "
            f"dataset={self.dataset_name!r}, nodes={self.num_nodes}, edges={self.num_edges})"
        )


def _write_cogx(
    nodes, edges, destination: Path, dataset_name: str, embedding_model: Optional[str] = None
) -> None:
    """Map graph nodes/edges onto typed COGX records; keep the rest raw."""
    with COGXArchiveWriter(destination, source_system="cognee") as writer:
        writer.add_note(f"Exported from Cognee dataset {dataset_name!r}.")
        writer.embedding_model = embedding_model
        for node_id, properties in nodes:
            properties = properties or {}
            node_type = properties.get("type")
            if node_type == "Entity" and properties.get("name"):
                writer.write(
                    COGXEntity(
                        external_system="cognee",
                        external_id=str(node_id),
                        name=properties["name"],
                        description=properties.get("description"),
                        created_at=parse_timestamp(properties.get("created_at")),
                        updated_at=parse_timestamp(properties.get("updated_at")),
                    )
                )
            elif node_type == "DocumentChunk" and properties.get("text"):
                writer.write(
                    COGXDocument(
                        external_system="cognee",
                        external_id=str(node_id),
                        content=properties["text"],
                        created_at=parse_timestamp(properties.get("created_at")),
                    )
                )
                # Also persist the chunk as a raw node: preserve-mode restore
                # rehydrates it as a graph node so facts referencing the chunk
                # (e.g. DocumentChunk -contains-> Entity) keep their topology
                # instead of dangling. The COGXDocument record above carries
                # the chunk's content for re-derive/cross-provider imports.
                writer.write_raw_node({"id": str(node_id), **properties})
            else:
                writer.write_raw_node({"id": str(node_id), **properties})

        for source, target, relationship, properties in edges:
            properties = properties or {}
            writer.write(
                COGXFact(
                    external_system="cognee",
                    external_id=f"{source}:{relationship}:{target}",
                    subject_ref=str(source),
                    predicate=str(relationship),
                    object_ref=str(target),
                    fact_text=properties.get("edge_text"),
                    valid_at=parse_timestamp(properties.get("valid_at")),
                    invalid_at=parse_timestamp(properties.get("invalid_at")),
                )
            )


async def export_dataset(
    dataset: Union[str, UUID] = "main_dataset",
    format: str = "pydantic",
    destination: Optional[Union[str, Path]] = None,
    user=None,
    link_relations: bool = False,
) -> Union[ExportResult, GraphSnapshot]:
    """Export an authorized dataset's graph. Requires read permission.

    ``format="pydantic"`` returns a :class:`GraphSnapshot` (typed DataPoint
    instances, in memory; pass ``destination`` to also persist it as JSON).
    All other formats write a file and return an :class:`ExportResult`.
    ``link_relations`` (pydantic only) re-attaches edges as object references
    on declared relation fields, e.g. ``entity.is_a``.
    """
    if format not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format {format!r}. Expected one of {EXPORT_FORMATS}.")

    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.users.methods import get_default_user

    if user is None:
        user = await get_default_user()

    datasets = await get_authorized_existing_datasets([dataset], "read", user)
    if not datasets:
        raise DatasetNotFoundError(message=f"Dataset not found or not readable: {dataset}")
    dataset_obj = datasets[0]

    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()

    # Ladybug's get_graph_data() synthesizes "SELF" self-loops when a graph
    # has no edges; they are not real relationships, so no emitter sees them.
    edges = [edge for edge in edges if not (edge[2] == "SELF" and edge[0] == edge[1])]

    if format == "pydantic":
        snapshot = build_snapshot(
            nodes,
            edges,
            dataset_name=dataset_obj.name,
            dataset_id=str(dataset_obj.id),
            link_relations=link_relations,
        )
        if destination is not None:
            snapshot.save(destination)
        logger.info(
            "Exported dataset %s as GraphSnapshot: %d nodes, %d edges",
            dataset_obj.name,
            len(snapshot.nodes),
            len(snapshot.edges),
        )
        return snapshot

    if destination is None:
        suffix = _FORMAT_SUFFIX.get(format, "")
        destination = f"{dataset_obj.name}_export{suffix}" if suffix else f"{dataset_obj.name}_cogx"
    destination = Path(destination)

    if format == "cogx":
        embedding_model = None
        try:
            from cognee.infrastructure.databases.vector.embeddings.config import (
                get_embedding_context_config,
            )

            embedding_model = get_embedding_context_config().embedding_model
        except Exception as error:  # noqa: BLE001 — manifest metadata is best effort
            logger.debug("Could not determine embedding model for manifest: %s", error)
        _write_cogx(nodes, edges, destination, dataset_obj.name, embedding_model=embedding_model)
    elif format == "json":
        write_json(nodes, edges, destination)
    elif format == "graphml":
        write_graphml(nodes, edges, destination)
    elif format == "cypher":
        write_cypher(nodes, edges, destination)

    logger.info(
        "Exported dataset %s: %d nodes, %d edges -> %s (%s)",
        dataset_obj.name,
        len(nodes),
        len(edges),
        destination,
        format,
    )

    return ExportResult(
        format=format,
        destination=str(destination),
        dataset_name=dataset_obj.name,
        dataset_id=str(dataset_obj.id),
        num_nodes=len(nodes),
        num_edges=len(edges),
    )
