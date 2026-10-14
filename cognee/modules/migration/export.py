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
from typing import Any
from uuid import UUID

from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.migration.cogx import (
    COGXArchiveWriter,
    COGXConfidence,
    COGXContributor,
    COGXDocument,
    COGXEntity,
    COGXEvidence,
    COGXFact,
    COGXOrigin,
    header_from_properties,
    parse_timestamp,
    write_social_layer,
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


def _user_payload(user_row) -> dict[str, Any]:
    return {
        "email": user_row.email,
        "hashed_password": user_row.hashed_password,
        "is_active": user_row.is_active,
        "is_superuser": user_row.is_superuser,
        "is_verified": user_row.is_verified,
    }


async def _gather_social_layer(dataset_obj) -> dict[str, Any]:
    """The dataset's owner and ACL grants, including user credentials.

    Emails, password hashes, and account flags are included so a
    cognee-to-cognee restore can recreate functional accounts — which makes
    an archive carrying this payload a SECRET. Only gathered on explicit
    request (``include_permissions=True``).
    """
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import ACL, Permission, User

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        owner = await session.get(User, dataset_obj.owner_id)
        rows = (
            await session.execute(
                select(User, Permission.name)
                .join(ACL, ACL.principal_id == User.id)
                .join(Permission, Permission.id == ACL.permission_id)
                .where(ACL.dataset_id == dataset_obj.id)
            )
        ).all()

    grants: dict[str, dict[str, Any]] = {}
    for user_row, permission_name in rows:
        grant = grants.setdefault(
            user_row.email, {"user": _user_payload(user_row), "permissions": []}
        )
        grant["permissions"].append(permission_name)
    return {
        "owner": _user_payload(owner) if owner else None,
        "grants": sorted(grants.values(), key=lambda g: g["user"]["email"]),
    }


async def _stored_migration_revision(dataset_id) -> str | None:
    """The source store's stamped data-migration revision, for the manifest.

    Per-dataset bookkeeping with backend access control on, the global row
    with it off. None (no row / never migrated / bookkeeping unreadable)
    means "unknown" — the importer then keeps its own stamp.
    """
    from cognee.context_global_variables import backend_access_control_enabled
    from cognee.infrastructure.databases.relational import get_relational_engine

    try:
        db_engine = get_relational_engine()
        if backend_access_control_enabled():
            from cognee.modules.users.models import DatasetDatabase

            async with db_engine.get_async_session() as session:
                record = await session.get(DatasetDatabase, dataset_id)
            return record.migration_revision if record else None

        from cognee.modules.migrations.models import (
            GLOBAL_DATABASE_VERSION_ROW_ID,
            GlobalDatabaseVersion,
        )

        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        return record.global_migration_revision if record else None
    except Exception as error:  # — manifest metadata is best effort
        logger.debug(
            "Could not determine migration revision for manifest: %s", error, exc_info=True
        )
        return None


# What the cognee exporter can state for each OMP header field. Declared in
# the manifest so an importer never mistakes an absent field for a negative.
COGX_EXPORT_COVERAGE = {
    "origin": "cognee node ids; an earlier origin on the node is kept",
    "content_hash": "sha256 of each record body",
    "subject": "entities linked to a chunk by contains edges; entity records name themselves",
    "semantic_kind": "node property only; no classification is performed",
    "validity": "valid_from/valid_to and invalidated_by read from node and edge properties when present",
    "evidence": "derived: the chunks whose contains edges reach the record; no source spans",
    "confidence": "score from the edge confidence property; contradicted for endpoints of contradicts edges; otherwise not stated",
    "version": "DataPoint version from node properties",
    "content_type": "text/plain for every body; documents keep their mime type",
    "checksum": "sha256 over each exported record, header and body",
    "tags": "node property only",
    "contributors": "pipeline, task and user stamps on the node (source_pipeline, source_task, source_user)",
    "accountable": "the user stamped on the node (source_user); otherwise not stated",
    "ai_involvement": "full for extracted entities and facts; none for source chunks",
}


def _contributors(properties: dict, *, extracted: bool) -> dict:
    """Provenance list and AI-involvement from a node's pipeline stamps."""
    header: dict = {}
    contributors = []
    user = properties.get("source_user")
    if user:
        contributors.append(COGXContributor(role="initiator", actor=str(user), kind="human"))
        header["accountable"] = str(user)
    pipeline = properties.get("source_pipeline") or properties.get("source_task")
    if pipeline:
        contributors.append(
            COGXContributor(role="system", actor=f"cognee:{pipeline}", kind="agent")
        )
    if contributors:
        header["contributors"] = contributors
    header["ai_involvement"] = "full" if extracted else "none"
    return header


def _header_kwargs(properties: dict, node_id, *, default_origin: bool = True) -> dict:
    """OMP header fields for one node or edge, from stored properties."""
    header = header_from_properties(properties)
    if default_origin and "origin" not in header:
        header["origin"] = COGXOrigin(system="cognee", id=str(node_id))
    return header


def _write_cogx(
    nodes,
    edges,
    destination: Path,
    dataset_name: str,
    embedding_model: str | None = None,
    migration_revision: str | None = None,
) -> None:
    """Map graph nodes/edges onto typed COGX records; keep the rest raw."""
    node_types = {str(node_id): (properties or {}).get("type") for node_id, properties in nodes}
    entity_names = {
        str(node_id): (properties or {}).get("name")
        for node_id, properties in nodes
        if (properties or {}).get("type") == "Entity" and (properties or {}).get("name")
    }
    # Topology the header is read from: which chunk contains which entity,
    # and which nodes a contradiction pass flagged.
    entities_in_chunk: dict[str, list[str]] = {}
    chunks_of_entity: dict[str, list[str]] = {}
    contradicted: set[str] = set()
    for source, target, relationship, _properties in edges:
        source, target = str(source), str(target)
        if relationship == "contains" and node_types.get(source) == "DocumentChunk":
            entities_in_chunk.setdefault(source, []).append(target)
            chunks_of_entity.setdefault(target, []).append(source)
        elif relationship == "contradicts":
            contradicted.update((source, target))

    def confidence_for(node_id: str, header: dict) -> COGXConfidence | None:
        if header.get("confidence") is not None:
            return header["confidence"]
        if node_id in contradicted:
            return COGXConfidence(outcome="contradicted", verified_by="automatic")
        return None

    with COGXArchiveWriter(destination, source_system="cognee") as writer:
        writer.add_note(f"Exported from Cognee dataset {dataset_name!r}.")
        writer.embedding_model = embedding_model
        writer.migration_revision = migration_revision
        writer.coverage = dict(COGX_EXPORT_COVERAGE)
        for node_id, properties in nodes:
            properties = properties or {}
            node_id = str(node_id)
            node_type = properties.get("type")
            header = _header_kwargs(properties, node_id)
            if isinstance(properties.get("version"), int) and "version" not in header:
                header["version"] = properties["version"]
            for key, value in _contributors(
                properties, extracted=node_type != "DocumentChunk"
            ).items():
                header.setdefault(key, value)
            if node_type == "Entity" and properties.get("name"):
                header.setdefault("subject", [properties["name"]])
                header.setdefault(
                    "evidence",
                    [
                        COGXEvidence(role="derived", ref=chunk_id)
                        for chunk_id in chunks_of_entity.get(node_id, [])
                    ],
                )
                writer.write(
                    COGXEntity(
                        external_system="cognee",
                        external_id=node_id,
                        name=properties["name"],
                        description=properties.get("description"),
                        created_at=parse_timestamp(properties.get("created_at")),
                        updated_at=parse_timestamp(properties.get("updated_at")),
                        confidence=confidence_for(node_id, header),
                        **{k: v for k, v in header.items() if k != "confidence"},
                    )
                )
            elif node_type == "DocumentChunk" and properties.get("text"):
                header.setdefault(
                    "subject",
                    [
                        entity_names[entity_id]
                        for entity_id in entities_in_chunk.get(node_id, [])
                        if entity_id in entity_names
                    ],
                )
                writer.write(
                    COGXDocument(
                        external_system="cognee",
                        external_id=node_id,
                        content=properties["text"],
                        created_at=parse_timestamp(properties.get("created_at")),
                        confidence=confidence_for(node_id, header),
                        **{k: v for k, v in header.items() if k != "confidence"},
                    )
                )
                # Also persist the chunk as a raw node: preserve-mode restore
                # rehydrates it as a graph node so facts referencing the chunk
                # (e.g. DocumentChunk -contains-> Entity) keep their topology
                # instead of dangling. The COGXDocument record above carries
                # the chunk's content for re-derive/cross-provider imports.
                writer.write_raw_node({"id": node_id, **properties})
            else:
                writer.write_raw_node({"id": node_id, **properties})

        for source, target, relationship, properties in edges:
            properties = properties or {}
            source, target = str(source), str(target)
            external_id = f"{source}:{relationship}:{target}"
            header = _header_kwargs(properties, external_id)
            for key, value in _contributors(properties, extracted=True).items():
                header.setdefault(key, value)
            if "evidence" not in header and relationship not in {"contains", "contradicts"}:
                # A fact between two entities is derived from the chunks that
                # mention both; fall back to chunks mentioning either side.
                both = [
                    c
                    for c in chunks_of_entity.get(source, [])
                    if c in chunks_of_entity.get(target, [])
                ]
                either = both or list(
                    dict.fromkeys(
                        chunks_of_entity.get(source, []) + chunks_of_entity.get(target, [])
                    )
                )
                if either:
                    header["evidence"] = [
                        COGXEvidence(
                            role="derived",
                            ref=chunk_id,
                            confidence=properties.get("confidence")
                            if isinstance(properties.get("confidence"), (int, float))
                            else None,
                        )
                        for chunk_id in either
                    ]
            header.setdefault("subject", [source, target])
            writer.write(
                COGXFact(
                    external_system="cognee",
                    external_id=external_id,
                    subject_ref=source,
                    predicate=str(relationship),
                    object_ref=target,
                    fact_text=properties.get("edge_text"),
                    valid_at=parse_timestamp(properties.get("valid_at")),
                    invalid_at=parse_timestamp(properties.get("invalid_at")),
                    **header,
                )
            )


async def export_dataset(
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    format: str = "pydantic",
    destination: str | Path | None = None,
    user=None,
    link_relations: bool = False,
    include_permissions: bool = False,
) -> ExportResult | GraphSnapshot:
    """Export an authorized dataset's graph. Requires read permission.

    ``include_permissions`` (cogx only): additionally write the dataset's
    social layer — owner and ACL grants WITH user credentials — into
    ``permissions.json`` so a cognee-to-cognee restore recreates functional
    accounts and grants. The archive then contains password hashes: treat it
    as a secret.

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
        except Exception as error:  # — manifest metadata is best effort
            logger.debug(
                "Could not determine embedding model for manifest: %s", error, exc_info=True
            )
        _write_cogx(
            nodes,
            edges,
            destination,
            dataset_obj.name,
            embedding_model=embedding_model,
            migration_revision=await _stored_migration_revision(dataset_obj.id),
        )
        if include_permissions:
            write_social_layer(destination, await _gather_social_layer(dataset_obj))
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
