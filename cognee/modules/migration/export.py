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

import json
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
    header_from_graph_properties,
    parse_omp_header,
    parse_timestamp,
    write_social_layer,
)
from cognee.modules.migration.formats import write_cypher, write_graphml, write_json
from cognee.modules.migration.snapshot import GraphSnapshot, build_snapshot
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME

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
    "origin": "the origin a memory was imported with (a chunk takes its document's); otherwise the cognee node or edge id",
    "content_hash": "sha256 of each record body",
    "subject": "names of the entities a chunk contains, or a fact's endpoints; entity records name themselves",
    "semantic_kind": "node property only; no classification is performed",
    "validity": "valid_from/valid_to and invalidated_by read from node and edge properties when present",
    "evidence": "facts: the chunks the edge evidence records (inferred if extracted, observed if structural); entities: the chunks containing them; no source spans",
    "confidence": "facts: the conflict review verdict (contradicted, corrected, confirmed) with the edge's confidence score; otherwise as imported, or the score alone",
    "version": "DataPoint version from node properties",
    "content_type": "text/plain for every body; documents keep their mime type",
    "checksum": "sha256 over each exported record, header and body",
    "tags": "node property only",
    "contributors": "pipeline, task and user stamps on the node (source_pipeline, source_task, source_user); the user's kind is human or agent",
    "accountable": "the stamped user, or the owner of a stamped agent; otherwise not stated",
    "ai_involvement": "entities: full if cognify extracted them; chunks: none; facts: from their recorded evidence; otherwise as imported or not stated",
}


def _contributors(properties: dict, actors: dict[str, "Actor"]) -> dict:
    """Provenance list and accountable human from a node's pipeline stamps."""
    header: dict = {}
    contributors = []
    label = properties.get("source_user")
    if label:
        actor = actors.get(str(label))
        contributors.append(
            COGXContributor(
                role="initiator", actor=str(label), kind=actor.kind if actor else "unknown"
            )
        )
        if actor is not None:
            header["accountable"] = actor.accountable
    pipeline = properties.get("source_pipeline") or properties.get("source_task")
    if pipeline:
        contributors.append(
            COGXContributor(role="system", actor=f"cognee:{pipeline}", kind="agent")
        )
    if contributors:
        header["contributors"] = contributors
    return header


def _element_header(
    properties: dict, element_id: str, actors: dict[str, "Actor"], imported: dict
) -> dict:
    """OMP header for one node or edge.

    ``imported`` is the header the memory arrived with (its own ``omp_header``,
    or for a chunk its document's) and wins; cognee's own properties and
    pipeline stamps fill the rest, and the origin defaults to cognee itself.
    """
    header = dict(imported)
    derived = {**header_from_graph_properties(properties), **_contributors(properties, actors)}
    for key, value in derived.items():
        header.setdefault(key, value)
    header.setdefault("origin", COGXOrigin(system="cognee", id=element_id))
    return header


@dataclass(frozen=True)
class EdgeSupport:
    """One chunk the edge evidence sidecar records as supporting an edge."""

    chunk_id: str
    kind: str  # "extracted" by the LLM, or "structural" (built without one)
    confidence: float | None


@dataclass(frozen=True)
class Actor:
    """The user a ``source_user`` stamp names: a human, or an agent whose
    owning human is the accountable one."""

    kind: str  # "human" or "agent"
    accountable: str


@dataclass
class ExportFacts:
    """What the COGX export reads outside the graph, loaded before writing."""

    # Recorded edge evidence, by fact external id (``source:relationship:target``).
    edge_support: dict[str, list["EdgeSupport"]]
    # Who each ``source_user`` stamp names, by the stamped label.
    actors: dict[str, "Actor"]


async def _load_export_facts(dataset_id: UUID, nodes, edges) -> ExportFacts:
    """Load the edge evidence and the stamped users a COGX export states."""
    return ExportFacts(
        edge_support=await _load_edge_support(dataset_id, edges),
        actors=await _load_actors(nodes),
    )


async def _load_edge_support(dataset_id: UUID, edges) -> dict[str, list[EdgeSupport]]:
    """The recorded evidence of every edge, keyed by its fact external id."""
    from cognee.modules.engine.utils import generate_edge_object_id
    from cognee.modules.provenance.edge_evidence.lookup import get_edge_support

    fact_ids = {
        UUID(generate_edge_object_id(source, target, relationship)): (
            f"{source}:{relationship}:{target}"
        )
        for source, target, relationship, _properties in edges
    }
    support: dict[str, list[EdgeSupport]] = {}
    for record in await get_edge_support(dataset_id, fact_ids):
        support.setdefault(fact_ids[record.edge_id], []).append(
            EdgeSupport(str(record.chunk_id), record.evidence_kind, record.confidence)
        )
    return support


async def _load_actors(nodes) -> dict[str, Actor]:
    """Resolve the users named by the nodes' ``source_user`` stamps (an email,
    or the user id when it has none). Labels matching no user are left out."""
    from sqlalchemy import or_, select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    labels = {str(p["source_user"]) for _id, p in nodes if p and p.get("source_user")}
    if not labels:
        return {}
    ids = [UUID(label) for label in labels if _is_uuid(label)]
    async with get_relational_engine().get_async_session() as session:
        users = (
            await session.scalars(select(User).where(or_(User.email.in_(labels), User.id.in_(ids))))
        ).all()
        owner_ids = {user.parent_user_id for user in users if user.parent_user_id}
        owners = {
            owner.id: owner
            for owner in await session.scalars(select(User).where(User.id.in_(owner_ids)))
        }
    actors: dict[str, Actor] = {}
    for user in users:
        actor = _actor_for(user, owners)
        actors[user.email] = actor
        actors[str(user.id)] = actor
    return actors


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _actor_for(user, owners: dict) -> Actor:
    """A human answers for itself; an agent's owning human answers for it."""
    if user.parent_user_id is None:
        return Actor(kind="human", accountable=user.email)
    owner = owners.get(user.parent_user_id)
    return Actor(kind="agent", accountable=owner.email if owner else str(user.parent_user_id))


@dataclass
class _Topology:
    """What a record's header reads from the rest of the graph."""

    names: dict[str, str]  # node id -> name, for fact subjects
    entities_in_chunk: dict[str, list[str]]  # chunk id -> ids of the entities it contains
    chunks_of_entity: dict[str, list[str]]  # entity id -> ids of the chunks containing it
    chunk_headers: dict[str, dict]  # chunk id -> the header its document was imported with

    @classmethod
    def of(cls, nodes, edges) -> "_Topology":
        by_id = {str(node_id): properties or {} for node_id, properties in nodes}
        names = {node_id: str(p["name"]) for node_id, p in by_id.items() if p.get("name")}
        topology = cls(names=names, entities_in_chunk={}, chunks_of_entity={}, chunk_headers={})
        for source, target, relationship, _properties in edges:
            source, target = str(source), str(target)
            if by_id.get(source, {}).get("type") != "DocumentChunk":
                continue
            if relationship == "contains" and by_id.get(target, {}).get("type") == "Entity":
                topology.entities_in_chunk.setdefault(source, []).append(target)
                topology.chunks_of_entity.setdefault(target, []).append(source)
            elif relationship == "is_part_of":
                topology.chunk_headers[source] = _document_header(by_id.get(target, {}))
        return topology


def _entity_record(
    node_id: str, properties: dict, topology: _Topology, facts: ExportFacts
) -> COGXEntity:
    imported = parse_omp_header(properties.get("omp_header"))
    header = _element_header(properties, node_id, facts.actors, imported)
    header.setdefault("subject", [properties["name"]])
    header.setdefault(
        "evidence",
        [
            COGXEvidence(role="derived", ref=chunk_id)
            for chunk_id in topology.chunks_of_entity.get(node_id, [])
        ],
    )
    if properties.get("source_pipeline") == COGNIFY_PIPELINE_NAME:
        header.setdefault("ai_involvement", "full")  # extracted by a model
    return COGXEntity(
        external_system="cognee",
        external_id=node_id,
        name=properties["name"],
        description=properties.get("description"),
        created_at=parse_timestamp(properties.get("created_at")),
        updated_at=parse_timestamp(properties.get("updated_at")),
        **header,
    )


def _chunk_record(
    node_id: str, properties: dict, topology: _Topology, facts: ExportFacts
) -> COGXDocument:
    imported = topology.chunk_headers.get(node_id, {})
    header = _element_header(properties, node_id, facts.actors, imported)
    header.setdefault(
        "subject",
        [topology.names[entity_id] for entity_id in topology.entities_in_chunk.get(node_id, [])],
    )
    header.setdefault("ai_involvement", "none")  # source text; chunking runs no model
    return COGXDocument(
        external_system="cognee",
        external_id=node_id,
        content=properties["text"],
        created_at=parse_timestamp(properties.get("created_at")),
        **header,
    )


_EVIDENCE_ROLES = {"extracted": "inferred", "structural": "observed"}


def _fact_record(
    source: str,
    target: str,
    relationship: str,
    properties: dict,
    topology: _Topology,
    facts: ExportFacts,
) -> COGXFact:
    external_id = f"{source}:{relationship}:{target}"
    imported = parse_omp_header(properties.get("omp_header"))
    header = _element_header(properties, external_id, facts.actors, imported)
    support = facts.edge_support.get(external_id, [])
    header.setdefault(
        "evidence",
        [
            COGXEvidence(role=_EVIDENCE_ROLES[s.kind], ref=s.chunk_id, confidence=s.confidence)
            for s in support
            if s.kind in _EVIDENCE_ROLES
        ],
    )
    involvement = _ai_involvement(support)
    if involvement is not None:
        header.setdefault("ai_involvement", involvement)
    header.setdefault(
        "subject", [topology.names.get(source, source), topology.names.get(target, target)]
    )
    reviewed = _reviewed_confidence(properties)
    if reviewed is not None:
        header["confidence"] = reviewed  # a review is newer than anything imported
    return COGXFact(
        external_system="cognee",
        external_id=external_id,
        subject_ref=source,
        predicate=relationship,
        object_ref=target,
        fact_text=properties.get("edge_text"),
        valid_at=parse_timestamp(properties.get("valid_at")),
        invalid_at=parse_timestamp(properties.get("invalid_at")),
        **header,
    )


def _ai_involvement(support: list[EdgeSupport]) -> str | None:
    """How much a model took part in an edge, from how its support was recorded;
    None (not stated) when nothing was recorded."""
    kinds = {s.kind for s in support}
    if not kinds:
        return None
    if kinds == {"extracted"}:
        return "full"
    if kinds == {"structural"}:
        return "none"
    return "mixed"


# Conflict review's verdict on a fact, as an OMP confidence outcome and verifier.
_REVIEW_OUTCOMES = {
    "conflicting": ("contradicted", "automatic"),
    "superseded": ("corrected", "automatic"),
    "current": ("confirmed", "automatic"),
    "accepted by user": ("confirmed", "human"),
}


def _reviewed_confidence(properties: dict) -> COGXConfidence | None:
    """The confidence conflict review states for a fact edge; None if unreviewed."""
    from cognee.modules.graph.utils.fact_conflicts import fact_status

    verdict = _REVIEW_OUTCOMES.get(fact_status(properties))
    if verdict is None:
        return None
    score = header_from_graph_properties(properties).get("confidence")
    return COGXConfidence(
        outcome=verdict[0],
        verified_by=verdict[1],
        score=score.score if score is not None else None,
    )


def _document_header(properties: dict) -> dict:
    """The OMP header a Document's data item was imported with, if any. The
    node carries the item's external metadata as a JSON string."""
    metadata = properties.get("external_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return {}
    if not isinstance(metadata, dict):
        return {}
    return parse_omp_header(metadata.get("omp_header"))


def _write_cogx(
    nodes,
    edges,
    destination: Path,
    dataset_name: str,
    facts: ExportFacts,
    embedding_model: str | None = None,
    migration_revision: str | None = None,
) -> None:
    """Map graph nodes/edges onto typed COGX records; keep the rest raw."""
    topology = _Topology.of(nodes, edges)
    with COGXArchiveWriter(destination, source_system="cognee") as writer:
        writer.add_note(f"Exported from Cognee dataset {dataset_name!r}.")
        writer.embedding_model = embedding_model
        writer.migration_revision = migration_revision
        writer.coverage = dict(COGX_EXPORT_COVERAGE)
        for node_id, properties in nodes:
            properties = properties or {}
            node_id = str(node_id)
            node_type = properties.get("type")
            if node_type == "Entity" and properties.get("name"):
                writer.write(_entity_record(node_id, properties, topology, facts))
            elif node_type == "DocumentChunk" and properties.get("text"):
                writer.write(_chunk_record(node_id, properties, topology, facts))
                # Also persist the chunk as a raw node: preserve-mode restore
                # rehydrates it as a graph node so facts referencing the chunk
                # (e.g. DocumentChunk -contains-> Entity) keep their topology
                # instead of dangling. The COGXDocument record above carries
                # the chunk's content for re-derive/cross-provider imports.
                writer.write_raw_node({"id": node_id, **properties})
            else:
                writer.write_raw_node({"id": node_id, **properties})

        for source, target, relationship, properties in edges:
            writer.write(
                _fact_record(
                    str(source), str(target), str(relationship), properties or {}, topology, facts
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
            await _load_export_facts(dataset_obj.id, nodes, edges),
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
