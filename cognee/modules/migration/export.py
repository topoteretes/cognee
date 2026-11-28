"""Export a Cognee dataset's knowledge graph to portable formats.

Formats:

- ``pydantic`` — a :class:`GraphSnapshot` of typed DataPoint instances (the
  Pydantic-native export: real ``Entity``/``DocumentChunk``/custom-model
  objects, losslessly serializable via ``model_dump_json``)
- ``cogx``    — a COGX archive directory (the canonical portable dump;
  re-importable via :class:`COGXArchiveSource`)
- ``omp``     — an Open Memory Protocol export (draft v0.6): ``manifest.json``
  plus ``memories.jsonl``, one memory object per line; ``omp-markdown``
  writes the same memories as a folder of markdown files with front matter.
  Both re-import through :class:`OMPSource`, and any OMP-conformant system
  can read them.
- ``json``    — full-fidelity nodes/edges JSON
- ``graphml`` — Gephi/yEd/NetworkX interop
- ``cypher`` — MERGE script loadable into any Neo4j-compatible database
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.migration.cogx import (
    COGX_VERSION,
    COGXArchiveWriter,
    COGXConfidence,
    COGXContributor,
    COGXDocument,
    COGXEntity,
    COGXEvidence,
    COGXFact,
    COGXLink,
    COGXOrigin,
    header_from_graph_properties,
    parse_omp_header,
    parse_timestamp,
    write_social_layer,
)
from cognee.modules.migration.formats import write_cypher, write_graphml, write_json
from cognee.modules.migration.omp import (
    OMPManifest,
    mark_dangling,
    memory_from_record,
    prepare_omp_destination,
    write_omp_export,
)
from cognee.modules.migration.snapshot import GraphSnapshot, build_snapshot
from cognee.shared.logging_utils import get_logger
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME

logger = get_logger("migration.export")

EXPORT_FORMATS = ("pydantic", "cogx", "omp", "omp-markdown", "json", "graphml", "cypher")

_FORMAT_SUFFIX = {"json": ".json", "graphml": ".graphml", "cypher": ".cypher"}
_DIRECTORY_SUFFIX = {"cogx": "_cogx", "omp": "_omp", "omp-markdown": "_omp"}


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


# What the cognee exporter can state for each OMP field (draft v0.6, 1.3 and
# 3.1). Declared in the manifest so an importer never mistakes an absent field
# for a negative.
COGX_EXPORT_COVERAGE = {
    "identifier": "the cognee node id, or source:relationship:target for a fact edge",
    "created_at": "the node or edge created_at; the export time where the store recorded none",
    "media_type": "text/plain for every body; documents keep their mime type",
    "checksum": "sha256 over the canonical JSON of each exported record, metadata and body",
    "provenance": "pipeline, task and user stamps on the node (source_pipeline, source_task, source_user): the user instructed, the pipeline authored what a model extracted and imported the rest; the user's kind is human or agent",
    "accountable": "the stamped user, or the owner of a stamped agent; otherwise not stated",
    "ai_involvement": "entities: full if cognify extracted them; chunks: none; facts: from their recorded evidence; otherwise as imported or not stated",
    "evidence": "facts: the chunks the edge evidence records (inferred if extracted, observed if structural); entities: the chunks containing them; no source spans",
    "links": "omp exports only: a chunk mentions the entities it contains; a fact links its subject and object, or only its subject when the object is a date its text states",
    "version": "DataPoint version from node properties, comparator integer",
    "validity": "valid_from/valid_until and invalidated_by read from node and edge properties when present",
    "weight": "node or edge property only; cognee assigns none",
    "tags": "node property only",
    "path": "not stated",
    "origin": "the origin a memory was imported with (a chunk takes its document's); otherwise the cognee node or edge id",
    "content_hash": "sha256 of each record body",
    "subject": "names of the entities and dates a chunk contains, or a fact's endpoints; entity records name themselves",
    "semantic_kind": "node property only; no classification is performed",
    "confidence": "facts: the conflict review verdict (contradicted, corrected, confirmed) with the edge's confidence score; otherwise as imported, or the score alone",
}
# What an OMP memory's checksum covers, which differs from a COGX record's.
OMP_CHECKSUM_COVERAGE = (
    "sha256 over the memory's identifier, version, created_at, updated_at and media_type, "
    "one per line, then its body; other fields are not covered"
)


def _contributors(properties: dict, actors: dict[str, "Actor"], authored: bool) -> dict:
    """Provenance list (OMP 3.1.1) and accountable human from a node's stamps.

    The stamped user instructed the write. The stamped pipeline authored the
    record when a model extracted it (``authored``), and imported it — stored
    it as it came — otherwise.
    """
    header: dict = {}
    provenance = []
    label = properties.get("source_user")
    if label:
        actor = actors.get(str(label))
        provenance.append(
            COGXContributor(
                id=str(label),
                kind=actor.kind if actor else "unknown",
                contribution="instructed",
            )
        )
        if actor is not None:
            header["accountable"] = actor.accountable
    pipeline = properties.get("source_pipeline") or properties.get("source_task")
    if pipeline:
        provenance.append(
            COGXContributor(
                id=f"cognee:{pipeline}",
                kind="agent",
                contribution="authored" if authored else "imported",
                ai_usage="full" if authored else "none",
            )
        )
    if provenance:
        header["provenance"] = provenance
    return header


def _element_header(
    properties: dict,
    element_id: str,
    actors: dict[str, "Actor"],
    imported: dict,
    authored: bool = False,
) -> dict:
    """OMP metadata for one node or edge.

    ``imported`` is the metadata the memory arrived with (its own
    ``omp_header``, or for a chunk its document's) and wins; cognee's own
    properties and pipeline stamps fill the rest, and the origin defaults to
    cognee itself. ``authored`` says whether a model wrote the record.
    """
    header = dict(imported)
    derived = {
        **header_from_graph_properties(properties),
        **_contributors(properties, actors, authored),
    }
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


# What a chunk ``contains`` that the exports name: its entities and dates.
_CHUNK_CONTENT_TYPES = ("Entity", "Timestamp")


@dataclass
class _Topology:
    """What a record's header reads from the rest of the graph."""

    names: dict[str, str]  # node id -> name, for fact subjects
    contents_of_chunk: dict[str, list[str]]  # chunk id -> ids of its entities and dates
    chunks_containing: dict[str, list[str]]  # entity or date id -> ids of chunks containing it
    chunk_headers: dict[str, dict]  # chunk id -> the header its document was imported with
    entity_types: dict[str, str]  # entity id -> the name of its type (its ``is_a`` target)
    memory_nodes: set[str]  # ids of the nodes an OMP export writes as memories
    date_nodes: set[str]  # ids of the Timestamp nodes: values a fact may end at

    @classmethod
    def of(cls, nodes, edges) -> "_Topology":
        by_id: dict[str, dict] = {}
        names: dict[str, str] = {}
        memory_nodes: set[str] = set()
        date_nodes: set[str] = set()
        for node_id, properties in nodes:
            node_id, properties = str(node_id), properties or {}
            by_id[node_id] = properties
            # A date is named by its text: Timestamp nodes before SDK-827 have no name.
            name = properties.get("name") or properties.get("timestamp_str")
            if name:
                names[node_id] = str(name)
            field = _MEMORY_NODE_FIELDS.get(properties.get("type"))
            if field and properties.get(field):
                memory_nodes.add(node_id)
            elif properties.get("type") == "Timestamp":
                date_nodes.add(node_id)
        topology = cls(
            names=names,
            contents_of_chunk={},
            chunks_containing={},
            chunk_headers={},
            entity_types={},
            memory_nodes=memory_nodes,
            date_nodes=date_nodes,
        )
        for source, target, relationship, _properties in edges:
            source, target = str(source), str(target)
            source_type = by_id.get(source, {}).get("type")
            target_type = by_id.get(target, {}).get("type")
            is_type = (source_type, target_type) == ("Entity", "EntityType")
            if relationship == "is_a" and is_type and target in names:
                topology.entity_types[source] = names[target]
            if source_type != "DocumentChunk":
                continue
            if relationship == "contains" and target_type in _CHUNK_CONTENT_TYPES:
                topology.contents_of_chunk.setdefault(source, []).append(target)
                topology.chunks_containing.setdefault(target, []).append(source)
            elif relationship == "is_part_of":
                topology.chunk_headers[source] = _document_header(by_id.get(target, {}))
        return topology


def _entity_record(
    node_id: str, properties: dict, topology: _Topology, facts: ExportFacts
) -> COGXEntity:
    imported = parse_omp_header(properties.get("omp_header"))
    extracted = properties.get("source_pipeline") == COGNIFY_PIPELINE_NAME
    header = _element_header(properties, node_id, facts.actors, imported, authored=extracted)
    header.setdefault("subject", [properties["name"]])
    header.setdefault(
        "evidence",
        [
            COGXEvidence(relation="derived", ref=chunk_id)
            for chunk_id in topology.chunks_containing.get(node_id, [])
        ],
    )
    if extracted:
        header.setdefault("ai_involvement", "full")  # extracted by a model
    return COGXEntity(
        external_system="cognee",
        external_id=node_id,
        name=properties["name"],
        entity_type=topology.entity_types.get(node_id),
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
        [topology.names.get(node, node) for node in topology.contents_of_chunk.get(node_id, [])],
    )
    header.setdefault("ai_involvement", "none")  # source text; chunking runs no model
    return COGXDocument(
        external_system="cognee",
        external_id=node_id,
        content=properties["text"],
        created_at=parse_timestamp(properties.get("created_at")),
        **header,
    )


_EVIDENCE_RELATIONS = {"extracted": "inferred", "structural": "observed"}


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
    support = facts.edge_support.get(external_id, [])
    involvement = _ai_involvement(support)
    header = _element_header(
        properties, external_id, facts.actors, imported, authored=involvement == "full"
    )
    header.setdefault(
        "evidence",
        [
            COGXEvidence(
                relation=_EVIDENCE_RELATIONS[s.kind], ref=s.chunk_id, confidence=s.confidence
            )
            for s in support
            if s.kind in _EVIDENCE_RELATIONS
        ],
    )
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


# Header fields that describe a whole imported document, not its chunks.
_DOCUMENT_ONLY_FIELDS = ("path", "stub", "links")


def _document_header(properties: dict) -> dict:
    """The OMP header a Document's data item was imported with, if any, as its
    chunks inherit it. The node carries the item's external metadata as a JSON
    string. Where the file sat, whether it is a stub and how it links to other
    memories describe the whole document, not a slice of it, so chunks do not
    inherit them."""
    metadata = properties.get("external_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return {}
    if not isinstance(metadata, dict):
        return {}
    header = parse_omp_header(metadata.get("omp_header"))
    return {key: value for key, value in header.items() if key not in _DOCUMENT_ONLY_FIELDS}


def _write_cogx(
    nodes,
    edges,
    destination: Path,
    dataset_name: str,
    facts: ExportFacts,
    embedding_model: str | None = None,
    migration_revision: str | None = None,
    origin: str | None = None,
) -> None:
    """Map graph nodes/edges onto typed COGX records; keep the rest raw.

    ``origin`` is the store-level identifier the manifest declares (OMP 5.3):
    with a record's ``external_id`` it is the memory's global key.
    """
    topology = _Topology.of(nodes, edges)
    with COGXArchiveWriter(destination, source_system="cognee", origin=origin) as writer:
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


# The node types exported as memories, and the property each needs to be one.
_MEMORY_NODE_FIELDS = {"Entity": "name", "DocumentChunk": "text"}


def _fact_edges(edges, topology: "_Topology"):
    """The edges exported as facts: from an exported memory to another one or
    to a date, and not a chunk's ``contains`` edge, which its ``mentions``
    links and subject names already carry."""
    memory_nodes, date_nodes = topology.memory_nodes, topology.date_nodes
    fact_edges = []
    for source, target, relationship, properties in edges:
        source, target = str(source), str(target)
        if source not in memory_nodes or (target not in memory_nodes and target not in date_nodes):
            continue
        contents = topology.contents_of_chunk.get(source, ())
        if not (relationship == "contains" and target in contents):
            fact_edges.append((source, target, str(relationship), properties or {}))
    return fact_edges


def _node_memory(node_id, properties, topology, mentions, facts, exported_at):
    """The OMP memory of one node in :attr:`_Topology.memory_nodes`."""
    if properties["type"] == "Entity":
        record = _entity_record(node_id, properties, topology, facts)
    else:
        record = _chunk_record(node_id, properties, topology, facts)
        record.links.extend(
            COGXLink(target=target, rel="mentions") for target in mentions.get(node_id, [])
        )
    return memory_from_record(record, exported_at=exported_at)


def _fact_memory(
    source, target, relationship, properties, topology, facts, exported_at, linked_object
):
    """The OMP memory of one fact edge, linking its ``subject``, and its
    ``object`` when that is a memory too (a date is stated in the text)."""
    record = _fact_record(source, target, relationship, properties, topology, facts)
    record.links.append(COGXLink(target=source, rel="subject"))
    if linked_object:
        record.links.append(COGXLink(target=target, rel="object"))
    return memory_from_record(record, exported_at=exported_at)


def _omp_memories(nodes, edges, facts: ExportFacts, exported_at: datetime):
    """The dataset as OMP memories: every entity, chunk and fact is one.

    A chunk links the entities it contains (``mentions``), so those
    ``contains`` edges are not facts again. A fact is an edge from an exported
    memory to another one, linked as its ``subject`` and ``object``, or to a
    date, which only its text states: dates, like entity types, are values,
    not memories. Cognee-internal nodes (entity types, summaries, document
    records) are graph structure: they and their other edges are left out —
    the manifest says so.
    """
    topology = _Topology.of(nodes, edges)
    memory_nodes = topology.memory_nodes
    mentions = {
        chunk: [target for target in contents if target in memory_nodes]
        for chunk, contents in topology.contents_of_chunk.items()
    }
    fact_edges = _fact_edges(edges, topology)
    # Every identifier of the export, known before the first memory is written,
    # so references outside it are marked dangling.
    exported = memory_nodes | {f"{s}:{r}:{t}" for s, t, r, _properties in fact_edges}
    for node_id, properties in nodes:
        if str(node_id) in memory_nodes:
            memory = _node_memory(str(node_id), properties, topology, mentions, facts, exported_at)
            yield mark_dangling(memory, exported)
    for source, target, relationship, properties in fact_edges:
        linked_object = target in memory_nodes
        memory = _fact_memory(
            source, target, relationship, properties, topology, facts, exported_at, linked_object
        )
        yield mark_dangling(memory, exported)


def _write_omp(
    nodes,
    edges,
    destination: Path,
    dataset_name: str,
    facts: ExportFacts,
    *,
    origin: str,
    layout: str,
    embedding_model: str | None = None,
    migration_revision: str | None = None,
) -> dict[str, Any]:
    """Write the dataset as an OMP export (draft v0.6): ``memories.jsonl`` or a
    folder of markdown memories, plus the 5.3 manifest. ``destination`` is
    empty or absent: :func:`export_dataset` prepared it before reading the graph."""
    exported_at = datetime.now(timezone.utc)
    manifest = OMPManifest.model_validate(
        {
            "origin": origin,
            "exported_at": exported_at,
            "cognee": {
                "cogx_version": COGX_VERSION,
                "dataset": dataset_name,
                "embedding_model": embedding_model,
                "migration_revision": migration_revision,
                "coverage": {**COGX_EXPORT_COVERAGE, "checksum": OMP_CHECKSUM_COVERAGE},
                "not_exported": "cognee-internal graph nodes (entity types, summaries, "
                "document records) are structure, not memories; dates are values, "
                "stated by the facts about them, and entity types travel on the entity",
            },
        }
    )
    return write_omp_export(
        _omp_memories(nodes, edges, facts, exported_at),
        destination,
        manifest=manifest,
        layout=layout,
    )


def _export_destination(format: str, destination: str | Path | None, dataset_name: str) -> Path:
    """Where an export goes: ``destination``, else the format's default in the
    working directory (``<dataset>_export.<ext>``, or a folder for cogx/omp)."""
    if destination is None:
        suffix = _FORMAT_SUFFIX.get(format)
        destination = (
            f"{dataset_name}_export{suffix}"
            if suffix
            else f"{dataset_name}{_DIRECTORY_SUFFIX[format]}"
        )
    return Path(destination)


async def export_dataset(
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    format: str = "pydantic",
    destination: str | Path | None = None,
    user=None,
    link_relations: bool = False,
    include_permissions: bool = False,
    overwrite: bool = False,
) -> ExportResult | GraphSnapshot:
    """Export an authorized dataset's graph. Requires read permission.

    ``overwrite`` (omp formats only): an OMP export refuses a non-empty
    ``destination``; ``True`` deletes it and everything in it first.

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
    if format in ("omp", "omp-markdown"):
        # Refuse a folder in use before reading the whole graph.
        prepare_omp_destination(
            _export_destination(format, destination, dataset_obj.name), overwrite
        )

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

    destination = _export_destination(format, destination, dataset_obj.name)

    if format in ("cogx", "omp", "omp-markdown"):
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
        export_facts = await _load_export_facts(dataset_obj.id, nodes, edges)
        migration_revision = await _stored_migration_revision(dataset_obj.id)
        # The store-level origin (OMP 5.3): a cognee dataset is one store.
        origin = f"cognee:{dataset_obj.id}"
        if format == "cogx":
            _write_cogx(
                nodes,
                edges,
                destination,
                dataset_obj.name,
                export_facts,
                embedding_model=embedding_model,
                migration_revision=migration_revision,
                origin=origin,
            )
            if include_permissions:
                write_social_layer(destination, await _gather_social_layer(dataset_obj))
        else:
            _write_omp(
                nodes,
                edges,
                destination,
                dataset_obj.name,
                export_facts,
                origin=origin,
                layout="markdown" if format == "omp-markdown" else "jsonl",
                embedding_model=embedding_model,
                migration_revision=migration_revision,
            )
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
