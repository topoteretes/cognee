"""Memory provenance projection.

Projects cognee's *relational* actor/ownership/session metadata
(Tenant → User → Agent → Dataset/"brain" → Data/file, plus agent read/write
access and agent-written Sessions) into a single ``(nodes, edges)`` graph in
the exact shape ``GraphDBInterface.get_graph_data()`` returns — so the schema
view can render the full ownership & data-flow story.

The actor layer is read purely from the relational database (no knowledge-graph
DB and no LLM required), so it works even when the graph backend is unavailable.
When ``include_memory=True`` the extracted memory (entities/relationships) is
folded in from the relational ``nodes``/``edges`` tables and linked back to the
files it was extracted from.

Entry points:
    * ``build_provenance_graph(...)`` — pure, side-effect-free assembly from
      plain records (unit-testable).
    * ``get_memory_provenance_graph(...)`` — async reader that pulls live data
      from the relational layer and calls the builder.
    * ``visualize_memory_provenance(...)`` / ``get_memory_provenance_payload(...)``
      — the same graph rendered as HTML or packaged as a JSON dict, sharing
      one ``preprocess()`` call so the two cannot drift.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict, cast

from cognee.shared.logging_utils import get_logger

logger = get_logger()


class Node(NamedTuple):
    """A graph node in ``get_graph_data()`` shape: ``(id, properties)``.

    A ``NamedTuple`` (not a dataclass) so the projection stays interchangeable
    with the raw ``GraphDBInterface.get_graph_data()`` output the renderer and
    preprocessor already consume, while still naming what each position means.
    """

    id: str
    properties: Dict[str, Any]


class EdgeData(NamedTuple):
    """A graph edge in ``get_graph_data()`` shape: ``(source, target, relation, properties)``."""

    source: str
    target: str
    relation: str
    properties: Dict[str, Any]


# ── Input record shapes (relational projection inputs) ───────────────────────
# TypedDicts make the expected keys explicit. ``id`` is required on each record;
# the remaining keys are optional (read via ``.get(...)``), hence ``total=False``.


class _HasId(TypedDict):
    id: str


class TenantRecord(_HasId, total=False):
    name: Optional[str]


class UserRecord(_HasId, total=False):
    name: Optional[str]
    tenant_ids: List[str]


class RoleRecord(_HasId, total=False):
    name: Optional[str]
    tenant_id: Optional[str]
    user_ids: List[str]


class AclGrantRecord(TypedDict, total=False):
    """One ACL row: a permission a principal holds on a dataset.

    ``principal_kind`` is one of "role" | "user" | "tenant" — the same
    vocabulary the CLO-399 dataset-groups endpoint uses, so a client already
    rendering that distinction elsewhere does not learn a second one here.
    """

    principal_id: str
    principal_kind: str
    dataset_id: str
    permission: str


class DatasetRecord(_HasId, total=False):
    name: Optional[str]
    owner_id: Optional[str]
    tenant_id: Optional[str]


class FileRecord(_HasId, total=False):
    name: Optional[str]
    dataset_ids: List[str]
    dataset_name: Optional[str]


class AgentDatasetRef(TypedDict, total=False):
    dataset_id: str
    role: str  # "read" | "read_write"


class AgentRecord(_HasId, total=False):
    name: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    datasets: List[AgentDatasetRef]


class SessionRecord(_HasId, total=False):
    name: Optional[str]
    user_id: Optional[str]
    dataset_id: Optional[str]
    agent_id: Optional[str]


class MemoryPayload(TypedDict, total=False):
    nodes: List[Tuple[str, Dict[str, Any]]]
    edges: List[Tuple[str, str, str, Dict[str, Any]]]
    links: List[Dict[str, Any]]


# Delete/share are still projected (as their own edge, not folded into reads/
# writes) rather than dropped: an ACL row exists because someone granted it,
# and hiding half of what a principal can do would misstate the story this
# graph exists to tell.
_ACL_EDGE_RELATIONS = {
    "read": "reads",
    "write": "writes",
    "delete": "can_delete",
    "share": "can_share",
}


def build_provenance_graph(
    *,
    tenants: Optional[List[TenantRecord]] = None,
    users: Optional[List[UserRecord]] = None,
    roles: Optional[List[RoleRecord]] = None,
    datasets: Optional[List[DatasetRecord]] = None,
    files: Optional[List[FileRecord]] = None,
    agents: Optional[List[AgentRecord]] = None,
    sessions: Optional[List[SessionRecord]] = None,
    acl_grants: Optional[List[AclGrantRecord]] = None,
    memory: Optional[MemoryPayload] = None,
) -> Tuple[List[Node], List[EdgeData]]:
    """Assemble actor/ownership/session records into a ``(nodes, edges)`` graph.

    Record shapes (all ids are strings):
        tenants:  {"id", "name"}
        users:    {"id", "name", "tenant_ids": [..]}
        roles:    {"id", "name", "tenant_id", "user_ids": [..]}
        datasets: {"id", "name", "owner_id", "tenant_id"}
        files:    {"id", "name", "dataset_ids": [..], "dataset_name"?}
        agents:   {"id", "name", "user_id", "session_id"?,
                   "datasets": [{"dataset_id", "role": "read"|"read_write"}]}
        sessions: {"id", "name", "user_id", "dataset_id", "agent_id"?}
        acl_grants: [{"principal_id", "principal_kind": "role"|"user"|"tenant",
                      "dataset_id", "permission": "read"|"write"|"delete"|"share"}]
        memory:   optional {"nodes": [(id, props)], "edges": [(s, t, rel, props)],
                            "links": [{"node_id", "data_id", "dataset_id"}]}

    Node ids are namespaced (``user:<id>`` etc.) so the actor layers never
    collide with each other or with raw memory-node ids.
    """
    tenants = tenants or []
    users = users or []
    roles = roles or []
    datasets = datasets or []
    files = files or []
    agents = agents or []
    sessions = sessions or []
    acl_grants = acl_grants or []

    nodes: Dict[str, Node] = {}
    edges: List[EdgeData] = []
    seen_edges = set()

    def add_node(node_id: str, node_type: str, name: str, **extra) -> None:
        if node_id not in nodes:
            props = {"type": node_type, "name": name}
            props.update({k: v for k, v in extra.items() if v is not None})
            nodes[node_id] = Node(node_id, props)

    def add_edge(source: str, target: str, relation: str) -> None:
        if source in nodes and target in nodes:
            key = (source, target, relation)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(EdgeData(source, target, relation, {}))

    # ── Actor / ownership nodes ──────────────────────────────────────
    for tenant in tenants:
        add_node(f"tenant:{tenant['id']}", "Tenant", tenant.get("name") or "Tenant")
    for user in users:
        add_node(f"user:{user['id']}", "User", user.get("name") or str(user["id"]))
    for role in roles:
        add_node(f"role:{role['id']}", "Role", role.get("name") or str(role["id"]))
    for dataset in datasets:
        add_node(f"dataset:{dataset['id']}", "Dataset", dataset.get("name") or "Dataset")
    for file in files:
        add_node(
            f"file:{file['id']}",
            "TextDocument",
            file.get("name") or "file",
            source_node_set=file.get("dataset_name"),
        )
    for agent in agents:
        add_node(f"agent:{agent['id']}", "Agent", agent.get("name") or str(agent["id"]))
    for sess in sessions:
        add_node(f"session:{sess['id']}", "Session", sess.get("name") or str(sess["id"]))

    # ── Edges ────────────────────────────────────────────────────────
    for user in users:
        for tenant_id in user.get("tenant_ids") or []:
            add_edge(f"tenant:{tenant_id}", f"user:{user['id']}", "has_member")

    for role in roles:
        role_tenant_id = role.get("tenant_id")
        if role_tenant_id:
            add_edge(f"tenant:{role_tenant_id}", f"role:{role['id']}", "has_role")
        for user_id in role.get("user_ids") or []:
            add_edge(f"role:{role['id']}", f"user:{user_id}", "has_member")

    for dataset in datasets:
        owner_id = dataset.get("owner_id")
        if owner_id:
            add_edge(f"user:{owner_id}", f"dataset:{dataset['id']}", "owns")

    for file in files:
        for dataset_id in file.get("dataset_ids") or []:
            add_edge(f"dataset:{dataset_id}", f"file:{file['id']}", "contains")

    for agent in agents:
        agent_user_id = agent.get("user_id")
        if agent_user_id:
            add_edge(f"user:{agent_user_id}", f"agent:{agent['id']}", "operates")
        for ref in agent.get("datasets") or []:
            dataset_id = ref.get("dataset_id")
            if not dataset_id:
                continue
            add_edge(f"agent:{agent['id']}", f"dataset:{dataset_id}", "reads")
            if ref.get("role") == "read_write":
                add_edge(f"agent:{agent['id']}", f"dataset:{dataset_id}", "writes")
        agent_session_id = agent.get("session_id")
        if agent_session_id:
            add_edge(f"agent:{agent['id']}", f"session:{agent_session_id}", "wrote")

    for sess in sessions:
        sess_agent_id = sess.get("agent_id")
        if sess_agent_id:
            add_edge(f"agent:{sess_agent_id}", f"session:{sess['id']}", "wrote")
        sess_dataset_id = sess.get("dataset_id")
        if sess_dataset_id:
            add_edge(f"session:{sess['id']}", f"dataset:{sess_dataset_id}", "recorded_in")

    # ACL grants: what a role, a user or the tenant principal itself can do
    # with a dataset, as distinct from who *owns* it (the "owns" edges above,
    # from Dataset.owner_id). A dataset shared via ShareDatasetModal with a
    # group, an individual, or the whole workspace shows up here and nowhere
    # else — before this, only ownership was visible, so a shared dataset
    # with no ACL projection looked unshared.
    for grant in acl_grants:
        edge_relation = _ACL_EDGE_RELATIONS.get(grant.get("permission"))
        if edge_relation is None:
            continue
        source = f"{grant['principal_kind']}:{grant['principal_id']}"
        add_edge(source, f"dataset:{grant['dataset_id']}", edge_relation)

    # ── Optional memory layer ────────────────────────────────────────
    if memory:
        for node_id, props in memory.get("nodes") or []:
            nid = str(node_id)
            if nid not in nodes:
                nodes[nid] = Node(nid, dict(props))
        for source, target, relation, eprops in memory.get("edges") or []:
            s, t = str(source), str(target)
            if s in nodes and t in nodes:
                key = (s, t, relation)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(EdgeData(s, t, relation or "related", dict(eprops or {})))
        for link in memory.get("links") or []:
            file_id = f"file:{link['data_id']}" if link.get("data_id") else None
            node_id = str(link["node_id"]) if link.get("node_id") else None
            if file_id and node_id:
                add_edge(file_id, node_id, "mentions")

    return list(nodes.values()), edges


# ── Live relational readers ──────────────────────────────────────────────────


async def _read_agents(user_ids: List[str]) -> List[AgentRecord]:
    """Best-effort enumeration of agent connections (registered + persisted)."""
    from uuid import UUID

    connections: List[Any] = []
    try:
        from cognee.modules.agents.registry import (
            list_persisted_agent_connections,
            list_registered_agent_connections,
        )

        try:
            connections += list(list_registered_agent_connections() or [])
        except Exception as error:  # pragma: no cover - defensive
            logger.debug(f"registered agent enumeration skipped: {error}")
        try:
            connections += list(
                await list_persisted_agent_connections(
                    [UUID(uid) for uid in user_ids], active_only=False
                )
                or []
            )
        except Exception as error:  # pragma: no cover - defensive
            logger.debug(f"persisted agent enumeration skipped: {error}")
    except Exception as error:  # pragma: no cover - module unavailable
        logger.debug(f"agent registry unavailable: {error}")
        return []

    agents: List[AgentRecord] = []
    seen = set()
    for conn in connections:
        if conn.id in seen:
            continue
        seen.add(conn.id)
        refs: List[AgentDatasetRef] = [
            {"dataset_id": str(ref.id), "role": ref.role or "read"}
            for ref in (conn.datasets or [])
            if getattr(ref, "id", None)
        ]
        agents.append(
            {
                "id": conn.id,
                "name": conn.agent_session_name or conn.id,
                "user_id": str(conn.user_id) if conn.user_id else None,
                "session_id": conn.session_id,
                "datasets": refs,
            }
        )
    return agents


async def _read_sessions(user_ids: List[str], agents: List[AgentRecord]) -> List[SessionRecord]:
    """Best-effort enumeration of session records."""
    from uuid import UUID

    agent_by_session = {sid: a["id"] for a in agents if (sid := a.get("session_id"))}
    sessions: List[SessionRecord] = []
    try:
        from cognee.modules.session_lifecycle.metrics import list_session_rows

        page = await list_session_rows(user_ids=[UUID(uid) for uid in user_ids], limit=10000)
        for row in getattr(page, "sessions", None) or []:
            record = getattr(row, "record", row)
            session_id = record.session_id
            sessions.append(
                {
                    "id": session_id,
                    "name": session_id,
                    "user_id": str(record.user_id) if record.user_id else None,
                    "dataset_id": str(record.dataset_id) if record.dataset_id else None,
                    "agent_id": agent_by_session.get(session_id),
                }
            )
    except Exception as error:  # pragma: no cover - defensive
        logger.debug(f"session enumeration skipped: {error}")
    return sessions


async def _read_memory_relational(
    limit: int = 5000, dataset_ids: Optional[List[str]] = None
) -> Optional[MemoryPayload]:
    """Read extracted memory from the relational ``nodes``/``edges`` tables.

    Avoids the knowledge-graph backend entirely, so it works when that backend
    is unavailable. Returns None when there is no memory to show.

    When ``dataset_ids`` is provided, only memory nodes belonging to those
    datasets are returned, and edges are kept only when BOTH endpoints are
    in-scope — so a scoped provenance graph never folds in another tenant's
    extracted memory.
    """
    try:
        from sqlalchemy import select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.graph.models.Node import Node as NodeRow
        from cognee.modules.graph.models.Edge import Edge as EdgeRow
    except Exception as error:  # pragma: no cover - models unavailable
        logger.debug(f"relational memory models unavailable: {error}")
        return None

    nodes: List[Tuple[str, Dict[str, Any]]] = []
    edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
    links: List[Dict[str, Any]] = []
    node_ids: set = set()
    try:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            node_stmt = select(NodeRow)
            if dataset_ids is not None:
                # Scope to the in-scope datasets. An empty list yields no rows,
                # which is the correct fail-closed behaviour.
                node_stmt = node_stmt.where(NodeRow.dataset_id.in_(dataset_ids))
            for row in (await session.execute(node_stmt.limit(limit))).scalars().all():
                node_id = str(row.id)
                node_ids.add(node_id)
                nodes.append(
                    Node(node_id, {"type": row.type or "Node", "name": row.label or str(row.slug)})
                )
                if row.data_id is not None:
                    links.append(
                        {
                            "node_id": node_id,
                            "data_id": str(row.data_id),
                            "dataset_id": str(row.dataset_id)
                            if row.dataset_id is not None
                            else None,
                        }
                    )
            for row in (await session.execute(select(EdgeRow).limit(limit * 4))).scalars().all():
                src, dst = str(row.source_node_id), str(row.destination_node_id)
                if dataset_ids is not None and (src not in node_ids or dst not in node_ids):
                    # Drop edges that reach outside the scoped node set.
                    continue
                edges.append(EdgeData(src, dst, row.relationship_name or "related", {}))
    except Exception as error:  # pragma: no cover - defensive
        logger.debug(f"relational memory read skipped: {error}")
        return None

    if not nodes:
        return None
    return {"nodes": nodes, "edges": edges, "links": links}


async def _read_memory_graph_provenance(
    limit: int = 5000, dataset_ids: Optional[List[str]] = None
) -> Optional[MemoryPayload]:
    """Read extracted memory from graph provenance on ledger-free graphs."""
    if not dataset_ids:
        return None

    from cognee.infrastructure.databases.provenance import (
        EdgeIdentity,
        parse_source_ref_key,
    )
    from cognee.infrastructure.databases.provenance.markers import (
        stores_provenance_in_graph,
    )
    from cognee.infrastructure.databases.unified import get_unified_engine
    from cognee.infrastructure.engine import is_internal_node

    unified = await get_unified_engine()
    graph = unified.graph
    if not await stores_provenance_in_graph(graph):
        return None

    refs_by_node: Dict[str, List[str]] = {}
    refs_by_edge: Dict[Any, List[str]] = {}
    for dataset_id in dataset_ids:
        for node_id, refs in (await graph.find_node_source_refs_by_dataset(dataset_id)).items():
            refs_by_node.setdefault(str(node_id), []).extend(refs)
        for edge, refs in (await graph.find_edge_source_refs_by_dataset(dataset_id)).items():
            refs_by_edge.setdefault(edge, []).extend(refs)

    graph_nodes, graph_edges = await graph.get_graph_data()

    # Internal nodes (``is_internal`` marker, e.g. per-user preference state)
    # must never be surfaced; drop them plus every edge and link touching them.
    internal_ids = {str(node_id) for node_id, props in graph_nodes if is_internal_node(props)}

    node_ids = set(refs_by_node)
    node_ids.update(edge.source_id for edge in refs_by_edge)
    node_ids.update(edge.target_id for edge in refs_by_edge)
    node_ids -= internal_ids
    nodes_by_id = {str(node_id): props for node_id, props in graph_nodes}

    nodes: List[Tuple[str, Dict[str, Any]]] = []
    for node_id in sorted(node_ids)[:limit]:
        props = nodes_by_id.get(node_id)
        if props is not None:
            nodes.append(Node(node_id, dict(props)))

    edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
    edge_ref_keys = set(refs_by_edge)
    for source, target, relation, props in graph_edges:
        edge = EdgeIdentity(str(source), str(target), str(relation))
        if edge not in edge_ref_keys:
            continue
        if str(source) not in node_ids or str(target) not in node_ids:
            continue
        edges.append(EdgeData(str(source), str(target), relation or "related", dict(props or {})))
        if len(edges) >= limit * 4:
            break

    links: List[Dict[str, Any]] = []
    for node_id, refs in refs_by_node.items():
        if node_id in internal_ids:
            continue
        for source_ref_key in refs:
            # v1 refs are document-scoped; v2 refs additionally name the
            # producing chunk. Foreign formats are skipped, never crash the
            # visualization.
            try:
                parsed = parse_source_ref_key(source_ref_key)
            except ValueError:
                continue
            link = {
                "node_id": node_id,
                "data_id": str(parsed.data_id),
                "dataset_id": str(parsed.dataset_id),
            }
            if parsed.chunk_id is not None:
                link["chunk_id"] = str(parsed.chunk_id)
            links.append(link)

    if not nodes:
        return None
    return {"nodes": nodes, "edges": edges, "links": links}


async def _read_roles_and_grants(
    tenant_ids: Optional[List[str]],
    dataset_ids: List[str],
    scope_user_ids: Optional[List[str]] = None,
) -> Tuple[List[RoleRecord], List[AclGrantRecord]]:
    """Best-effort read of roles (with membership) and ACL grants on the
    in-scope datasets.

    ``Principal.id``/``Principal.type`` are read as bare columns rather than
    through ``select(Principal)`` or an outer join onto ``User``/``Role``/
    ``Tenant``: the latter makes SQLAlchemy try to resolve the polymorphic
    identity across all three subclass tables at once and it folds rows
    together — the exact bug CLO-399's dataset-groups endpoint hit first.
    Reading the two plain columns sidesteps that; there is no entity to
    misidentify when nothing is asking the ORM to build one.

    ``tenant_ids=None`` with ``scope_user_ids`` set (the OSS/single-user path,
    where roles are not owned by a user) keeps only roles the scoped users are
    actually members of, so this cannot surface a group of people outside the
    requested scope.
    """
    from uuid import UUID as _UUID

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import ACL, Permission, Role
    from cognee.modules.users.models.Principal import Principal

    def _to_uuid(value):
        # Role.tenant_id and ACL.dataset_id are sqlalchemy.UUID(as_uuid=True):
        # on SQLite the bind processor calls value.hex, which a plain string
        # does not have. Callers pass either — get_memory_provenance_graph's
        # dataset_ids are already stringified — so coerce once here instead
        # of pushing that constraint onto every caller.
        return value if isinstance(value, _UUID) else _UUID(str(value))

    roles: List[RoleRecord] = []
    grants: List[AclGrantRecord] = []

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        role_stmt = select(Role).options(selectinload(Role.users))
        if tenant_ids is not None:
            role_stmt = role_stmt.where(Role.tenant_id.in_(_to_uuid(t) for t in tenant_ids))
        role_rows = (await session.execute(role_stmt)).scalars().all()

        if tenant_ids is None and scope_user_ids is not None:
            allowed = set(scope_user_ids)
            role_rows = [r for r in role_rows if {str(u.id) for u in r.users} & allowed]

        for role in role_rows:
            roles.append(
                {
                    "id": str(role.id),
                    "name": role.name,
                    "tenant_id": str(role.tenant_id),
                    "user_ids": [str(u.id) for u in (role.users or [])],
                }
            )

        if dataset_ids:
            grant_stmt = (
                select(Principal.id, Principal.type, ACL.dataset_id, Permission.name)
                .select_from(ACL)
                .join(Principal, Principal.id == ACL.principal_id)
                .join(Permission, Permission.id == ACL.permission_id)
                .where(ACL.dataset_id.in_(_to_uuid(d) for d in dataset_ids))
            )
            for principal_id, kind, dataset_id, permission in await session.execute(grant_stmt):
                grants.append(
                    {
                        "principal_id": str(principal_id),
                        "principal_kind": kind,
                        "dataset_id": str(dataset_id),
                        "permission": permission,
                    }
                )

    return roles, grants


async def get_memory_provenance_graph(
    include_memory: bool = False,
    scope_tenant_ids: Optional[List[Any]] = None,
    scope_user_ids: Optional[List[Any]] = None,
) -> Tuple[List[Node], List[EdgeData]]:
    """Read live relational data and project it into a provenance ``(nodes, edges)``.

    Args:
        include_memory: when True, fold in the extracted memory from the
            relational ``nodes``/``edges`` tables and link it to source files.
        scope_tenant_ids: when set, restrict the graph to these tenants (and the
            users/datasets/agents/sessions/memory within them). REQUIRED in
            multi-tenant deployments — without a scope this reads EVERY tenant's
            actors, datasets and files, leaking data across tenants.
        scope_user_ids: alternative scope used when there is no tenant context
            (single-user/OSS installs): restrict to these users and what they own.
        When neither scope is given the read is global — the OSS local default,
        where the single user owns everything.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import joinedload, selectinload

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.users.models import Tenant, User

    scoped = scope_tenant_ids is not None or scope_user_ids is not None

    tenants: List[Dict[str, Any]] = []
    users: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    files: Dict[str, Dict[str, Any]] = {}

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        tenant_stmt = select(Tenant)
        if scope_tenant_ids is not None:
            tenant_stmt = tenant_stmt.where(Tenant.id.in_(scope_tenant_ids))
        for tenant in (await session.execute(tenant_stmt)).scalars().all():
            tenants.append({"id": str(tenant.id), "name": tenant.name})

        user_stmt = select(User).options(selectinload(User.tenants))
        if scope_tenant_ids is not None:
            user_stmt = user_stmt.where(User.tenant_id.in_(scope_tenant_ids))
        elif scope_user_ids is not None:
            user_stmt = user_stmt.where(User.id.in_(scope_user_ids))
        user_rows = (await session.execute(user_stmt)).scalars().all()
        for user in user_rows:
            tenant_ids = [str(t.id) for t in (user.tenants or [])]
            if getattr(user, "tenant_id", None):
                tenant_ids.append(str(user.tenant_id))
            users.append(
                {
                    "id": str(user.id),
                    "name": getattr(user, "name", None) or f"user:{str(user.id)[:8]}",
                    "tenant_ids": sorted(set(tenant_ids)),
                }
            )

        dataset_stmt = select(Dataset).options(joinedload(Dataset.data))
        if scope_tenant_ids is not None:
            dataset_stmt = dataset_stmt.where(Dataset.tenant_id.in_(scope_tenant_ids))
        elif scope_user_ids is not None:
            dataset_stmt = dataset_stmt.where(Dataset.owner_id.in_(scope_user_ids))
        dataset_rows = (await session.execute(dataset_stmt)).unique().scalars().all()
        for dataset in dataset_rows:
            datasets.append(
                {
                    "id": str(dataset.id),
                    "name": dataset.name,
                    "owner_id": str(dataset.owner_id) if dataset.owner_id is not None else None,
                    "tenant_id": str(dataset.tenant_id) if dataset.tenant_id is not None else None,
                }
            )
            for data in dataset.data or []:
                file_id = str(data.id)
                record = files.setdefault(
                    file_id,
                    {
                        "id": file_id,
                        "name": data.name,
                        "dataset_ids": [],
                        "dataset_name": dataset.name,
                    },
                )
                record["dataset_ids"].append(str(dataset.id))

    user_ids = [u["id"] for u in users]
    dataset_ids = [d["id"] for d in datasets]
    agents = await _read_agents(user_ids)
    if scoped:
        # _read_agents also folds in globally-registered (in-process) agent
        # connections; drop any that don't belong to an in-scope user so the
        # scoped graph can't surface another user's agent.
        allowed_user_ids = set(user_ids)
        agents = [a for a in agents if a.get("user_id") in allowed_user_ids]
    sessions = await _read_sessions(user_ids, agents)
    roles, acl_grants = await _read_roles_and_grants(
        tenant_ids=scope_tenant_ids,
        dataset_ids=dataset_ids,
        # user_ids is already resolved to this call's in-scope users (strings);
        # only meaningful without a tenant scope, where roles carry no other
        # link back to "in scope" at all.
        scope_user_ids=user_ids
        if scope_tenant_ids is None and scope_user_ids is not None
        else None,
    )
    # Scope memory to the in-scope datasets so it never leaks across tenants.
    memory = None
    if include_memory:
        memory = await _read_memory_graph_provenance(dataset_ids=dataset_ids)
        if memory is None:
            memory = await _read_memory_relational(dataset_ids=dataset_ids if scoped else None)

    return build_provenance_graph(
        tenants=cast(List[TenantRecord], tenants),
        users=cast(List[UserRecord], users),
        roles=cast(List[RoleRecord], roles),
        datasets=cast(List[DatasetRecord], datasets),
        files=cast(List[FileRecord], list(files.values())),
        agents=agents,
        sessions=sessions,
        acl_grants=cast(List[AclGrantRecord], acl_grants),
        memory=memory,
    )


async def visualize_memory_provenance(
    destination_file_path: Optional[str] = None,
    include_memory: bool = False,
    scope_tenant_ids: Optional[List[Any]] = None,
    scope_user_ids: Optional[List[Any]] = None,
) -> str:
    """Render the live memory-provenance graph to a self-contained HTML file.

    ``scope_tenant_ids`` / ``scope_user_ids`` restrict the graph to a tenant or
    user (see ``get_memory_provenance_graph``); pass them in multi-tenant
    deployments to avoid leaking other tenants' data.
    """
    from cognee.modules.visualization.cognee_network_visualization import (
        cognee_network_visualization,
    )

    graph_data = await get_memory_provenance_graph(
        include_memory=include_memory,
        scope_tenant_ids=scope_tenant_ids,
        scope_user_ids=scope_user_ids,
    )
    html = await cognee_network_visualization(graph_data, destination_file_path)
    if destination_file_path:
        logger.info(f"Memory provenance visualization saved at: {destination_file_path}")
    return html


async def get_memory_provenance_payload(
    include_memory: bool = False,
    scope_tenant_ids: Optional[List[Any]] = None,
    scope_user_ids: Optional[List[Any]] = None,
) -> dict:
    """The live memory-provenance graph as a JSON-safe dict, for a client
    that renders it itself instead of an embedded HTML page.

    Routes the same ``(nodes, edges)`` from ``get_memory_provenance_graph``
    through ``build_visualization_payload`` — the same preprocessing
    (``preprocess()``) the HTML path already runs the provenance graph
    through via ``cognee_network_visualization``, just packaged as a dict
    instead of interpolated into a template. The two cannot drift on the
    data for the same reason CLO-401's dataset visualization JSON and HTML
    paths cannot: both come from one ``preprocess()`` call.

    ``scope_tenant_ids`` / ``scope_user_ids``: see ``get_memory_provenance_graph``.
    """
    from cognee.modules.visualization.cognee_network_visualization import (
        build_visualization_payload,
    )

    graph_data = await get_memory_provenance_graph(
        include_memory=include_memory,
        scope_tenant_ids=scope_tenant_ids,
        scope_user_ids=scope_user_ids,
    )
    return build_visualization_payload(graph_data)
