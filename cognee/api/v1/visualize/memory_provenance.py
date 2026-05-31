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

Two entry points:
    * ``build_provenance_graph(...)`` — pure, side-effect-free assembly from
      plain records (unit-testable).
    * ``get_memory_provenance_graph(...)`` — async reader that pulls live data
      from the relational layer and calls the builder.
"""

from typing import Any, Dict, List, Optional, Tuple

from cognee.shared.logging_utils import get_logger

logger = get_logger()

Node = Tuple[str, Dict[str, Any]]
EdgeData = Tuple[str, str, str, Dict[str, Any]]


def build_provenance_graph(
    *,
    tenants: Optional[List[Dict[str, Any]]] = None,
    users: Optional[List[Dict[str, Any]]] = None,
    datasets: Optional[List[Dict[str, Any]]] = None,
    files: Optional[List[Dict[str, Any]]] = None,
    agents: Optional[List[Dict[str, Any]]] = None,
    sessions: Optional[List[Dict[str, Any]]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Node], List[EdgeData]]:
    """Assemble actor/ownership/session records into a ``(nodes, edges)`` graph.

    Record shapes (all ids are strings):
        tenants:  {"id", "name"}
        users:    {"id", "name", "tenant_ids": [..]}
        datasets: {"id", "name", "owner_id", "tenant_id"}
        files:    {"id", "name", "dataset_ids": [..], "dataset_name"?}
        agents:   {"id", "name", "user_id", "session_id"?,
                   "datasets": [{"dataset_id", "role": "read"|"read_write"}]}
        sessions: {"id", "name", "user_id", "dataset_id", "agent_id"?}
        memory:   optional {"nodes": [(id, props)], "edges": [(s, t, rel, props)],
                            "links": [{"node_id", "data_id", "dataset_id"}]}

    Node ids are namespaced (``user:<id>`` etc.) so the actor layers never
    collide with each other or with raw memory-node ids.
    """
    tenants = tenants or []
    users = users or []
    datasets = datasets or []
    files = files or []
    agents = agents or []
    sessions = sessions or []

    nodes: Dict[str, Node] = {}
    edges: List[EdgeData] = []
    seen_edges = set()

    def add_node(node_id: str, node_type: str, name: str, **extra) -> None:
        if node_id not in nodes:
            props = {"type": node_type, "name": name}
            props.update({k: v for k, v in extra.items() if v is not None})
            nodes[node_id] = (node_id, props)

    def add_edge(source: str, target: str, relation: str) -> None:
        if source in nodes and target in nodes:
            key = (source, target, relation)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append((source, target, relation, {}))

    # ── Actor / ownership nodes ──────────────────────────────────────
    for tenant in tenants:
        add_node(f"tenant:{tenant['id']}", "Tenant", tenant.get("name") or "Tenant")
    for user in users:
        add_node(f"user:{user['id']}", "User", user.get("name") or str(user["id"]))
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

    for dataset in datasets:
        if dataset.get("owner_id"):
            add_edge(f"user:{dataset['owner_id']}", f"dataset:{dataset['id']}", "owns")

    for file in files:
        for dataset_id in file.get("dataset_ids") or []:
            add_edge(f"dataset:{dataset_id}", f"file:{file['id']}", "contains")

    for agent in agents:
        if agent.get("user_id"):
            add_edge(f"user:{agent['user_id']}", f"agent:{agent['id']}", "operates")
        for ref in agent.get("datasets") or []:
            dataset_id = ref.get("dataset_id")
            if not dataset_id:
                continue
            add_edge(f"agent:{agent['id']}", f"dataset:{dataset_id}", "reads")
            if ref.get("role") == "read_write":
                add_edge(f"agent:{agent['id']}", f"dataset:{dataset_id}", "writes")
        if agent.get("session_id"):
            add_edge(f"agent:{agent['id']}", f"session:{agent['session_id']}", "wrote")

    for sess in sessions:
        if sess.get("agent_id"):
            add_edge(f"agent:{sess['agent_id']}", f"session:{sess['id']}", "wrote")
        if sess.get("dataset_id"):
            add_edge(f"session:{sess['id']}", f"dataset:{sess['dataset_id']}", "recorded_in")

    # ── Optional memory layer ────────────────────────────────────────
    if memory:
        for node_id, props in memory.get("nodes") or []:
            nid = str(node_id)
            if nid not in nodes:
                nodes[nid] = (nid, dict(props))
        for source, target, relation, eprops in memory.get("edges") or []:
            s, t = str(source), str(target)
            if s in nodes and t in nodes:
                key = (s, t, relation)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append((s, t, relation or "related", dict(eprops or {})))
        for link in memory.get("links") or []:
            file_id = f"file:{link['data_id']}" if link.get("data_id") else None
            node_id = str(link["node_id"]) if link.get("node_id") else None
            if file_id and node_id:
                add_edge(file_id, node_id, "mentions")

    return list(nodes.values()), edges


# ── Live relational readers ──────────────────────────────────────────────────


async def _read_agents(user_ids: List[str]) -> List[Dict[str, Any]]:
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

    agents: List[Dict[str, Any]] = []
    seen = set()
    for conn in connections:
        if conn.id in seen:
            continue
        seen.add(conn.id)
        refs = [
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


async def _read_sessions(user_ids: List[str], agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort enumeration of session records."""
    from uuid import UUID

    agent_by_session = {a["session_id"]: a["id"] for a in agents if a.get("session_id")}
    sessions: List[Dict[str, Any]] = []
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


async def _read_memory_relational(limit: int = 5000) -> Optional[Dict[str, Any]]:
    """Read extracted memory from the relational ``nodes``/``edges`` tables.

    Avoids the knowledge-graph backend entirely, so it works when that backend
    is unavailable. Returns None when there is no memory to show.
    """
    try:
        from sqlalchemy import select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.graph.models.Node import Node as NodeRow
        from cognee.modules.graph.models.Edge import Edge as EdgeRow
    except Exception as error:  # pragma: no cover - models unavailable
        logger.debug(f"relational memory models unavailable: {error}")
        return None

    nodes: List[Node] = []
    edges: List[EdgeData] = []
    links: List[Dict[str, Any]] = []
    try:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            for row in (await session.execute(select(NodeRow).limit(limit))).scalars().all():
                node_id = str(row.id)
                nodes.append((node_id, {"type": row.type or "Node", "name": row.label or str(row.slug)}))
                if row.data_id is not None:
                    links.append(
                        {
                            "node_id": node_id,
                            "data_id": str(row.data_id),
                            "dataset_id": str(row.dataset_id) if row.dataset_id is not None else None,
                        }
                    )
            for row in (await session.execute(select(EdgeRow).limit(limit * 4))).scalars().all():
                edges.append(
                    (str(row.source_node_id), str(row.destination_node_id), row.relationship_name or "related", {})
                )
    except Exception as error:  # pragma: no cover - defensive
        logger.debug(f"relational memory read skipped: {error}")
        return None

    if not nodes:
        return None
    return {"nodes": nodes, "edges": edges, "links": links}


async def get_memory_provenance_graph(
    include_memory: bool = False,
) -> Tuple[List[Node], List[EdgeData]]:
    """Read live relational data and project it into a provenance ``(nodes, edges)``.

    Args:
        include_memory: when True, fold in the extracted memory from the
            relational ``nodes``/``edges`` tables and link it to source files.

    Returns:
        ``(nodes, edges)`` in ``get_graph_data()`` shape. Empty-ish when the
        database has no actor/ownership rows yet.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import joinedload, selectinload

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.users.models import Tenant, User

    tenants: List[Dict[str, Any]] = []
    users: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    files: Dict[str, Dict[str, Any]] = {}

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        for tenant in (await session.execute(select(Tenant))).scalars().all():
            tenants.append({"id": str(tenant.id), "name": tenant.name})

        user_rows = (
            await session.execute(select(User).options(selectinload(User.tenants)))
        ).scalars().all()
        for user in user_rows:
            tenant_ids = [str(t.id) for t in (user.tenants or [])]
            if getattr(user, "tenant_id", None):
                tenant_ids.append(str(user.tenant_id))
            users.append(
                {
                    "id": str(user.id),
                    "name": getattr(user, "email", None) or str(user.id),
                    "tenant_ids": sorted(set(tenant_ids)),
                }
            )

        dataset_rows = (
            (await session.execute(select(Dataset).options(joinedload(Dataset.data))))
            .unique()
            .scalars()
            .all()
        )
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
                    {"id": file_id, "name": data.name, "dataset_ids": [], "dataset_name": dataset.name},
                )
                record["dataset_ids"].append(str(dataset.id))

    user_ids = [u["id"] for u in users]
    agents = await _read_agents(user_ids)
    sessions = await _read_sessions(user_ids, agents)
    memory = await _read_memory_relational() if include_memory else None

    return build_provenance_graph(
        tenants=tenants,
        users=users,
        datasets=datasets,
        files=list(files.values()),
        agents=agents,
        sessions=sessions,
        memory=memory,
    )


async def visualize_memory_provenance(
    destination_file_path: Optional[str] = None,
    include_memory: bool = False,
) -> str:
    """Render the live memory-provenance graph to a self-contained HTML file."""
    from cognee.modules.visualization.cognee_network_visualization import (
        cognee_network_visualization,
    )

    graph_data = await get_memory_provenance_graph(include_memory=include_memory)
    html = await cognee_network_visualization(graph_data, destination_file_path)
    if destination_file_path:
        logger.info(f"Memory provenance visualization saved at: {destination_file_path}")
    return html
