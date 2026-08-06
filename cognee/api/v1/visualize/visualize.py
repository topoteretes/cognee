from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Union
from uuid import UUID
from cognee.modules.users.models.User import User

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
    aggregate_multi_user_graphs,
    build_visualization_payload,
    build_semantic_payload,
    build_brain_summary_payload,
)
from cognee.modules.visualization.session_events import collect_session_events
from cognee.modules.visualization.subgraph_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_SEED_TOP_K,
    fetch_visualization_graph_data,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.modules.users.methods import get_default_user
from cognee.context_global_variables import set_database_global_context_variables
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR


import asyncio


logger = get_logger()


async def fetch_visualization_data(
    user: Optional[User] = None,
    dataset: Optional[Union[str, UUID]] = DEFAULT_DATASET_NAME,
    *,
    full: bool = False,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Optional[Any] = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
    include_session_events: bool = True,
    session_ids: list = None,
) -> Tuple[Any, Optional[list]]:
    """Authorize, fetch and bound the graph data behind a visualization.

    Shared by the HTML renderer (``visualize_graph``) and the JSON payload
    builders (``visualize_graph_json``, ``visualize_semantic_json``) so every
    caller starts from the identical authorized, bounded ``graph_data`` — the
    same ``fetch_visualization_graph_data`` call, the same seed/depth/cap
    knobs, inside the same dataset-scoped database context. What each caller
    does with the result — render HTML or serialize JSON — is the only thing
    that differs from here on.

    Args, Returns: as documented on ``visualize_graph``, which this was split
    out of. Returns ``(graph_data, search_events)``.
    """
    if not user:
        user = await get_default_user()

    # Only authorize when a dataset is given. get_authorized_existing_datasets
    # expects a list, so wrap the single dataset. With no dataset the context
    # is set with None: a no-op when access control is off, and an (expected)
    # error in multi-user mode where a dataset is required.
    resolved_dataset = None
    if dataset:
        authorized = await get_authorized_existing_datasets([dataset], "read", user)
        resolved_dataset = authorized[0] if authorized else None

    graph_data = await fetch_dataset_graph_data(
        resolved_dataset,
        full=full,
        query=query,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        neighborhood_depth=neighborhood_depth,
        neighborhood_seed_top_k=neighborhood_seed_top_k,
        max_nodes=max_nodes,
    )

    search_events = None
    if include_session_events:
        from cognee.modules.visualization.session_events import collect_session_events

        search_events = await collect_session_events(user=user, session_ids=session_ids)

    return graph_data, search_events


async def fetch_dataset_graph_data(
    dataset: Optional[Any],
    *,
    full: bool = False,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Optional[Any] = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Any:
    """Bounded graph read for one already-authorized dataset (or none).

    Split out of ``fetch_visualization_data`` so ``build_brains_payload`` can
    read one dataset at a time without re-running authorization on every one
    of them — it already has the authorized list from
    ``get_all_user_permission_datasets`` and would otherwise be paying for
    the same permission check twice per dataset.

    ``dataset=None`` sets the context with ``None``: a no-op when access
    control is off, and an (expected) error in multi-user mode where a
    dataset is required.
    """
    async with set_database_global_context_variables(
        dataset.id if dataset else None,
        dataset.owner_id if dataset else None,
    ):
        graph_engine = await get_graph_engine()
        return await fetch_visualization_graph_data(
            graph_engine,
            full=full,
            query=query,
            seed_node_ids=seed_node_ids,
            recall_result=recall_result,
            neighborhood_depth=neighborhood_depth,
            seed_top_k=neighborhood_seed_top_k,
            max_nodes=max_nodes,
        )


async def visualize_graph(
    destination_file_path: str = None,
    include_session_events: bool = True,
    session_ids: list = None,
    user: Optional[User] = None,
    dataset: Optional[Union[str, UUID]] = "main_dataset",
    *,
    full: bool = False,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Optional[Any] = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> str:
    """Render the knowledge graph to a self-contained HTML file.

    By default renders a bounded subgraph around relevant seed nodes rather than
    the entire graph. Seeds are chosen by priority: ``seed_node_ids`` >
    ``recall_result`` provenance > ``query`` vector hits > highest-degree nodes.
    Pass ``full=True`` for the legacy whole-graph render.

    Args:
        destination_file_path: Where to write the HTML (default: home dir).
        include_session_events: When True (default), best-effort collect the
            backend's search and feedback history from the session layer and
            show it on the Memory tab's timeline — searches as retrieval
            spotlights, rated answers as reinforcement (improve) events.
            Collection never fails the render; an unavailable session layer
            simply yields no events.
        session_ids: Restrict event collection to these sessions. Defaults to
            the user's most recently active sessions.
        user: User whose sessions are read. Defaults to the default user.
        dataset: Dataset to render, given by name or UUID. Wrapped into a
            single-element list for get_authorized_existing_datasets; the
            first authorized match selects which user+dataset database is
            visualized. Defaults to "main_dataset" (the same default used by
            add/cognify/remember). Pass None to skip dataset resolution and
            render the current context's graph.
        full: When True, render the entire graph (legacy behavior).
        query: Optional query string; its nearest vector hits seed the subgraph.
        seed_node_ids: Explicit seed node ids for neighborhood expansion.
        recall_result: A recall/search result whose graph provenance
            (``used_graph_element_ids``) seeds the subgraph — "the subgraph
            behind this answer".
        neighborhood_depth: k-hop expansion depth around the seeds (default 2).
        neighborhood_seed_top_k: Maximum number of seed nodes (default 10).
        max_nodes: Hard cap on rendered nodes after expansion (default 500).
    """
    graph_data, search_events = await fetch_visualization_data(
        user=user,
        dataset=dataset,
        full=full,
        query=query,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        neighborhood_depth=neighborhood_depth,
        neighborhood_seed_top_k=neighborhood_seed_top_k,
        max_nodes=max_nodes,
        include_session_events=include_session_events,
        session_ids=session_ids,
    )

    graph = await cognee_network_visualization(
        graph_data, destination_file_path, search_events=search_events
    )

    if destination_file_path:
        logger.info(f"The HTML file has been stored at path: {destination_file_path}")
    else:
        logger.info(
            "The HTML file has been stored on your home directory! Navigate there with cd ~"
        )

    return graph


async def visualize_graph_json(
    include_session_events: bool = True,
    session_ids: list = None,
    user: Optional[User] = None,
    dataset: Optional[Union[str, UUID]] = DEFAULT_DATASET_NAME,
    *,
    full: bool = False,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Optional[Any] = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict:
    """The graph, as a JSON-safe dict, without the semantic layout.

    Same authorization, dataset resolution and bounded fetch as
    ``visualize_graph`` (they share ``fetch_visualization_data``), so a JSON
    caller sees exactly the subgraph the HTML page would have shown for the
    same arguments. Semantic positions/clusters are a separate, heavier call
    — see ``visualize_semantic_json`` — so a client that never opens the
    semantic tab never pays for the embedding fetch and PCA behind it.

    Returns:
        dict: every field of ``PreprocessedGraph`` (nodes, links, color_maps,
        schema_graph, schema_data, pipeline_stages, edge_classes, bundles,
        provenance_index, has_meaningful_topological_rank, memory_map) plus
        ``search_events``.
    """
    graph_data, search_events = await fetch_visualization_data(
        user=user,
        dataset=dataset,
        full=full,
        query=query,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        neighborhood_depth=neighborhood_depth,
        neighborhood_seed_top_k=neighborhood_seed_top_k,
        max_nodes=max_nodes,
        include_session_events=include_session_events,
        session_ids=session_ids,
    )

    return build_visualization_payload(graph_data, search_events=search_events)


async def visualize_semantic_json(
    user: Optional[User] = None,
    dataset: Optional[Union[str, UUID]] = DEFAULT_DATASET_NAME,
    *,
    full: bool = False,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    recall_result: Optional[Any] = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict:
    """Semantic positions and clusters for the same subgraph, computed on demand.

    Pass the same dataset/seed/depth/cap arguments used for
    ``visualize_graph_json`` to get semantic layout for the same subgraph.
    This re-runs the (cheap) fetch and preprocessing rather than reusing a
    previous call's result, because the point of splitting this out is to
    make the expensive part — embeddings for up to 2000 nodes, then PCA —
    conditional on the semantic tab actually being opened, not to cache the
    cheap part.
    """
    graph_data, _ = await fetch_visualization_data(
        user=user,
        dataset=dataset,
        full=full,
        query=query,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        neighborhood_depth=neighborhood_depth,
        neighborhood_seed_top_k=neighborhood_seed_top_k,
        max_nodes=max_nodes,
        include_session_events=False,
    )

    return await build_semantic_payload(graph_data)


async def build_brains_payload(
    user: Optional[User] = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict:
    """Every dataset the caller may read, each as a small graph preview.

    ``{dataset_id: {"name", "nodes", "links", "node_set_colors"}}`` — see
    ``build_brain_summary_payload`` for why that shape and not the full
    visualize payload per dataset.

    Built on ``get_all_user_permission_datasets`` (CLO-399's group work), the
    same union already used to decide which datasets a user can see: their
    own, their tenant's, and anything granted to a role they belong to. A
    dataset shared with a group the caller is in appears here with no extra
    code — the authorization this reuses is already group-aware.

    Each dataset's fetch is independently bounded by ``max_nodes``, same as a
    single-dataset visualization; there is no separate, larger cap for "all
    datasets at once" to defeat.
    """
    if not user:
        user = await get_default_user()

    datasets = await get_all_user_permission_datasets(user, "read")

    payload: dict = {}
    for dataset in datasets:
        graph_data = await fetch_dataset_graph_data(dataset, max_nodes=max_nodes)
        payload[str(dataset.id)] = build_brain_summary_payload(dataset.name, graph_data)

    return payload


def _as_naive_utc(value: datetime) -> datetime:
    """Normalize to match the write side.

    ``SessionQAEntry.time`` is always ``datetime.utcnow().isoformat()`` —
    naive, UTC. A ``since`` sent with a timezone offset compares fine against
    another aware datetime, but not against those naive strings once parsed
    back, so it is converted here rather than left to fail the comparison
    below with a raised ``TypeError`` at an unexpected place.
    """
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _event_time(event: Dict[str, Any]) -> Optional[datetime]:
    raw = event.get("time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


async def get_live_events(
    dataset_id: UUID,
    since: Optional[datetime] = None,
    user: Optional[User] = None,
) -> Dict[str, Any]:
    """Delta of search/improve events since a cursor, for the Memory tab's
    live timeline.

    ``dataset_id`` gates who may call this — the same read-permission check
    every other visualize endpoint runs (``get_authorized_existing_datasets``)
    — not which events come back. Session events are collected per *user*,
    not per dataset, matching how ``visualize_graph``'s own
    ``include_session_events`` already embeds the same events into whichever
    dataset's page happens to be open. Scoping the event *content* to one
    dataset would mean checking each event's referenced node/edge ids against
    that dataset's graph membership — real extra cost that this endpoint's
    one job (avoid recomputing the whole graph payload every ~1.5s just to
    refresh a timeline) does not need, and it would make this endpoint
    disagree with the very same events already shown in ``search_events`` on
    ``/visualize/json`` for that dataset.

    ``since=None`` returns every available event — the first call, before a
    client has a cursor of its own.

    Returns ``{"events": [...], "cursor": <ISO datetime or None>}``. The
    filter is strict (``>``, not ``>=``): pass the previous response's
    ``cursor`` straight back as the next call's ``since`` and nothing is
    delivered twice. ``cursor`` is the newest event's own timestamp string,
    not a re-derived one, so it can never drift from what the client actually
    received.

    Raises:
        PermissionDeniedError: dataset does not exist, or the caller lacks
            read permission on it.
    """
    if not user:
        user = await get_default_user()

    authorized = await get_authorized_existing_datasets([dataset_id], "read", user)
    if not authorized:
        raise PermissionDeniedError(message="Not authorized to read this dataset")

    events = await collect_session_events(user=user)

    if since is not None:
        cutoff = _as_naive_utc(since)
        events = [
            event
            for event in events
            if (event_time := _event_time(event)) is not None and event_time > cutoff
        ]

    if events:
        cursor = events[-1]["time"]
    else:
        cursor = since.isoformat() if since is not None else None

    return {"events": events, "cursor": cursor}


async def visualize_multi_user_graph(
    user_dataset_pairs: List[Tuple[Any, Any]],
    destination_file_path: str = None,
) -> Any:
    """Generate a visualization combining graph data from multiple user+dataset pairs.

    Args:
        user_dataset_pairs: list of (User, Dataset) tuples to aggregate.
        destination_file_path: optional path to save the HTML output.

    Returns:
        The HTML visualization string.
    """
    graph_data = await aggregate_multi_user_graphs(user_dataset_pairs)

    graph = await cognee_network_visualization(graph_data, destination_file_path)

    if destination_file_path:
        logger.info(f"Multi-user visualization saved at: {destination_file_path}")
    else:
        logger.info(
            "Multi-user visualization saved to your home directory! Navigate there with cd ~"
        )

    return graph


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(visualize_graph())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
