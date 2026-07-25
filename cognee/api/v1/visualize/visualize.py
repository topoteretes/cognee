from typing import Any, List, Tuple, Optional, Union
from uuid import UUID
from cognee.modules.users.models.User import User

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
    aggregate_multi_user_graphs,
)
from cognee.modules.visualization.subgraph_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_SEED_TOP_K,
    fetch_visualization_graph_data,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user
from cognee.context_global_variables import set_database_global_context_variables
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR


import asyncio


logger = get_logger()


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
    live: bool = False,
    live_events_url: Optional[str] = None,
    include_actors: bool = True,
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
        live: When True, the rendered page polls the backend's
            ``/api/v1/visualize/live-events`` endpoint and spotlights each new
            search's retrieved subgraph on the Memory tab as it happens —
            keep the page open and run searches to watch them light up.
            Requires the cognee API server to be running (and reachable under
            ``live_events_url``); searches must run with session memory
            enabled (the default) for their provenance to be recorded.
        live_events_url: Overrides the polled URL. Defaults to
            ``http://localhost:8000/api/v1/visualize/live-events``. Setting
            this implies ``live=True``.
        include_actors: When True (default), the render includes the actor
            layer — the user, their registered agent connections, and the
            dataset(s) as first-class nodes wired to the documents they
            contain (User —operates→ Agent —reads/writes→ Dataset —contains→
            documents). Best-effort: an unavailable layer renders nothing.
    """
    if not user:
        user = await get_default_user()

    # Only authorize when a dataset is given. get_authorized_existing_datasets
    # expects a list, so wrap the single dataset. With no dataset the context
    # is set with None: a no-op when access control is off, and an (expected)
    # error in multi-user mode where a dataset is required.
    if dataset:
        dataset = await get_authorized_existing_datasets([dataset], "read", user)

    async with set_database_global_context_variables(
        dataset[0].id if dataset else None,
        dataset[0].owner_id if dataset else None,
    ):
        graph_engine = await get_graph_engine()
        graph_data = await fetch_visualization_graph_data(
            graph_engine,
            full=full,
            query=query,
            seed_node_ids=seed_node_ids,
            recall_result=recall_result,
            neighborhood_depth=neighborhood_depth,
            seed_top_k=neighborhood_seed_top_k,
            max_nodes=max_nodes,
        )

        search_events = None
        if include_session_events:
            from cognee.modules.visualization.session_events import collect_session_events

            search_events = await collect_session_events(user=user, session_ids=session_ids)
            # Agents can be operated by OTHER users in scope (e.g. a
            # colleague's agent) — fold their sessions in too so agent
            # cards can show their own conversation history. Best-effort
            # and same-scope only.
            if include_actors:
                try:
                    from sqlalchemy import select as _select

                    from cognee.infrastructure.databases.relational import (
                        get_relational_engine,
                    )
                    from cognee.modules.users.models import User as UserModel

                    engine = get_relational_engine()
                    async with engine.get_async_session() as db:
                        stmt = _select(UserModel)
                        if getattr(user, "tenant_id", None):
                            stmt = stmt.where(UserModel.tenant_id == user.tenant_id)
                        others = [
                            u
                            for u in (await db.execute(stmt)).scalars().all()
                            if str(u.id) != str(user.id)
                            and not getattr(u, "parent_user_id", None)
                            and not str(getattr(u, "email", "")).endswith("@cognee.agent")
                        ]
                    seen_keys = {
                        (e.get("session_id"), e.get("qa_id")) for e in (search_events or [])
                    }
                    for other in others:
                        for event in await collect_session_events(user=other):
                            key = (event.get("session_id"), event.get("qa_id"))
                            if key not in seen_keys:
                                seen_keys.add(key)
                                search_events.append(event)
                except Exception as error:  # noqa: BLE001 — extra sessions are best-effort
                    logger.debug("Cross-user session events skipped: %s", error)

        if include_actors and dataset:
            from cognee.api.v1.visualize.memory_provenance import get_actor_overlay

            graph_node_ids = {str(node_id) for node_id, _ in graph_data[0]}
            overlay_nodes, overlay_edges = await get_actor_overlay(dataset, user, graph_node_ids)
            if overlay_nodes:
                graph_data = (
                    list(graph_data[0]) + overlay_nodes,
                    list(graph_data[1]) + overlay_edges,
                )

        # Every OTHER brain the user can read gets its graph embedded too, so
        # the Business view can switch brains client-side instead of telling
        # the user a personal brain "lives in another graph".
        extra_brains = {}
        if include_actors and dataset:
            try:
                from cognee.modules.users.permissions.methods import (
                    get_all_user_permission_datasets,
                )

                readable = await get_all_user_permission_datasets(user, "read")
                rendered_id = str(dataset[0].id)
                seen = {rendered_id}
                for other in readable:
                    other_id = str(other.id)
                    if other_id in seen:
                        continue
                    seen.add(other_id)
                    async with set_database_global_context_variables(other.id, other.owner_id):
                        other_engine = await get_graph_engine()
                        other_graph = await other_engine.get_graph_data()
                    extra_brains[other_id] = {"name": other.name, "graph_data": other_graph}
            except Exception as error:  # noqa: BLE001 — extra brains are best-effort
                logger.warning("Extra brains unavailable; rendering without them: %s", error)

        if live and not live_events_url:
            live_events_url = "http://localhost:8000/api/v1/visualize/live-events"

        graph = await cognee_network_visualization(
            graph_data,
            destination_file_path,
            search_events=search_events,
            live_events_url=live_events_url,
            extra_brains=extra_brains,
        )

        if destination_file_path:
            logger.info(f"The HTML file has been stored at path: {destination_file_path}")
        else:
            logger.info(
                "The HTML file has been stored on your home directory! Navigate there with cd ~"
            )

        return graph


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
