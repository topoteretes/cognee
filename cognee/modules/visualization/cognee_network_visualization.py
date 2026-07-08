"""Cognee graph visualization — orchestrator.

Single entry point ``cognee_network_visualization(graph_data, ...)``:

  1. Preprocesses the raw graph into a renderer-facing snapshot
     (``preprocessor.preprocess``).
  2. Reads the HTML shell from ``template.html``.
  3. Asks each view module (``views/*``) and layout module (``layouts/*``)
     to emit its JS chunk.
  4. Substitutes ``__TOKEN__`` placeholders — JS chunks first, then the
     JSON data payloads — and writes the final HTML.

The split into views/ + layouts/ + preprocessor + template is the
Phase 1c refactor described in the plan
(``/Users/vasilije/.claude/plans/floating-snacking-waffle.md``). It
preserves the public API for backward compatibility — every existing
caller of ``cognee_network_visualization`` or
``aggregate_multi_user_graphs`` continues to work without change.
"""

import json
import os
from typing import Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.modules.visualization.preprocessor import preprocess
from cognee.modules.visualization.views import (
    inspector,
    memory_map,
    schema_view,
    semantic_map,
    story_view,
    ui_chrome,
)
from cognee.modules.visualization.layouts import pipeline_layout, semantic_layout
from cognee.modules.visualization.embedding_join import fetch_node_embeddings, select_nodes
from cognee.modules.visualization.semantic_clusters import compute_clusters

logger = get_logger()


async def _semantic_payload(pre):
    """Best-effort semantic positions + clusters. Never blocks the classic render.

    Returns ``(positions, clusters)`` or ``(None, None)`` when they can't be
    computed — in which case the semantic tab shows a friendly empty state.
    Bounded: the layout and clustering run on the same capped node sample as
    the embedding fetch (``select_nodes``).
    """
    try:
        nodes = select_nodes(pre.nodes)
        embeddings = await fetch_node_embeddings(nodes)
        if not embeddings:
            return None, None
        # PCA is the deterministic zero-dependency default; set
        # SEMANTIC_MAP_PROJECTION=umap to opt in when umap-learn is installed
        # (silently falls back to PCA when it isn't).
        method = os.environ.get("SEMANTIC_MAP_PROJECTION", "pca").strip().lower()
        positions = semantic_layout.compute_positions(nodes, pre.links, embeddings, method=method)
        clusters = compute_clusters(nodes, embeddings)
        return positions, clusters
    except Exception as exc:
        logger.warning("Semantic map: payload computation failed (%s); tab shows empty state.", exc)
        return None, None


_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.html")


def _safe_json_embed(obj) -> str:
    """JSON-encode while neutralising ``</`` so the result is safe to
    embed inside a ``<script>`` element."""
    return json.dumps(obj).replace("</", "<\\/")


def _read_template() -> str:
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


async def cognee_network_visualization(
    graph_data,
    destination_file_path: Optional[str] = None,
    schema_data: Optional[dict] = None,
    search_events: Optional[list] = None,
) -> str:
    """Render the graph to a self-contained HTML file and return the HTML.

    Args:
        graph_data: ``(nodes_data, edges_data)`` tuple as produced by
            ``GraphDBInterface.get_graph_data()``.
        destination_file_path: Where to write the HTML.  Defaults to
            ``~/graph_visualization.html``.
        schema_data: Optional pre-built schema payload.  When absent, the
            preprocessor derives a type-graph from the nodes/links it sees.
        search_events: Optional list of operation events shown on the Memory
            tab's timeline. Two kinds::

                {"kind": "search", "time": "2026-06-10T10:31:02",
                 "qa_id": "...", "question": "...", "answer": "...",
                 "node_ids": ["uuid", ...], "edge_ids": ["uuid", ...]}

                {"kind": "improve", "time": "...", "qa_id": "...",
                 "question": "...", "rating": 5, "feedback_text": "...",
                 "applied": true,
                 "node_ids": ["uuid", ...], "edge_ids": ["uuid", ...]}

            ``search`` renders a retrieval spotlight; ``improve`` renders a
            reinforcement overlay (the elements whose feedback_weight the
            rated answer updated — green for positive ratings, amber for
            negative). Entries without ``kind`` default to ``search``.

            No cache is read here — ``visualize_graph(include_session_events
            =True)`` collects these automatically from the session layer via
            ``cognee.modules.visualization.session_events``; pass them
            explicitly only for custom pipelines.

    Returns:
        The full HTML as a string.
    """
    pre = preprocess(graph_data, schema_data=schema_data)

    # Best-effort semantic layout; guarded so it never blocks the classic render.
    semantic_positions, semantic_clusters = await _semantic_payload(pre)

    html = _read_template()

    # 1) JS chunks: ordered so the first script block (ui_chrome + schema)
    #    runs before the main story-view IIFE in the second block.
    html = html.replace("__UI_CHROME_JS__", ui_chrome.emit_js(pre))
    html = html.replace("__SCHEMA_VIEW_JS__", schema_view.emit_js(pre))
    html = html.replace("__STORY_VIEW_JS__", story_view.emit_js(pre))
    html = html.replace("__PIPELINE_LAYOUT_JS__", pipeline_layout.emit_js(pre))
    html = html.replace("__INSPECTOR_JS__", inspector.emit_js(pre))
    html = html.replace("__MEMORY_VIEW_JS__", memory_map.emit_js(pre))
    html = html.replace("__SEMANTIC_LAYOUT_JS__", semantic_layout.emit_js(pre))
    html = html.replace("__SEMANTIC_VIEW_JS__", semantic_map.emit_js(pre))

    # 2) Data tokens: substituted last so JSON-embedded ``__SCHEMA_GRAPH_DATA__``
    #    inside the schema JS chunk gets resolved correctly.
    html = html.replace("__NODES_DATA__", _safe_json_embed(pre.nodes))
    html = html.replace("__LINKS_DATA__", _safe_json_embed(pre.links))
    html = html.replace("__TASK_COLORS__", _safe_json_embed(pre.color_maps["task"]))
    html = html.replace("__PIPELINE_COLORS__", _safe_json_embed(pre.color_maps["pipeline"]))
    html = html.replace("__NODESET_COLORS__", _safe_json_embed(pre.color_maps["node_set"]))
    html = html.replace("__USER_COLORS__", _safe_json_embed(pre.color_maps["user"]))
    html = html.replace(
        "__SCHEMA_DATA__",
        _safe_json_embed(schema_data) if schema_data else "null",
    )
    html = html.replace(
        "__SCHEMA_GRAPH_DATA__",
        _safe_json_embed(pre.schema_graph or {"nodes": [], "links": []}),
    )
    # Unconditional, JSON-fallback substitutions: a leaked __MEMORY_DATA__ /
    # __SEARCH_EVENTS__ token would fail the no-placeholder assembly test.
    html = html.replace("__MEMORY_DATA__", _safe_json_embed(pre.memory_map or {}))
    html = html.replace("__SEARCH_EVENTS__", _safe_json_embed(search_events or []))
    # Semantic tokens: null when there are no embeddings, so the tab renders a
    # friendly empty state without leaving a placeholder behind.
    html = html.replace(
        "__SEMANTIC_POSITIONS__",
        _safe_json_embed(semantic_positions) if semantic_positions else "null",
    )
    html = html.replace(
        "__SEMANTIC_CLUSTERS__",
        _safe_json_embed(semantic_clusters) if semantic_clusters else "null",
    )

    if not destination_file_path:
        destination_file_path = os.path.join(os.path.expanduser("~"), "graph_visualization.html")

    dir_path = os.path.dirname(destination_file_path)
    file_name = os.path.basename(destination_file_path)
    file_storage = LocalFileStorage(dir_path)
    await file_storage.store(file_name, html, overwrite=True)

    logger.info(f"Graph visualization saved as {destination_file_path}")

    return html


async def aggregate_multi_user_graphs(user_dataset_pairs):
    """Aggregate graph data from multiple user+dataset pairs into a single graph.

    Args:
        user_dataset_pairs: list of ``(user, dataset)`` tuples where ``user``
            is a User model instance and ``dataset`` is a Dataset model instance.

    Returns:
        A tuple ``(nodes_data, edges_data)`` in the same format as
        ``get_graph_data()``, with nodes tagged with ``source_user`` from
        the owning user's email.
    """
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.context_global_variables import set_database_global_context_variables

    all_nodes: dict = {}
    all_edges: list = []
    seen_edges: set = set()

    for user, dataset in user_dataset_pairs:
        async with set_database_global_context_variables(dataset.id, user.id):
            graph_engine = await get_graph_engine()
            nodes_data, edges_data = await graph_engine.get_graph_data()

            user_label = getattr(user, "email", None) or str(user.id)

            for node_id, node_info in nodes_data:
                node_key = str(node_id)
                if node_key not in all_nodes:
                    node_info = (
                        dict(node_info) if not isinstance(node_info, dict) else node_info.copy()
                    )
                    if not node_info.get("source_user"):
                        node_info["source_user"] = user_label
                    all_nodes[node_key] = (node_id, node_info)

            for edge in edges_data:
                source, target, relation = edge[0], edge[1], edge[2]
                edge_key = (str(source), str(target), relation)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    all_edges.append(edge)

    return (list(all_nodes.values()), all_edges)
