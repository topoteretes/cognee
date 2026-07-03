"""Assembly tests for the semantic tab — offline, embedding seam mocked.

Pins the wiring contract: the semantic tab, layout, and view JS are injected;
the position/cluster data tokens are substituted (real JSON when embeddings
exist, ``null`` when they don't); and no ``__TOKEN__`` placeholder ever leaks.
The embedding fetch is mocked at the orchestrator seam so these tests need no
vector store or LLM.
"""

import asyncio
import re

from cognee.modules.visualization import cognee_network_visualization as orch
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)

TOKEN_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")


def _graph():
    nodes = [
        ("11111111-1111-1111-1111-111111111111", {"type": "Entity", "name": "Ada"}),
        ("22222222-2222-2222-2222-222222222222", {"type": "Entity", "name": "Alan"}),
        ("33333333-3333-3333-3333-333333333333", {"type": "Entity", "name": "London"}),
    ]
    edges = [
        (nodes[0][0], nodes[1][0], "knows", {}),
        (nodes[0][0], nodes[2][0], "lived_in", {}),
    ]
    return (nodes, edges)


def _render(tmp_path, monkeypatch, embeddings):
    async def fake_fetch(nodes, **kwargs):
        return embeddings

    monkeypatch.setattr(orch, "fetch_node_embeddings", fake_fetch)
    return asyncio.run(cognee_network_visualization(_graph(), str(tmp_path / "out.html")))


def test_semantic_tab_and_view_are_wired(tmp_path, monkeypatch):
    html = _render(tmp_path, monkeypatch, {})
    assert 'data-view="semantic"' in html
    assert 'id="semantic-view"' in html
    assert "window._renderSemanticView" in html
    assert "window._semanticPositions" in html
    # Core seam: the pure decision layer is injected and the shell consumes it.
    # (Behavior of the core is covered by the Node suite, not by string presence.)
    assert "root.SemanticCore = api" in html
    assert "const Core = window.SemanticCore" in html
    assert "Core.styleFor" in html
    # Type-filter wiring: the legend follows the color mode and isolates a type.
    assert "state.isolatedType" in html
    assert "state.colorBy === 'type'" in html
    # Layout-toggle wiring: Semantic/Structural buttons + the force sim path.
    assert 'class="sem-layout-btn' in html
    assert 'data-layout="structural"' in html
    assert "structuralPositions" in html
    assert "d3.forceSimulation" in html
    # Recall-overlay wiring: the query dropdown + the search-events consumer.
    assert 'id="semantic-recall"' in html
    assert "SEARCH_EVENTS" in html
    assert "state.recall" in html


def test_no_token_leaks_with_embeddings(tmp_path, monkeypatch):
    emb = {
        "11111111-1111-1111-1111-111111111111": [1.0, 0.0, 0.0],
        "22222222-2222-2222-2222-222222222222": [0.9, 0.1, 0.0],
        "33333333-3333-3333-3333-333333333333": [-1.0, 0.0, 0.0],
    }
    html = _render(tmp_path, monkeypatch, emb)
    assert TOKEN_RE.findall(html) == []
    # Positions and clusters are real JSON, not the literal null.
    assert "window._semanticPositions = null" not in html
    assert "const CLUSTERS = null" not in html
    # Cluster payload structure made it into the embedded JSON.
    assert '"neighbors"' in html
    assert '"node_cluster"' in html


def test_no_token_leaks_without_embeddings(tmp_path, monkeypatch):
    html = _render(tmp_path, monkeypatch, {})
    assert TOKEN_RE.findall(html) == []
    # Empty state: both semantic data tokens collapse to null.
    assert "window._semanticPositions = null" in html
    assert "const CLUSTERS = null" in html


def test_semantic_failure_never_breaks_render(tmp_path, monkeypatch):
    async def boom(nodes, **kwargs):
        raise RuntimeError("vector store down")

    monkeypatch.setattr(orch, "fetch_node_embeddings", boom)
    html = asyncio.run(cognee_network_visualization(_graph(), str(tmp_path / "out.html")))
    # Classic render still completes; semantic tab falls back to the empty state.
    assert html.startswith("<!DOCTYPE html>")
    assert TOKEN_RE.findall(html) == []
    assert "window._semanticPositions = null" in html
