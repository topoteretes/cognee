"""Tests for cognee.modules.visualization.layouts.semantic_layout.

Runs fully offline — no LLM, no vector-store connection required.
Embeddings are injected via monkeypatching so the suite is deterministic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal PreprocessedGraph stub (mirrors the real dataclass interface)
# ---------------------------------------------------------------------------

@dataclass
class _FakePreprocessed:
    nodes: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nodes(n: int) -> list[dict]:
    return [{"id": f"node-{i}", "label": f"Node {i}", "type": "Entity"} for i in range(n)]


def _random_embedding(dim: int = 16, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


def _fake_embeddings(node_ids: list[str], dim: int = 16) -> dict[str, list[float]]:
    return {nid: _random_embedding(dim, seed=hash(nid) % (2**31)) for nid in node_ids}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPCA2D:
    """Unit-test the PCA projection in isolation."""

    def test_output_shape(self):
        from cognee.modules.visualization.layouts.semantic_layout import _pca_2d

        matrix = [_random_embedding(32, seed=i) for i in range(10)]
        result = _pca_2d(matrix)

        assert len(result) == 10
        for x, y in result:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_coordinates_normalised(self):
        from cognee.modules.visualization.layouts.semantic_layout import _pca_2d

        matrix = [_random_embedding(16, seed=i) for i in range(20)]
        result = _pca_2d(matrix)

        xs = [p[0] for p in result]
        ys = [p[1] for p in result]

        assert min(xs) >= -1.001
        assert max(xs) <= 1.001
        assert min(ys) >= -1.001
        assert max(ys) <= 1.001

    def test_single_dimension_matrix(self):
        """PCA on 1-D embeddings should still return 2D points."""
        from cognee.modules.visualization.layouts.semantic_layout import _pca_2d

        matrix = [[float(i)] for i in range(5)]
        result = _pca_2d(matrix)
        assert len(result) == 5

    def test_deterministic(self):
        from cognee.modules.visualization.layouts.semantic_layout import _pca_2d

        matrix = [_random_embedding(8, seed=i) for i in range(8)]
        assert _pca_2d(matrix) == _pca_2d(matrix)


class TestTopologyFallback:
    """Unit-test the fallback position for un-embedded nodes."""

    def test_neighbour_centroid(self):
        from cognee.modules.visualization.layouts.semantic_layout import _topology_fallback

        id_to_pos = {"a": (10.0, 20.0), "b": (30.0, 40.0)}
        links = [{"source": "a", "target": "orphan"}, {"source": "b", "target": "orphan"}]
        result = _topology_fallback(["orphan"], id_to_pos, links)

        # Centroid of (10,20) and (30,40) is (20, 30) — allow jitter ±0.05
        assert abs(result["orphan"][0] - 20.0) < 1.0
        assert abs(result["orphan"][1] - 30.0) < 1.0

    def test_no_neighbours_gets_random_position(self):
        from cognee.modules.visualization.layouts.semantic_layout import _topology_fallback

        result = _topology_fallback(["lonely"], {}, [])
        assert "lonely" in result
        x, y = result["lonely"]
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0

    def test_deterministic(self):
        from cognee.modules.visualization.layouts.semantic_layout import _topology_fallback

        r1 = _topology_fallback(["a", "b", "c"], {}, [])
        r2 = _topology_fallback(["a", "b", "c"], {}, [])
        assert r1 == r2


class TestComputeSemanticPositions:
    """Integration test for the main async entry point."""

    @pytest.mark.asyncio
    async def test_returns_position_for_every_node(self):
        from cognee.modules.visualization.layouts.semantic_layout import (
            compute_semantic_positions,
        )

        nodes = _make_nodes(10)
        pre = _FakePreprocessed(nodes=nodes)
        node_ids = [n["id"] for n in nodes]
        fake_emb = _fake_embeddings(node_ids, dim=16)

        with patch(
            "cognee.modules.visualization.layouts.semantic_layout._fetch_embeddings",
            new=AsyncMock(return_value=fake_emb),
        ):
            positions = await compute_semantic_positions(pre)

        assert set(positions.keys()) == set(node_ids)
        for nid, (x, y) in positions.items():
            assert isinstance(x, float)
            assert isinstance(y, float)
            assert not math.isnan(x)
            assert not math.isnan(y)

    @pytest.mark.asyncio
    async def test_positions_scaled_to_spread(self):
        from cognee.modules.visualization.layouts.semantic_layout import (
            _SPREAD,
            compute_semantic_positions,
        )

        nodes = _make_nodes(8)
        pre = _FakePreprocessed(nodes=nodes)
        node_ids = [n["id"] for n in nodes]
        fake_emb = _fake_embeddings(node_ids, dim=8)

        with patch(
            "cognee.modules.visualization.layouts.semantic_layout._fetch_embeddings",
            new=AsyncMock(return_value=fake_emb),
        ):
            positions = await compute_semantic_positions(pre)

        for x, y in positions.values():
            assert abs(x) <= _SPREAD * 1.01
            assert abs(y) <= _SPREAD * 1.01

    @pytest.mark.asyncio
    async def test_missing_embeddings_handled_gracefully(self):
        """Nodes without embeddings should still receive a position."""
        from cognee.modules.visualization.layouts.semantic_layout import (
            compute_semantic_positions,
        )

        nodes = _make_nodes(6)
        pre = _FakePreprocessed(nodes=nodes)
        node_ids = [n["id"] for n in nodes]

        # Only provide embeddings for the first 3
        partial_emb = _fake_embeddings(node_ids[:3], dim=16)

        with patch(
            "cognee.modules.visualization.layouts.semantic_layout._fetch_embeddings",
            new=AsyncMock(return_value=partial_emb),
        ):
            positions = await compute_semantic_positions(pre)

        assert set(positions.keys()) == set(node_ids)

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty(self):
        from cognee.modules.visualization.layouts.semantic_layout import (
            compute_semantic_positions,
        )

        pre = _FakePreprocessed(nodes=[], links=[])

        with patch(
            "cognee.modules.visualization.layouts.semantic_layout._fetch_embeddings",
            new=AsyncMock(return_value={}),
        ):
            positions = await compute_semantic_positions(pre)

        assert positions == {}

    @pytest.mark.asyncio
    async def test_too_few_nodes_for_projection_uses_random(self):
        """Fewer than _MIN_NODES_FOR_PROJECTION embedded nodes → random layout."""
        from cognee.modules.visualization.layouts.semantic_layout import (
            compute_semantic_positions,
        )

        nodes = _make_nodes(2)
        pre = _FakePreprocessed(nodes=nodes)
        node_ids = [n["id"] for n in nodes]
        fake_emb = _fake_embeddings(node_ids, dim=4)

        with patch(
            "cognee.modules.visualization.layouts.semantic_layout._fetch_embeddings",
            new=AsyncMock(return_value=fake_emb),
        ):
            positions = await compute_semantic_positions(pre)

        assert len(positions) == 2


class TestEmitJs:
    """Smoke-test that emit_js returns a non-empty string."""

    def test_returns_string(self):
        from cognee.modules.visualization.layouts.semantic_layout import emit_js

        result = emit_js()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_semantic_positions_token(self):
        from cognee.modules.visualization.layouts.semantic_layout import emit_js

        result = emit_js()
        assert "__SEMANTIC_POSITIONS__" in result

    def test_contains_apply_function(self):
        from cognee.modules.visualization.layouts.semantic_layout import emit_js

        result = emit_js()
        assert "_applySemanticLayout" in result
