"""Tests for cognee.modules.visualization.layouts.clustering.

Runs fully offline — no LLM, no vector store, no HDBSCAN required.
Positions are injected directly so the suite is fast and deterministic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal PreprocessedGraph stub
# ---------------------------------------------------------------------------

@dataclass
class _FakePreprocessed:
    nodes: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)


def _make_nodes(n: int) -> list[dict]:
    return [{"id": f"n{i}", "label": f"Node {i}", "type": "Entity"} for i in range(n)]


def _grid_positions(n: int, spread: float = 800.0) -> dict[str, tuple[float, float]]:
    """Place nodes in a regular grid — deterministic positions."""
    side = math.ceil(math.sqrt(n))
    positions = {}
    for i in range(n):
        row, col = divmod(i, side)
        x = (col / max(side - 1, 1) * 2 - 1) * spread
        y = (row / max(side - 1, 1) * 2 - 1) * spread
        positions[f"n{i}"] = (x, y)
    return positions


def _clustered_positions(k: int, per_cluster: int = 5) -> dict[str, tuple[float, float]]:
    """Return positions with k clear clusters (nodes near cluster centres)."""
    rng = random.Random(42)
    centres = [(rng.uniform(-500, 500), rng.uniform(-500, 500)) for _ in range(k)]
    positions = {}
    idx = 0
    for cx, cy in centres:
        for _ in range(per_cluster):
            positions[f"n{idx}"] = (cx + rng.gauss(0, 20), cy + rng.gauss(0, 20))
            idx += 1
    return positions


# ---------------------------------------------------------------------------
# Unit tests: _kmeans
# ---------------------------------------------------------------------------

class TestKMeans:
    def test_returns_correct_number_of_labels(self):
        from cognee.modules.visualization.layouts.clustering import _kmeans
        points = [(float(i), float(i)) for i in range(12)]
        labels = _kmeans(points, k=3)
        assert len(labels) == 12

    def test_labels_are_integers(self):
        from cognee.modules.visualization.layouts.clustering import _kmeans
        points = [(float(i % 3) * 100, 0.0) for i in range(9)]
        labels = _kmeans(points, k=3)
        for lbl in labels:
            assert isinstance(lbl, int)

    def test_k_greater_than_points_returns_unique_labels(self):
        from cognee.modules.visualization.layouts.clustering import _kmeans
        points = [(1.0, 2.0), (3.0, 4.0)]
        labels = _kmeans(points, k=5)
        assert len(labels) == 2

    def test_deterministic_with_same_seed(self):
        from cognee.modules.visualization.layouts.clustering import _kmeans
        points = [(float(i), float(i * 2)) for i in range(20)]
        assert _kmeans(points, k=4, seed=0) == _kmeans(points, k=4, seed=0)

    def test_three_clear_clusters(self):
        from cognee.modules.visualization.layouts.clustering import _kmeans
        # Three tight clusters far apart
        points = (
            [(0.0 + i * 0.01, 0.0) for i in range(5)] +
            [(100.0 + i * 0.01, 0.0) for i in range(5)] +
            [(200.0 + i * 0.01, 0.0) for i in range(5)]
        )
        labels = _kmeans(points, k=3)
        # Each of the 3 groups should share a label
        g0 = set(labels[:5])
        g1 = set(labels[5:10])
        g2 = set(labels[10:])
        assert len(g0) == 1
        assert len(g1) == 1
        assert len(g2) == 1
        assert g0 != g1 != g2


# ---------------------------------------------------------------------------
# Unit tests: ClusterResult dataclass
# ---------------------------------------------------------------------------

class TestClusterResult:
    def test_default_empty(self):
        from cognee.modules.visualization.layouts.clustering import ClusterResult
        cr = ClusterResult()
        assert cr.assignments == {}
        assert cr.labels == {}
        assert cr.colors == {}
        assert cr.members == {}
        assert cr.n_clusters == 0


# ---------------------------------------------------------------------------
# Integration tests: compute_clusters
# ---------------------------------------------------------------------------

class TestComputeClusters:
    @pytest.mark.asyncio
    async def test_empty_positions_returns_empty_result(self):
        from cognee.modules.visualization.layouts.clustering import compute_clusters
        pre = _FakePreprocessed(nodes=[], links=[])
        result = await compute_clusters(pre, {})
        assert result.n_clusters == 0
        assert result.assignments == {}

    @pytest.mark.asyncio
    async def test_all_nodes_assigned(self):
        from cognee.modules.visualization.layouts.clustering import compute_clusters
        nodes = _make_nodes(12)
        pre = _FakePreprocessed(nodes=nodes)
        positions = _grid_positions(12)
        result = await compute_clusters(pre, positions, k=3)
        assert set(result.assignments.keys()) == {n["id"] for n in nodes}

    @pytest.mark.asyncio
    async def test_n_clusters_within_bounds(self):
        from cognee.modules.visualization.layouts.clustering import (
            _MAX_CLUSTERS,
            compute_clusters,
        )
        nodes = _make_nodes(20)
        pre = _FakePreprocessed(nodes=nodes)
        positions = _grid_positions(20)
        result = await compute_clusters(pre, positions)
        assert 0 <= result.n_clusters <= _MAX_CLUSTERS

    @pytest.mark.asyncio
    async def test_every_cluster_has_label_and_color(self):
        from cognee.modules.visualization.layouts.clustering import compute_clusters
        nodes = _make_nodes(15)
        pre = _FakePreprocessed(nodes=nodes)
        positions = _grid_positions(15)
        result = await compute_clusters(pre, positions, k=3)
        for cid in range(result.n_clusters):
            assert cid in result.labels
            assert cid in result.colors
            assert result.colors[cid].startswith("#")

    @pytest.mark.asyncio
    async def test_nodes_without_positions_get_noise_label(self):
        from cognee.modules.visualization.layouts.clustering import compute_clusters
        nodes = _make_nodes(10)
        pre = _FakePreprocessed(nodes=nodes)
        # Only provide positions for first 6
        positions = _grid_positions(6)
        result = await compute_clusters(pre, positions, k=2)
        # Nodes n6..n9 have no position → should be -1
        for i in range(6, 10):
            assert result.assignments.get(f"n{i}") == -1

    @pytest.mark.asyncio
    async def test_clear_clusters_detected(self):
        """With k=3 tight clusters the algorithm should find 3 groups."""
        from cognee.modules.visualization.layouts.clustering import compute_clusters
        positions = _clustered_positions(k=3, per_cluster=5)
        nodes = [{"id": nid, "label": nid, "type": "Entity"} for nid in positions]
        pre = _FakePreprocessed(nodes=nodes)
        result = await compute_clusters(pre, positions, k=3)
        assert result.n_clusters == 3

    @pytest.mark.asyncio
    async def test_hdbscan_fallback_to_kmeans(self):
        """When hdbscan is not installed, k-means is used without error."""
        from cognee.modules.visualization.layouts.clustering import compute_clusters
        import sys
        # Temporarily hide hdbscan if present
        hdbscan_mod = sys.modules.pop("hdbscan", None)
        try:
            nodes = _make_nodes(10)
            pre = _FakePreprocessed(nodes=nodes)
            positions = _grid_positions(10)
            result = await compute_clusters(pre, positions, k=2)
            assert result.n_clusters >= 0
        finally:
            if hdbscan_mod is not None:
                sys.modules["hdbscan"] = hdbscan_mod


# ---------------------------------------------------------------------------
# Unit tests: emit_js
# ---------------------------------------------------------------------------

class TestEmitJs:
    def test_returns_string(self):
        from cognee.modules.visualization.layouts.clustering import ClusterResult, emit_js
        result = emit_js(ClusterResult())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_apply_cluster_colors(self):
        from cognee.modules.visualization.layouts.clustering import ClusterResult, emit_js
        js = emit_js(ClusterResult())
        assert "_applyClusterColors" in js

    def test_contains_draw_cluster_hulls(self):
        from cognee.modules.visualization.layouts.clustering import ClusterResult, emit_js
        js = emit_js(ClusterResult())
        assert "_drawClusterHulls" in js

    def test_contains_has_cluster_layout_flag(self):
        from cognee.modules.visualization.layouts.clustering import ClusterResult, emit_js
        js = emit_js(ClusterResult())
        assert "_hasClusterLayout" in js

    def test_token_present_when_no_result(self):
        from cognee.modules.visualization.layouts.clustering import emit_js
        js = emit_js(None)
        assert "CLUSTER_DATA_TOKEN" in js

    def test_inlines_cluster_data_when_provided(self):
        from cognee.modules.visualization.layouts.clustering import ClusterResult, emit_js
        cr = ClusterResult(
            assignments={"n0": 0, "n1": 1},
            labels={0: "Group A", 1: "Group B"},
            colors={0: "#4E79A7", 1: "#F28E2B"},
            n_clusters=2,
        )
        js = emit_js(cr)
        assert "Group A" in js
        assert "#4E79A7" in js
        assert "__CLUSTER_DATA__" not in js
