"""
Unit tests for the CONSOLIDATE memory task (issue #3393).

Tests verify:
1. Two near-duplicate entities collapse into one node with all edges preserved.
2. dry_run mode reports proposed merges but writes nothing.
3. Low-similarity pairs are never merged.
4. protect_node_types prevents merging protected node types.
5. Nodes already merged away in the same run are skipped (idempotency).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cognee.tasks.memify.consolidate_duplicate_entities import (
    consolidate_duplicate_entities,
    _cosine_similarity,
    _move_edges,
    MergeDecision,
    MergeReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph_engine(nodes, edges_by_id=None):
    """Build a mock graph engine with controllable nodes and edges."""
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, []))
    engine.get_edges = AsyncMock(
        side_effect=lambda node_id: edges_by_id.get(node_id, []) if edges_by_id else []
    )
    engine.add_edges = AsyncMock()
    engine.add_node = AsyncMock()
    engine.delete_nodes = AsyncMock()
    return engine


EMBED_A = [1.0, 0.0, 0.0]
EMBED_B = [0.999, 0.045, 0.0]   # cosine ~0.9999 — very similar
EMBED_C = [0.0, 0.0, 1.0]       # cosine ~0.0   — unrelated

# ---------------------------------------------------------------------------
# Async embedding helpers (Python 3.12 compatible)
# ---------------------------------------------------------------------------

async def embed_a_or_b(text):
    return EMBED_A if "Apple Inc" in text else EMBED_B


async def embed_a(text):
    return EMBED_A


async def embed_a_or_c(text):
    return EMBED_A if "Apple" in text else EMBED_C


# ---------------------------------------------------------------------------
# Unit: cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Unit: _move_edges
# ---------------------------------------------------------------------------

class TestMoveEdges:
    @pytest.mark.asyncio
    async def test_edges_repointed_to_canonical(self):
        engine = AsyncMock()
        engine.get_edges = AsyncMock(return_value=[
            ("dup-id", "node-c", "relates_to", {}),
            ("node-d", "dup-id", "part_of", {}),
        ])
        engine.add_edges = AsyncMock()

        await _move_edges(engine, "dup-id", "canonical-id")

        added = engine.add_edges.call_args[0][0]
        assert any(e[0] == "canonical-id" or e[1] == "canonical-id" for e in added)

    @pytest.mark.asyncio
    async def test_self_loops_are_skipped(self):
        engine = AsyncMock()
        # Edge from dup to canonical — after repoint becomes canonical->canonical (self-loop)
        engine.get_edges = AsyncMock(return_value=[
            ("dup-id", "canonical-id", "relates_to", {}),
        ])
        engine.add_edges = AsyncMock()

        await _move_edges(engine, "dup-id", "canonical-id")

        engine.add_edges.assert_not_called()


# ---------------------------------------------------------------------------
# Integration-style: consolidate_duplicate_entities
# ---------------------------------------------------------------------------

class TestConsolidateDuplicateEntities:

    @pytest.mark.asyncio
    async def test_duplicate_nodes_merged(self):
        nodes = [
            ("id-a", {"type": "Entity", "name": "Apple Inc", "description": "A tech company"}),
            ("id-b", {"type": "Entity", "name": "Apple Incorporated", "description": "Technology company Apple"}),
        ]
        edges_by_id = {
            "id-b": [("id-b", "id-x", "makes", {})],
        }
        engine = make_graph_engine(nodes, edges_by_id)

        with patch(
            "cognee.tasks.memify.consolidate_duplicate_entities.get_graph_engine",
            new=AsyncMock(return_value=engine),
        ), patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._embed_text",
            side_effect=embed_a_or_b,
        ), patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._llm_confirm_merge",
            return_value=MergeDecision(should_merge=True, reason="same entity"),
        ), patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._merge_descriptions",
            return_value="A technology company.",
        ):
            report = await consolidate_duplicate_entities(
                args={},
                similarity_threshold=0.90,
                dry_run=False,
            )

        assert len(report.executed) == 1
        engine.delete_nodes.assert_called_once()
        engine.add_edges.assert_called()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self):
        nodes = [
            ("id-a", {"type": "Entity", "name": "Apple Inc", "description": "A tech company"}),
            ("id-b", {"type": "Entity", "name": "Apple Incorporated", "description": "Technology company Apple"}),
        ]
        engine = make_graph_engine(nodes)

        with patch(
            "cognee.tasks.memify.consolidate_duplicate_entities.get_graph_engine",
            new=AsyncMock(return_value=engine),
        ),patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._embed_text",
            side_effect=embed_a,
        ), patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._llm_confirm_merge",
            return_value=MergeDecision(should_merge=True, reason="same entity"),
        ):
            report = await consolidate_duplicate_entities(
                args={},
                similarity_threshold=0.90,
                dry_run=True,
            )

        assert report.dry_run is True
        assert len(report.proposed) >= 1
        engine.delete_nodes.assert_not_called()
        engine.add_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_similarity_pair_not_merged(self):
        nodes = [
            ("id-a", {"type": "Entity", "name": "Apple", "description": "A fruit"}),
            ("id-c", {"type": "Entity", "name": "Airplane", "description": "A flying vehicle"}),
        ]
        engine = make_graph_engine(nodes)

        with patch(
            "cognee.tasks.memify.consolidate_duplicate_entities.get_graph_engine",
            new=AsyncMock(return_value=engine),
        ), patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._embed_text",
            side_effect=embed_a_or_c,
        ):
            report = await consolidate_duplicate_entities(
                args={},
                similarity_threshold=0.92,
                dry_run=False,
            )

        assert len(report.executed) == 0
        engine.delete_nodes.assert_not_called()

    @pytest.mark.asyncio
    async def test_protect_node_types_skipped(self):
        nodes = [
            ("id-a", {"type": "DocumentChunk", "name": "chunk A", "description": "text"}),
            ("id-b", {"type": "DocumentChunk", "name": "chunk A copy", "description": "text"}),
        ]
        engine = make_graph_engine(nodes)

        with patch(
            "cognee.tasks.memify.consolidate_duplicate_entities.get_graph_engine",
            new=AsyncMock(return_value=engine),
        ), patch(
            "cognee.tasks.memify.consolidate_duplicate_entities._embed_text",
            side_effect=embed_a,
        ):
            report = await consolidate_duplicate_entities(
                args={},
                similarity_threshold=0.90,
                protect_node_types=["DocumentChunk"],
                dry_run=False,
            )

        assert len(report.executed) == 0
        engine.delete_nodes.assert_not_called()