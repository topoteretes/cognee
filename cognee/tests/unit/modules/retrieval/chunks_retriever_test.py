import os
import pytest
import pathlib
from typing import List
import cognee
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity


class DocumentChunkWithEntities(DataPoint):
    text: str
    chunk_size: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    contains: List[Entity] = None

    metadata: dict = {"index_fields": ["text"]}


class TestChunksRetriever:
    @pytest.mark.asyncio
    async def test_chunk_context_simple(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_chunk_context_simple"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_chunk_context_simple"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document = TextDocument(
            name="Steve Rodger's career",
            raw_data_location="somewhere",
            external_metadata="",
            mime_type="text/plain",
            importance_weight = 0.5
        )

        chunk1 = DocumentChunk(
            text="Steve Rodger",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=0.5
        )
        chunk2 = DocumentChunk(
            text="Mike Broski",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=1
        )
        chunk3 = DocumentChunk(
            text="Christina Mayer",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=0.5
        )

        entities = [chunk1, chunk2, chunk3]

        await add_data_points(entities)

        retriever = ChunksRetriever()

        context = await retriever.get_context("Mike")

        assert context[0]["text"] == "Mike Broski", "Failed to get Mike Broski"

    @pytest.mark.asyncio
    async def test_chunk_context_complex(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_chunk_context_complex"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_chunk_context_complex"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document1 = TextDocument(
            name="Employee List",
            raw_data_location="somewhere",
            external_metadata="",
            mime_type="text/plain",
        )

        document2 = TextDocument(
            name="Car List",
            raw_data_location="somewhere",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="Steve Rodger",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )
        chunk2 = DocumentChunk(
            text="Mike Broski",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )
        chunk3 = DocumentChunk(
            text="Christina Mayer",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document1,
            contains=[],
        )

        chunk4 = DocumentChunk(
            text="Range Rover",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk5 = DocumentChunk(
            text="Hyundai",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )
        chunk6 = DocumentChunk(
            text="Chrysler",
            chunk_size=2,
            chunk_index=2,
            cut_type="sentence_end",
            is_part_of=document2,
            contains=[],
        )

        entities = [chunk1, chunk2, chunk3, chunk4, chunk5, chunk6]

        await add_data_points(entities)

        retriever = ChunksRetriever(top_k=20)

        context = await retriever.get_context("Christina")

        assert context[0]["text"] == "Christina Mayer", "Failed to get Christina Mayer"

    @pytest.mark.asyncio
    async def test_chunk_context_on_empty_graph(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_chunk_context_on_empty_graph"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_chunk_context_on_empty_graph"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        retriever = ChunksRetriever()

        with pytest.raises(NoDataError):
            await retriever.get_context("Christina Mayer")

        vector_engine = get_vector_engine()
        await vector_engine.create_collection(
            "DocumentChunk_text", payload_schema=DocumentChunkWithEntities
        )

        context = await retriever.get_context("Christina Mayer")
        assert len(context) == 0, "Found chunks when none should exist"

    @pytest.mark.asyncio
    async def test_importance_weight_default_value(self):

        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_importance_weight_default"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_importance_weight_default"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document = TextDocument(
            name="Test Document",
            raw_data_location="test",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="Test chunk 1 (Missing Weight)",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
        )

        chunk2 = DocumentChunk(
            text="Test chunk 2 (Explicit Low Weight)",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=0.1
        )

        entities = [chunk1, chunk2]
        await add_data_points(entities)

        retriever = ChunksRetriever()

        with patch.object(retriever.vector_engine, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                type('ScoredPoint', (), {'payload': chunk1.model_dump(), 'score': 0.9}),
                type('ScoredPoint', (), {'payload': chunk2.model_dump(), 'score': 0.6})
            ]

            context = await retriever.get_context("test query")

            args, kwargs = mock_search.call_args
            assert 'score_threshold' not in kwargs
            assert len(context) == 2
            assert context[0]["text"] == "Test chunk 1 (Missing Weight)"
            assert context[1]["text"] == "Test chunk 2 (Explicit Low Weight)"

    @pytest.mark.asyncio
    async def test_importance_weight_ranking(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_importance_weight_ranking"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_importance_weight_ranking"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document = TextDocument(
            name="Test Document",
            raw_data_location="test",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="High importance, low score",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=1.0
        )

        chunk2 = DocumentChunk(
            text="Low importance, high score",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=0.1
        )

        entities = [chunk1, chunk2]
        await add_data_points(entities)

        retriever = ChunksRetriever()

        with patch.object(retriever.vector_engine, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                type('ScoredPoint', (), {'payload': chunk1.model_dump(), 'score': 0.6}),
                type('ScoredPoint', (), {'payload': chunk2.model_dump(), 'score': 0.9})
            ]

            context = await retriever.get_context("test query")

            assert len(context) == 2
            assert context[0]["text"] == "High importance, low score"
            assert context[1]["text"] == "Low importance, high score"

    @pytest.mark.asyncio
    async def test_importance_weight_boundary_values(self):

        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_importance_weight_boundary_values"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_importance_weight_boundary_values"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document = TextDocument(
            name="Test Document",
            raw_data_location="test",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="Zero weight chunk",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=0.0
        )

        chunk2 = DocumentChunk(
            text="Full weight chunk",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=1.0
        )

        entities = [chunk1, chunk2]
        await add_data_points(entities)

        retriever = ChunksRetriever()
        with patch.object(retriever.vector_engine, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                type('ScoredPoint', (), {'payload': chunk1.model_dump(), 'score': 0.8}),  # 原始得分高 (0.8)
                type('ScoredPoint', (), {'payload': chunk2.model_dump(), 'score': 0.5})  # 原始得分低 (0.5)
            ]

            context = await retriever.get_context("test query")
            assert len(context) == 2
            assert context[0]["text"] == "Full weight chunk"
            assert context[1]["text"] == "Zero weight chunk"

    @pytest.mark.asyncio
    async def test_ranking_stability_on_equal_score(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_ranking_stability"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_ranking_stability"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        document = TextDocument(
            name="Test Document",
            raw_data_location="test",
            external_metadata="",
            mime_type="text/plain",
        )

        chunk1 = DocumentChunk(
            text="Stable Chunk 1 (High Weight, Low Score)",
            chunk_size=2,
            chunk_index=0,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=1.0
        )

        chunk2 = DocumentChunk(
            text="Stable Chunk 2 (Low Weight, High Score)",
            chunk_size=2,
            chunk_index=1,
            cut_type="sentence_end",
            is_part_of=document,
            contains=[],
            importance_weight=0.5
        )

        entities = [chunk1, chunk2]
        await add_data_points(entities)

        retriever = ChunksRetriever()

        with patch.object(retriever.vector_engine, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                type('ScoredPoint', (), {'payload': chunk2.model_dump(), 'score': 1.0}),
                type('ScoredPoint', (), {'payload': chunk1.model_dump(), 'score': 3.0})
            ]

            context = await retriever.get_context("test query for stability")

            assert len(context) == 2
            assert context[0]["text"] == "Stable Chunk 2 (Low Weight, High Score)"
            assert context[1]["text"] == "Stable Chunk 1 (High Weight, Low Score)"
