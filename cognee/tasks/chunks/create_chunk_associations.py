"""
Chunk association task for creating semantic links between document chunks.

This module uses vector similarity search to identify candidate chunk pairs,
then applies LLM-based comparison to determine whether chunks should be
linked with weighted "associated_with" edges in the knowledge graph.
"""

from typing import AsyncGenerator, List, Optional, Union
from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage import index_graph_edges

logger = get_logger("chunk_associations")


class ChunkSimilarity(BaseModel):
    """LLM-structured output representing the semantic similarity between two chunks."""

    are_similar: bool = Field(description="Whether chunks are semantically related")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Similarity score 0.0-1.0")
    reasoning: str = Field(description="Brief explanation of similarity assessment")
    association_type: Optional[str] = Field(
        default=None, description="Type: topical, causal, temporal, elaboration, contextual"
    )


async def _compare_chunks(
    chunk_1: str,
    chunk_2: str,
    user_prompt_location: str,
    system_prompt_location: str,
) -> Optional[ChunkSimilarity]:
    """Compare two text chunks for semantic similarity using an LLM.

    Renders the user and system prompts with the chunk texts and calls the LLM
    to produce a structured ChunkSimilarity response. Returns a fallback
    ChunkSimilarity with are_similar=False on LLM failure.

    Args:
        chunk_1: Text content of the first chunk.
        chunk_2: Text content of the second chunk.
        user_prompt_location: Filename of the user prompt template.
        system_prompt_location: Filename of the system prompt template.

    Returns:
        A ChunkSimilarity object, or a fallback with score 0.0 on error.
    """
    context = {"chunk_1": chunk_1, "chunk_2": chunk_2}
    user_prompt = render_prompt(user_prompt_location, context)
    system_prompt = read_query_prompt(system_prompt_location)

    try:
        return await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=ChunkSimilarity,
        )
    except Exception as e:
        logger.warning(f"LLM comparison failed: {e}")
        return ChunkSimilarity(
            are_similar=False, similarity_score=0.0, reasoning="LLM error", association_type=None
        )


def _create_edge(chunk_1_id: str, chunk_2_id: str, similarity: ChunkSimilarity):
    """Build a graph edge tuple from two chunk IDs and their similarity result.

    Args:
        chunk_1_id: UUID of the source chunk node.
        chunk_2_id: UUID of the target chunk node.
        similarity: The LLM-produced similarity assessment.

    Returns:
        A tuple of (source_id, target_id, relationship_name, properties_dict).
    """
    return (
        chunk_1_id,
        chunk_2_id,
        "associated_with",
        {
            "relationship_name": "associated_with",
            "source_node_id": chunk_1_id,
            "target_node_id": chunk_2_id,
            "weight": similarity.similarity_score,
            "association_type": similarity.association_type,
            "reasoning": similarity.reasoning,
            "ontology_valid": False,
        },
    )


async def create_chunk_associations(
    chunks: Union[List[str], str],
    similarity_threshold: float = 0.7,
    min_chunk_length: int = 10,
    top_k_candidates: Optional[int] = None,
    user_prompt_location: str = "chunk_association_user.txt",
    system_prompt_location: str = "chunk_association_system.txt",
) -> AsyncGenerator[str, None]:
    """Create semantic association edges between document chunks in the knowledge graph.

    For each valid chunk, performs a vector similarity search to find candidate
    pairs, then uses an LLM to assess semantic relatedness. Pairs that meet the
    similarity threshold are persisted as weighted "associated_with" edges.

    This is an async generator that yields the original chunks after processing,
    allowing it to be composed in cognee pipelines.

    Args:
        chunks: List of chunk text strings, or a single string, to evaluate.
        similarity_threshold: Minimum LLM similarity score (0.0-1.0) required
            to create an association edge. (default 0.7)
        min_chunk_length: Minimum character length for a chunk to be considered.
            Chunks shorter than this are skipped. (default 10)
        top_k_candidates: Maximum number of vector search candidates per chunk.
            None means no limit. (default None)
        user_prompt_location: Filename of the user prompt template. (default
            "chunk_association_user.txt")
        system_prompt_location: Filename of the system prompt template. (default
            "chunk_association_system.txt")

    Yields:
        Each chunk from the input list, unchanged.

    Raises:
        Exception: Re-raised if persisting edges to the graph database fails.
    """
    if not isinstance(chunks, list):
        chunks = [chunks]

    valid_chunks = [
        chunk
        for chunk in chunks
        if chunk and isinstance(chunk, str) and len(chunk) >= min_chunk_length
    ]

    if len(valid_chunks) < 2:
        logger.info(f"Less than 2 valid chunks ({len(valid_chunks)}), skipping associations")
        for chunk in chunks:
            yield chunk
        return

    logger.info(
        f"Creating associations for {len(valid_chunks)} chunks with threshold {similarity_threshold}"
    )

    vector_engine = get_vector_engine()

    id_to_text = {}
    for chunk_text in valid_chunks:
        try:
            results = await vector_engine.search("DocumentChunk_text", chunk_text, limit=1)
            if results:
                chunk_id = str(results[0].id)
                id_to_text[chunk_id] = chunk_text
        except Exception as e:
            logger.warning(f"Failed to find chunk ID for text: {chunk_text[:50]}... Error: {e}")

    logger.info(f"Found {len(id_to_text)} chunk IDs from vector search")

    edges = []
    compared_pairs = set()

    search_limit = (top_k_candidates + 1) if top_k_candidates is not None else None

    for chunk_id, chunk_text in id_to_text.items():
        try:
            candidates = await vector_engine.search(
                "DocumentChunk_text", chunk_text, limit=search_limit
            )
        except Exception as e:
            logger.warning(f"Vector search failed for chunk: {chunk_text[:50]}... Error: {e}")
            continue

        for candidate in candidates:
            candidate_id = str(candidate.id)

            if candidate_id == chunk_id or candidate_id not in id_to_text:
                continue

            pair_key = tuple(sorted([chunk_id, candidate_id]))
            if pair_key in compared_pairs:
                continue
            compared_pairs.add(pair_key)

            candidate_text = id_to_text[candidate_id]

            similarity = await _compare_chunks(
                chunk_text, candidate_text, user_prompt_location, system_prompt_location
            )

            if (
                similarity
                and similarity.are_similar
                and similarity.similarity_score >= similarity_threshold
            ):
                edges.append(_create_edge(chunk_id, candidate_id, similarity))
                logger.info(
                    f"Association created: score={similarity.similarity_score:.2f}, type={similarity.association_type}"
                )

    logger.info(f"Created {len(edges)} association edges")

    if edges:
        try:
            graph_engine = await get_graph_engine()
            await graph_engine.add_edges(edges)
            await index_graph_edges(edges)
            logger.info(f"Successfully persisted {len(edges)} edges to graph database")
        except Exception as e:
            logger.error(f"Failed to persist edges to graph database: {e}")
            raise

    for chunk in chunks:
        yield chunk
