from typing import List, Optional
from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage import index_graph_edges

logger = get_logger("chunk_associations")


class ChunkSimilarity(BaseModel):
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
    chunks: List[str],
    similarity_threshold: float = 0.7,
    min_chunk_length: int = 10,
    top_k_candidates: Optional[int] = None,
    user_prompt_location: str = "chunk_association_user.txt",
    system_prompt_location: str = "chunk_association_system.txt",
):
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

    chunk_id_map = {}
    for chunk_text in valid_chunks:
        try:
            results = await vector_engine.search("DocumentChunk_text", chunk_text, limit=1)
            if results:
                chunk_id_map[chunk_text] = results[0].id
        except Exception as e:
            logger.warning(f"Failed to find chunk ID for text: {chunk_text[:50]}... Error: {e}")

    logger.info(f"Found {len(chunk_id_map)} chunk IDs from vector search")

    edges = []
    compared_pairs = set()

    search_limit = (top_k_candidates + 1) if top_k_candidates else None

    for chunk_text in valid_chunks:
        if chunk_text not in chunk_id_map:
            continue

        try:
            candidates = await vector_engine.search(
                "DocumentChunk_text", chunk_text, limit=search_limit
            )
        except Exception as e:
            logger.warning(f"Vector search failed for chunk: {chunk_text[:50]}... Error: {e}")
            continue

        for candidate in candidates:
            try:
                candidate_text = candidate.payload.get("text", "")
            except (AttributeError, KeyError):
                continue

            if not candidate_text or candidate_text == chunk_text:
                continue

            if candidate_text not in chunk_id_map:
                continue

            pair_key = tuple(sorted([chunk_text, candidate_text]))
            if pair_key in compared_pairs:
                continue
            compared_pairs.add(pair_key)

            similarity = await _compare_chunks(
                chunk_text, candidate_text, user_prompt_location, system_prompt_location
            )

            if (
                similarity
                and similarity.are_similar
                and similarity.similarity_score >= similarity_threshold
            ):
                chunk_id = chunk_id_map[chunk_text]
                candidate_id = chunk_id_map[candidate_text]
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
