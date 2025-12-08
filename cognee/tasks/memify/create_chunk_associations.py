import asyncio

from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage import index_graph_edges

logger = get_logger("create_chunk_associations")


class AssociationDecision(BaseModel):
    """LLM response model for association decision"""

    should_associate: bool


async def ask_llm_for_association(
    chunk1_text: str, chunk2_text: str, similarity_score: float
) -> bool:
    """
    Use LLM to decide if two chunks should be associated.

    Args:
        chunk1_text: Text of first chunk
        chunk2_text: Text of second chunk
        similarity_score: Computed similarity score (0.0 to 1.0)

    Returns:
        bool: True if chunks should be associated, False otherwise
    """
    user_context = {
        "chunk1_text": chunk1_text,
        "chunk2_text": chunk2_text,
        "similarity_score": similarity_score,
    }

    user_prompt = render_prompt("chunk_association_classifier_user.txt", context=user_context)
    system_prompt = render_prompt("chunk_association_classifier_system.txt", context={})

    decision = await LLMGateway.acreate_structured_output(
        text_input=user_prompt, system_prompt=system_prompt, response_model=AssociationDecision
    )

    return decision.should_associate


async def create_chunk_associations(data, similarity_threshold: float = 0.90, max_candidates_per_chunk: int = None):
    """
    Analyzes chunks for semantic similarity and creates weighted association edges.

    Args:
        data: Either a single chunk text (str) or list of chunk texts from previous task
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider for association
        max_candidates_per_chunk: Maximum candidates per chunk. None = no limit (processes all above threshold)
    """
    chunks = data if isinstance(data, list) else [data]
    if not chunks:
        return data

    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    # Step 1: Find all similar chunk pairs above threshold
    seen_pairs = set()
    candidates = []

    for chunk_text in chunks:
        # Get ALL chunks if no limit, otherwise use default vector search limit
        search_params = {"collection_name": "DocumentChunk_text", "query_text": chunk_text}
        if max_candidates_per_chunk is None:
            search_params["limit"] = None  # Get all chunks

        similar_chunks = await vector_engine.search(**search_params)
        if not similar_chunks:
            continue

        origin_id = similar_chunks[0].id
        valid_pairs = []

        for similar in similar_chunks[1:]:  # Skip self
            pair_key = tuple(sorted([str(origin_id), str(similar.id)]))

            if pair_key in seen_pairs or similar.score < similarity_threshold:
                continue

            valid_pairs.append({
                "origin_id": origin_id,
                "similar_id": similar.id,
                "text1": chunk_text,
                "text2": similar.payload.get("text", ""),
                "score": similar.score,
            })

        # Sort by similarity and apply limit if specified
        valid_pairs.sort(key=lambda x: x["score"], reverse=True)
        pairs_to_process = valid_pairs if max_candidates_per_chunk is None else valid_pairs[:max_candidates_per_chunk]

        for pair in pairs_to_process:
            pair_key = tuple(sorted([str(pair["origin_id"]), str(pair["similar_id"])]))
            seen_pairs.add(pair_key)
            candidates.append(pair)

    if not candidates:
        logger.info("No candidate pairs found")
        return data

    # Step 2: Ask LLM for all pairs in parallel
    logger.info(f"Evaluating {len(candidates)} pairs with LLM in parallel...")
    llm_tasks = [
        ask_llm_for_association(pair["text1"], pair["text2"], pair["score"])
        for pair in candidates
    ]
    decisions = await asyncio.gather(*llm_tasks)

    # Step 3: Create edges for approved pairs
    edges = [
        (
            pair["origin_id"],
            pair["similar_id"],
            "associated_with",
            {
                "relationship_name": "associated_with",
                "source_node_id": pair["origin_id"],
                "target_node_id": pair["similar_id"],
                "weight": pair["score"],
                "ontology_valid": False,
            },
        )
        for pair, approved in zip(candidates, decisions)
        if approved
    ]

    if edges:
        await graph_engine.add_edges(edges)
        await index_graph_edges(edges)
        logger.info(f"Created {len(edges)} chunk associations")
    else:
        logger.info("No chunk associations created")

    return data
