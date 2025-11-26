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


async def create_chunk_associations(data, similarity_threshold: float = 0.7):
    """
    Analyzes chunks for semantic similarity and creates weighted association edges.

    Args:
        data: Either a single chunk text (str) or list of chunk texts from previous task
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider for association
    """
    logger.info(f"create_chunk_associations called with data type: {type(data)}, data: {data}")

    # Handle data being passed from extract_subgraph_chunks
    chunks = data if isinstance(data, list) else [data]

    if not chunks:
        logger.info("No chunks provided for association")
        return

    logger.info(f"Processing chunks: {chunks}")

    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    edges_to_save = []
    processed_pairs = set()

    logger.info(f"Processing {len(chunks)} chunks for associations")

    for chunk_text in chunks:
        # Get this chunk's ID and find similar chunks in one search
        similar_chunks = await vector_engine.search(
            collection_name="DocumentChunk_text", query_text=chunk_text, limit=5
        )

        if not similar_chunks:
            continue

        origin_chunk_id = similar_chunks[0].id  # First result is the chunk itself

        for similar_chunk in similar_chunks[1:]:  # Skip first (self)
            # Skip duplicates
            pair_key = tuple(sorted([str(origin_chunk_id), str(similar_chunk.id)]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # Filter by threshold
            if similar_chunk.score < similarity_threshold:
                logger.debug(
                    f"Skipping pair - score {similar_chunk.score} below threshold {similarity_threshold}"
                )
                continue

            logger.info(
                f"Asking LLM for chunks with similarity {similar_chunk.score}: "
                f"'{chunk_text[:50]}...' <-> '{similar_chunk.payload.get('text', '')[:50]}...'"
            )

            # Ask LLM
            llm_decision = await ask_llm_for_association(
                chunk_text, similar_chunk.payload.get("text", ""), similar_chunk.score
            )
            logger.info(f"LLM decision: {llm_decision}")

            if llm_decision:
                edges_to_save.append(
                    (
                        origin_chunk_id,
                        similar_chunk.id,
                        "associated_with",
                        {
                            "relationship_name": "associated_with",
                            "source_node_id": origin_chunk_id,
                            "target_node_id": similar_chunk.id,
                            "weight": similar_chunk.score,
                            "ontology_valid": False,
                        },
                    )
                )

    # Store edges
    if edges_to_save:
        await graph_engine.add_edges(edges_to_save)
        await index_graph_edges(edges_to_save)
        logger.info(f"Created {len(edges_to_save)} chunk associations")
    else:
        logger.info("No chunk associations created")

    # Return the data so it can flow to next task in pipeline
    return data
