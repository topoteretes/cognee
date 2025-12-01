from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.shared.logging_utils import get_logger

from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from .models import FeedbackEnrichment


class ImprovedAnswerResponse(BaseModel):
    """Response model for improved answer generation containing answer and explanation."""

    answer: str
    explanation: str


logger = get_logger("generate_improved_answers")


def _validate_input_data(enrichments: List[FeedbackEnrichment]) -> bool:
    """Validate that input contains required fields for all enrichments."""
    return all(
        enrichment.question is not None
        and enrichment.original_answer is not None
        and enrichment.context is not None
        and enrichment.feedback_text is not None
        and enrichment.feedback_id is not None
        and enrichment.interaction_id is not None
        for enrichment in enrichments
    )


def _render_reaction_prompt(
    question: str, context: str, wrong_answer: str, negative_feedback: str
) -> str:
    """Render the feedback reaction prompt with provided variables."""
    prompt_template = read_query_prompt("feedback_reaction_prompt.txt")
    return prompt_template.format(
        question=question,
        context=context,
        wrong_answer=wrong_answer,
        negative_feedback=negative_feedback,
    )


async def _generate_improved_answer_for_single_interaction(
    enrichment: FeedbackEnrichment, retriever, reaction_prompt_location: str
) -> Optional[FeedbackEnrichment]:
    """Generate improved answer for a single enrichment using structured retriever completion."""
    try:
        query_text = _render_reaction_prompt(
            enrichment.question,
            enrichment.context,
            enrichment.original_answer,
            enrichment.feedback_text,
        )

        retrieved_context = await retriever.get_context(query_text)
        completion = await retriever.get_structured_completion(
            query=query_text,
            context=retrieved_context,
            response_model=ImprovedAnswerResponse,
            max_iter=4,
        )
        new_context_text = await retriever.resolve_edges_to_text(retrieved_context)

        if completion:
            enrichment.improved_answer = completion.answer
            enrichment.new_context = new_context_text
            enrichment.explanation = completion.explanation
            return enrichment
        else:
            logger.warning(
                "Failed to get structured completion from retriever", question=enrichment.question
            )
            return None

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to generate improved answer",
            error=str(exc),
            question=enrichment.question,
        )
        return None


async def generate_improved_answers(
    enrichments: List[FeedbackEnrichment],
    top_k: int = 20,
    reaction_prompt_location: str = "feedback_reaction_prompt.txt",
) -> List[FeedbackEnrichment]:
    """Generate improved answers using CoT retriever and LLM."""
    if not enrichments:
        logger.info("No enrichments provided; returning empty list")
        return []

    if not _validate_input_data(enrichments):
        logger.error("Input data validation failed; missing required fields")
        return []

    retriever = GraphCompletionCotRetriever(
        top_k=top_k,
        save_interaction=False,
        user_prompt_path="graph_context_for_question.txt",
        system_prompt_path="answer_simple_question.txt",
    )

    improved_answers: List[FeedbackEnrichment] = []

    for enrichment in enrichments:
        result = await _generate_improved_answer_for_single_interaction(
            enrichment, retriever, reaction_prompt_location
        )

        if result:
            improved_answers.append(result)
        else:
            logger.warning(
                "Failed to generate improved answer",
                question=enrichment.question,
                interaction_id=enrichment.interaction_id,
            )

    logger.info("Generated improved answers", count=len(improved_answers))
    return improved_answers
