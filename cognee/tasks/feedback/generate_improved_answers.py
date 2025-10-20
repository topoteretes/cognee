from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.shared.logging_utils import get_logger

from .utils import create_retriever


class ImprovedAnswerResponse(BaseModel):
    """Response model for improved answer generation containing answer and explanation."""

    answer: str
    explanation: str


logger = get_logger("generate_improved_answers")


def _validate_input_data(feedback_interactions: List[Dict]) -> bool:
    """Validate that input contains required fields for all items."""
    required_fields = [
        "question",
        "answer",
        "context",
        "feedback_text",
        "feedback_id",
        "interaction_id",
    ]
    return all(
        all(item.get(field) is not None for field in required_fields)
        for item in feedback_interactions
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
    feedback_interaction: Dict, retriever, reaction_prompt_location: str
) -> Optional[Dict]:
    """Generate improved answer for a single feedback-interaction pair using structured retriever completion."""
    try:
        question_text = feedback_interaction["question"]
        original_answer_text = feedback_interaction["answer"]
        context_text = feedback_interaction["context"]
        feedback_text = feedback_interaction["feedback_text"]

        query_text = _render_reaction_prompt(
            question_text, context_text, original_answer_text, feedback_text
        )

        retrieved_context = await retriever.get_context(query_text)
        completion = await retriever.get_structured_completion(
            query=query_text, context=retrieved_context, response_model=ImprovedAnswerResponse
        )
        new_context_text = await retriever.resolve_edges_to_text(retrieved_context)

        if completion:
            return {
                **feedback_interaction,
                "improved_answer": completion.answer,
                "new_context": new_context_text,
                "explanation": completion.explanation,
            }
        else:
            logger.warning(
                "Failed to get structured completion from retriever", question=question_text
            )
            return None

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to generate improved answer",
            error=str(exc),
            question=feedback_interaction.get("question"),
        )
        return None


async def generate_improved_answers(
    feedback_interactions: List[Dict],
    retriever_name: str = "graph_completion_cot",
    top_k: int = 20,
    reaction_prompt_location: str = "feedback_reaction_prompt.txt",
) -> List[Dict]:
    """Generate improved answers using configurable retriever and LLM."""
    if not feedback_interactions:
        logger.info("No feedback interactions provided; returning empty list")
        return []

    if not _validate_input_data(feedback_interactions):
        logger.error("Input data validation failed; missing required fields")
        return []

    retriever = create_retriever(
        retriever_name=retriever_name,
        top_k=top_k,
        user_prompt_path="graph_context_for_question.txt",
        system_prompt_path="answer_simple_question.txt",
    )

    improved_answers: List[Dict] = []

    for feedback_interaction in feedback_interactions:
        result = await _generate_improved_answer_for_single_interaction(
            feedback_interaction, retriever, reaction_prompt_location
        )

        if result:
            improved_answers.append(result)
        else:
            logger.warning(
                "Failed to generate improved answer",
                question=feedback_interaction.get("question"),
                interaction_id=feedback_interaction.get("interaction_id"),
            )

    logger.info("Generated improved answers", count=len(improved_answers))
    return improved_answers
