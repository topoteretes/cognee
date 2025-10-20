from __future__ import annotations

from typing import Dict, List, Optional
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt
from cognee.shared.logging_utils import get_logger
from cognee.modules.engine.models import NodeSet

from .models import FeedbackEnrichment


logger = get_logger("create_enrichments")


def _validate_improved_answers(improved_answers: List[Dict]) -> bool:
    """Validate that all items contain required fields for enrichment creation."""
    required_fields = [
        "question",
        "answer",  # This is the original answer field from feedback_interaction
        "improved_answer",
        "new_context",
        "feedback_id",
        "interaction_id",
    ]
    return all(
        all(item.get(field) is not None for field in required_fields) for item in improved_answers
    )


def _validate_uuid_fields(improved_answers: List[Dict]) -> bool:
    """Validate that feedback_id and interaction_id are valid UUID objects."""
    try:
        for item in improved_answers:
            feedback_id = item.get("feedback_id")
            interaction_id = item.get("interaction_id")
            if not isinstance(feedback_id, type(feedback_id)) or not isinstance(
                interaction_id, type(interaction_id)
            ):
                return False
        return True
    except Exception:
        return False


async def _generate_enrichment_report(
    question: str, improved_answer: str, new_context: str, report_prompt_location: str
) -> str:
    """Generate educational report using feedback report prompt."""
    try:
        prompt_template = read_query_prompt(report_prompt_location)
        rendered_prompt = prompt_template.format(
            question=question,
            improved_answer=improved_answer,
            new_context=new_context,
        )
        return await LLMGateway.acreate_structured_output(
            text_input=rendered_prompt,
            system_prompt="You are a helpful assistant that creates educational content.",
            response_model=str,
        )
    except Exception as exc:
        logger.warning("Failed to generate enrichment report", error=str(exc), question=question)
        return f"Educational content for: {question} - {improved_answer}"


async def _create_enrichment_datapoint(
    improved_answer_item: Dict,
    report_text: str,
) -> Optional[FeedbackEnrichment]:
    """Create a single FeedbackEnrichment DataPoint with proper ID and nodeset assignment."""
    try:
        question = improved_answer_item["question"]
        improved_answer = improved_answer_item["improved_answer"]

        # Create nodeset following UserQAFeedback pattern
        nodeset = NodeSet(
            id=uuid5(NAMESPACE_OID, name="FeedbackEnrichment"), name="FeedbackEnrichment"
        )

        enrichment = FeedbackEnrichment(
            id=str(uuid5(NAMESPACE_OID, f"{question}_{improved_answer}")),
            text=report_text,
            question=question,
            original_answer=improved_answer_item["answer"],  # Use "answer" field
            improved_answer=improved_answer,
            feedback_id=improved_answer_item["feedback_id"],
            interaction_id=improved_answer_item["interaction_id"],
            belongs_to_set=nodeset,
        )

        return enrichment
    except Exception as exc:
        logger.error(
            "Failed to create enrichment datapoint",
            error=str(exc),
            question=improved_answer_item.get("question"),
        )
        return None


async def create_enrichments(
    improved_answers: List[Dict],
    report_prompt_location: str = "feedback_report_prompt.txt",
) -> List[FeedbackEnrichment]:
    """Create FeedbackEnrichment DataPoint instances from improved answers."""
    if not improved_answers:
        logger.info("No improved answers provided; returning empty list")
        return []

    if not _validate_improved_answers(improved_answers):
        logger.error("Input validation failed; missing required fields")
        return []

    if not _validate_uuid_fields(improved_answers):
        logger.error("UUID validation failed; invalid feedback_id or interaction_id")
        return []

    logger.info("Creating enrichments", count=len(improved_answers))

    enrichments: List[FeedbackEnrichment] = []

    for improved_answer_item in improved_answers:
        question = improved_answer_item["question"]
        improved_answer = improved_answer_item["improved_answer"]
        new_context = improved_answer_item["new_context"]

        report_text = await _generate_enrichment_report(
            question, improved_answer, new_context, report_prompt_location
        )

        enrichment = await _create_enrichment_datapoint(improved_answer_item, report_text)

        if enrichment:
            enrichments.append(enrichment)
        else:
            logger.warning(
                "Failed to create enrichment",
                question=question,
                interaction_id=improved_answer_item.get("interaction_id"),
            )

    logger.info("Created enrichments", successful=len(enrichments))
    return enrichments
