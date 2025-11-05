from __future__ import annotations

from typing import List
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt
from cognee.shared.logging_utils import get_logger
from cognee.modules.engine.models import NodeSet

from .models import FeedbackEnrichment


logger = get_logger("create_enrichments")


def _validate_enrichments(enrichments: List[FeedbackEnrichment]) -> bool:
    """Validate that all enrichments contain required fields for completion."""
    return all(
        enrichment.question is not None
        and enrichment.original_answer is not None
        and enrichment.improved_answer is not None
        and enrichment.new_context is not None
        and enrichment.feedback_id is not None
        and enrichment.interaction_id is not None
        for enrichment in enrichments
    )


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


async def create_enrichments(
    enrichments: List[FeedbackEnrichment],
    report_prompt_location: str = "feedback_report_prompt.txt",
) -> List[FeedbackEnrichment]:
    """Fill text and belongs_to_set fields of existing FeedbackEnrichment DataPoints."""
    if not enrichments:
        logger.info("No enrichments provided; returning empty list")
        return []

    if not _validate_enrichments(enrichments):
        logger.error("Input validation failed; missing required fields")
        return []

    logger.info("Completing enrichments", count=len(enrichments))

    nodeset = NodeSet(id=uuid5(NAMESPACE_OID, name="FeedbackEnrichment"), name="FeedbackEnrichment")

    completed_enrichments: List[FeedbackEnrichment] = []

    for enrichment in enrichments:
        report_text = await _generate_enrichment_report(
            enrichment.question,
            enrichment.improved_answer,
            enrichment.new_context,
            report_prompt_location,
        )

        enrichment.text = report_text
        enrichment.belongs_to_set = [nodeset]

        completed_enrichments.append(enrichment)

    logger.info("Completed enrichments", successful=len(completed_enrichments))
    return completed_enrichments
