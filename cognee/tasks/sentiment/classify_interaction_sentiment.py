"""Task for classifying sentiment of recent user interactions."""

from __future__ import annotations

from typing import Iterable, List
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.models import NodeSet
from cognee.shared.logging_utils import get_logger

from .models import (
    InteractionSentiment,
    InteractionSentimentEvaluation,
    InteractionSnapshot,
    InteractionSentimentLabel,
)


logger = get_logger("sentiment.classify_interaction_sentiment")


SYSTEM_PROMPT = """
You are a sentiment analysis specialist for Cognee saved interactions.
Given a user question, the agent's answer, and optional context, analyse
the user's sentiment toward the interaction.

Return a single JSON object **only** with the following fields:
- sentiment: one of positive, neutral, or negative (lowercase strings)
- confidence: a number between 0 and 1 inclusive indicating certainty
- summary: a concise explanation (max ~60 words) supporting the label

Do not include additional keys, text, commentary, or markdown.
""".strip()


def _format_prompt(interaction: InteractionSnapshot) -> str:
    context_section = (
        f"Context:\n{interaction.context.strip()}\n\n" if interaction.context.strip() else ""
    )
    return (
        "Analyse the following interaction:\n\n"
        f"Question:\n{interaction.question}\n\n"
        f"Answer:\n{interaction.answer}\n\n"
        f"{context_section}"
        "Provide the overall sentiment from the user's perspective."
    )


async def classify_interaction_sentiment(
    interactions: Iterable[InteractionSnapshot],
    *,
    nodeset_name: str = "InteractionSentiments",
) -> List[InteractionSentiment]:
    """Classify sentiment for a collection of recent user interactions."""

    interactions = list(interactions or [])
    if not interactions:
        return []

    nodeset = NodeSet(id=uuid5(NAMESPACE_OID, nodeset_name), name=nodeset_name)
    sentiments: List[InteractionSentiment] = []

    for interaction in interactions:
        prompt = _format_prompt(interaction)

        evaluation = await LLMGateway.acreate_structured_output(
            text_input=prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=InteractionSentimentEvaluation,
        )

        summary = evaluation.summary.strip()
        logger.debug(
            "Classified sentiment for interaction %s: %s",
            interaction.interaction_id,
            evaluation.sentiment,
        )

        sentiments.append(
            InteractionSentiment(
                interaction_id=interaction.interaction_id,
                sentiment=InteractionSentimentLabel(evaluation.sentiment),
                confidence=float(evaluation.confidence),
                summary=summary,
                belongs_to_set=nodeset,
            )
        )

    return sentiments
