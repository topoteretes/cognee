from pydantic import BaseModel, field_validator

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.engine.models import EntityTypeCategory
from cognee.shared.logging_utils import get_logger

logger = get_logger("classify_entity_types")

# An answer is about 20 tokens, so one call stays far below the completion budget.
NAMES_PER_CALL = 200


class EntityTypeCategoryAnswer(BaseModel):
    name: str
    category: EntityTypeCategory

    @field_validator("category", mode="before")
    @classmethod
    def off_taxonomy_is_other(cls, value):
        # One label outside the taxonomy would fail the whole call and lose every
        # other answer in it.
        if value in {category.value for category in EntityTypeCategory}:
            return value
        return EntityTypeCategory.other


class EntityTypeCategories(BaseModel):
    answers: list[EntityTypeCategoryAnswer]


async def classify_entity_type_names(names: list[str]) -> dict[str, str]:
    """The category of each EntityType name the LLM answered for, keyed by name.

    A name the model left out, or a call that failed, is missing from the result,
    so the caller leaves it unclassified and a later run asks again. A label outside
    the taxonomy is "other".
    """
    system_prompt = read_query_prompt("classify_entity_types.txt")
    categories: dict[str, str] = {}

    for start in range(0, len(names), NAMES_PER_CALL):
        batch = names[start : start + NAMES_PER_CALL]
        try:
            result = await LLMGateway.acreate_structured_output(
                text_input="\n".join(batch),
                system_prompt=system_prompt,
                response_model=EntityTypeCategories,
            )
        except Exception:
            # Classification is an enrichment: its failure must not drop the graph
            # the run has already extracted.
            logger.warning(
                "Could not classify %d entity type names; they stay unclassified",
                len(batch),
                exc_info=True,
            )
            continue

        asked = set(batch)
        categories.update(
            {
                answer.name: answer.category.value
                for answer in result.answers
                if answer.name in asked
            }
        )

    return categories
