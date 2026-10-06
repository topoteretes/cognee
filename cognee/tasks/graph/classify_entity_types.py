from pydantic import BaseModel, field_validator

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import EntityType, EntityTypeCategory
from cognee.modules.graph.utils import collect_stored_data_points
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
            # The model configured for extraction, not the base one: this call is part
            # of reading the graph out of the text, and may be a different provider.
            with pipeline_stage("extraction"):
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


async def classify_chunk_entity_types(chunks: list[DocumentChunk]) -> None:
    """File the unclassified EntityTypes that storing the chunks would write, in memory.

    Runs before the chunks are stored, so the category is written with the node and
    needs no second write. Types restored from the graph already have one and are
    skipped, which leaves only the names this run introduced. The types come from
    the same walk storage does, so one an ontology linked through ``relations``
    is classified too.
    """
    unclassified: dict[str, list[EntityType]] = {}
    for chunk in chunks:
        for data_point in await collect_stored_data_points(chunk):
            if isinstance(data_point, EntityType) and data_point.category is None:
                unclassified.setdefault(data_point.name, []).append(data_point)

    categories = await classify_entity_type_names(list(unclassified))
    for name, category in categories.items():
        for entity_type in unclassified[name]:
            entity_type.category = category
