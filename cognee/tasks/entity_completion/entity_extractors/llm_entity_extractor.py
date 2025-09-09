from cognee.shared.logging_utils import get_logger
from typing import List

from pydantic import BaseModel

from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.infrastructure.entities.BaseEntityExtractor import BaseEntityExtractor
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.infrastructure.llm.LLMGateway import LLMGateway

logger = get_logger("llm_entity_extractor")


class EntityList(BaseModel):
    """Response model containing a list of extracted entities."""

    entities: List[Entity]


class LLMEntityExtractor(BaseEntityExtractor):
    """Entity extractor that uses an LLM to identify entities in text."""

    def __init__(
        self,
        system_prompt_template: str = "extract_entities_system.txt",
        user_prompt_template: str = "extract_entities_user.txt",
    ):
        """Initialize the LLM entity extractor."""
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self._entity_type_cache = {}

    def _get_entity_type(self, type_name: str) -> EntityType:
        """Get or create an EntityType object."""
        type_name = type_name.upper()

        if type_name not in self._entity_type_cache:
            self._entity_type_cache[type_name] = EntityType(
                name=type_name, description=f"Entity type for {type_name.lower()} entities"
            )

        return self._entity_type_cache[type_name]

    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using an LLM."""
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for entity extraction")
            return []

        try:
            logger.info(f"Extracting entities from text: {text[:100]}...")

            user_prompt = render_prompt(self.user_prompt_template, {"text": text})
            system_prompt = read_query_prompt(self.system_prompt_template)

            response = await LLMGateway.acreate_structured_output(
                text_input=user_prompt,
                system_prompt=system_prompt,
                response_model=EntityList,
            )

            if not response.entities:
                logger.warning("No entities were extracted from the text")
                return []

            logger.info(f"Extracted {len(response.entities)} entities")
            return response.entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
