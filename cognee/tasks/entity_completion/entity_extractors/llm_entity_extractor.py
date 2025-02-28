import json
import logging
import re
from typing import List

from pydantic import BaseModel, ValidationError

from cognee.infrastructure.entities.BaseEntityExtractor import BaseEntityExtractor
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.modules.retrieval.utils.completion import generate_completion

logger = logging.getLogger("llm_entity_extractor")


class ExtractedEntity(BaseModel):
    """Model for an entity extracted by the LLM."""

    name: str
    type: str
    description: str


class EntityList(BaseModel):
    """Response model containing a list of extracted entities."""

    entities: List[ExtractedEntity]


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

            raw_response = await generate_completion(
                query=text,
                context=text,  # Using the same text as context
                user_prompt_path=self.user_prompt_template,
                system_prompt_path=self.system_prompt_template,
            )

            try:
                if not raw_response.strip().startswith("{"):
                    json_match = re.search(r"(\{.*\})", raw_response, re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(1)
                        response_dict = json.loads(potential_json)
                    else:
                        logger.error("Could not find JSON content in the response")
                        return []
                else:
                    response_dict = json.loads(raw_response)

                response = EntityList.model_validate(response_dict)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                return []

            entities = []
            for extracted in response.entities:
                entity_type = self._get_entity_type(extracted.type)
                entity = Entity(
                    name=extracted.name, is_a=entity_type, description=extracted.description
                )
                entities.append(entity)

            logger.info(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
