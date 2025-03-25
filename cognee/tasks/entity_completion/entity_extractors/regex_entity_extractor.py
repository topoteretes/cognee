from cognee.shared.logging_utils import get_logger
from typing import List, Optional

from cognee.infrastructure.entities.BaseEntityExtractor import BaseEntityExtractor
from cognee.modules.engine.models import Entity
from cognee.root_dir import get_absolute_path
from cognee.tasks.entity_completion.entity_extractors.regex_entity_config import RegexEntityConfig

logger = get_logger("regex_entity_extractor")


class RegexEntityExtractor(BaseEntityExtractor):
    """Entity extractor that uses regular expressions to identify entities in text."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the regex entity extractor with an optional custom config path."""
        if config_path is None:
            config_path = get_absolute_path(
                "tasks/entity_completion/entity_extractors/regex_entity_config.json"
            )

        self.config = RegexEntityConfig(config_path)
        logger.info(
            f"Initialized RegexEntityExtractor with {len(self.config.get_entity_names())} entity types"
        )

    def _create_entity(self, match_text: str, entity_type_obj, description_template: str) -> Entity:
        """Create an entity from a regex match."""
        return Entity(
            name=match_text,
            is_a=entity_type_obj,
            description=description_template.format(match_text),
        )

    def _extract_entities_by_type(self, entity_type: str, text: str) -> List[Entity]:
        """Extract entities of a specific type from the given text."""
        try:
            pattern = self.config.get_compiled_pattern(entity_type)
            description_template = self.config.get_description_template(entity_type)
            entity_type_obj = self.config.get_entity_type(entity_type)

            return [
                self._create_entity(match.group(0), entity_type_obj, description_template)
                for match in pattern.finditer(text)
            ]
        except KeyError:
            logger.warning(f"Unknown entity type: {entity_type}")
            return []

    def _text_to_entities(self, text: str) -> List[Entity]:
        """Extract all entity types from the given text and return them as a list."""
        all_entities = []

        for entity_type in self.config.get_entity_names():
            extracted_entities = self._extract_entities_by_type(entity_type, text)
            all_entities.extend(extracted_entities)

        logger.info(f"Extracted {len(all_entities)} entities")
        return all_entities

    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract all configured entity types from the given text."""
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for entity extraction")
            return []

        try:
            logger.info(f"Extracting entities from text: {text[:100]}...")
            return self._text_to_entities(text)
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
