from typing import List

from cognee.modules.engine.models import Entity, EntityType
from cognee.tasks.entity_completion.entity_extractors.base_entity_extractor import (
    BaseEntityExtractor,
)


class DummyEntityExtractor(BaseEntityExtractor):
    """Simple entity extractor that returns a constant entity."""

    async def extract_entities(self, text: str) -> List[Entity]:
        entity_type = EntityType(name="Person", description="A human individual")
        entity = Entity(name="Albert Einstein", is_a=entity_type, description="A famous physicist")
        return [entity]
