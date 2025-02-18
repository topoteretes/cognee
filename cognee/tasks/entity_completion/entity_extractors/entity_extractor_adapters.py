from enum import Enum
from typing import Type

from cognee.tasks.entity_completion.entity_extractors.base_entity_extractor import (
    BaseEntityExtractor,
)
from cognee.tasks.entity_completion.entity_extractors.dummy_entity_extractor import (
    DummyEntityExtractor,
)


class EntityExtractorAdapter(Enum):
    DUMMY = ("DummyExtractor", DummyEntityExtractor)

    def __new__(cls, adapter_name: str, adapter_class: Type[BaseEntityExtractor]):
        obj = object.__new__(cls)
        obj._value_ = adapter_name
        obj.adapter_class = adapter_class
        return obj

    def __str__(self):
        return self.value
