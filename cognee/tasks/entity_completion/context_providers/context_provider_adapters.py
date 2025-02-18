from enum import Enum
from typing import Type

from cognee.tasks.entity_completion.context_providers.base_context_provider import (
    BaseContextProvider,
)
from cognee.tasks.entity_completion.context_providers.dummy_context_provider import (
    DummyContextProvider,
)


class ContextProviderAdapter(Enum):
    DUMMY = ("DummyProvider", DummyContextProvider)

    def __new__(cls, adapter_name: str, adapter_class: Type[BaseContextProvider]):
        obj = object.__new__(cls)
        obj._value_ = adapter_name
        obj.adapter_class = adapter_class
        return obj

    def __str__(self):
        return self.value
