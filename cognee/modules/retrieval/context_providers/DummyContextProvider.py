from typing import List

from cognee.infrastructure.context.BaseContextProvider import (
    BaseContextProvider,
)
from cognee.modules.engine.models import Entity


class DummyContextProvider(BaseContextProvider):
    """Simple context getter that returns a constant context."""

    async def get_context(self, entities: list[Entity], query: str) -> str:
        return "Albert Einstein was a theoretical physicist."
