import asyncio
from typing import Any, AsyncIterator
from cognee.modules.migration.cogx import COGXDocument, COGXEpisode, COGXTurn, COGXRecord
from cognee.modules.migration.sources.base import MemorySource


class LangChainMemorySource(MemorySource):
    source_system = "langchain"

    def __init__(self, data: Any, mode: str = "re-derive"):
        super().__init__(mode=mode)
        self.data = data

    async def records(self) -> AsyncIterator[COGXRecord]:
        items = self.data
        if not isinstance(items, list):
            if hasattr(items, "messages"):
                items = items.messages
            elif hasattr(items, "get_messages") and callable(items.get_messages):
                items = items.get_messages()
                if asyncio.iscoroutine(items):
                    items = await items
            else:
                items = [items]

        is_messages = False
        first_item = items[0] if items else None
        if first_item:
            if (
                hasattr(first_item, "type")
                or hasattr(first_item, "role")
                or "Message" in type(first_item).__name__
            ):
                is_messages = True

        if is_messages:
            turns = []
            for item in items:
                content = getattr(item, "content", "")
                if not isinstance(content, str):
                    content = str(content)
                role = "user"
                item_type = getattr(item, "type", "")
                if item_type == "ai":
                    role = "assistant"
                elif item_type == "system":
                    role = "system"
                elif hasattr(item, "role"):
                    role = item.role
                turns.append(COGXTurn(role=role, content=content))
            if turns:
                yield COGXEpisode(
                    external_system=self.source_system,
                    external_id="langchain-chat-history",
                    turns=turns,
                )
        else:
            for index, item in enumerate(items):
                content = getattr(item, "page_content", None)
                if content is None:
                    content = getattr(item, "content", "")
                metadata = getattr(item, "metadata", {}) or {}
                yield COGXDocument(
                    external_system=self.source_system,
                    external_id=str(metadata.get("id") or f"langchain-doc-{index}"),
                    content=content,
                    metadata=metadata,
                )
