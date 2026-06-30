from typing import Any, AsyncIterator
from cognee.modules.migration.cogx import COGXDocument, COGXFact, COGXRecord
from cognee.modules.migration.sources.base import MemorySource


class LlamaIndexMemorySource(MemorySource):
    source_system = "llamaindex"

    def __init__(self, data: Any, mode: str = "re-derive"):
        super().__init__(mode=mode)
        self.data = data

    async def records(self) -> AsyncIterator[COGXRecord]:
        items = self.data
        if not isinstance(items, list):
            items = [items]

        for index, item in enumerate(items):
            node_id = getattr(item, "node_id", None) or getattr(
                item, "id_", f"llamaindex-node-{index}"
            )
            content = getattr(item, "text", "")
            if not content and hasattr(item, "get_content"):
                content = item.get_content()
            metadata = getattr(item, "metadata", {}) or {}

            yield COGXDocument(
                external_system=self.source_system,
                external_id=str(node_id),
                content=content,
                metadata=metadata,
            )

            relationships = getattr(item, "relationships", {}) or {}
            for rel_type, related_node in relationships.items():
                related_nodes = related_node if isinstance(related_node, list) else [related_node]
                for rn in related_nodes:
                    related_id = getattr(rn, "node_id", None) or getattr(rn, "id_", None)
                    if related_id:
                        rel_name = getattr(rel_type, "name", str(rel_type))
                        yield COGXFact(
                            external_system=self.source_system,
                            external_id=f"{node_id}-{rel_name}-{related_id}",
                            subject_ref=str(node_id),
                            predicate=rel_name,
                            object_ref=str(related_id),
                        )
