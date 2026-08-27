from dataclasses import dataclass, field

from cognee.infrastructure.databases.unified import GraphVectorStoreInterface
from cognee.infrastructure.databases.unified.provenance_delete_planner import (
    SourceRefRemovalResult,
)


@dataclass
class FakeGraphVectorStore(GraphVectorStoreInterface):
    """Test-only implementation of the unified graph/vector provenance contract."""

    deleted_source_refs: list[str] = field(default_factory=list)
    deleted_dataset_ids: list[str] = field(default_factory=list)
    rolled_back_pipeline_run_ids: list[str] = field(default_factory=list)

    async def delete_by_source_ref(self, source_ref_key: str) -> SourceRefRemovalResult:
        self.deleted_source_refs.append(source_ref_key)
        return SourceRefRemovalResult()

    async def delete_by_dataset_id(self, dataset_id: str) -> SourceRefRemovalResult:
        self.deleted_dataset_ids.append(dataset_id)
        return SourceRefRemovalResult()

    async def rollback_by_pipeline_run_id(self, pipeline_run_id: str) -> None:
        self.rolled_back_pipeline_run_ids.append(pipeline_run_id)
