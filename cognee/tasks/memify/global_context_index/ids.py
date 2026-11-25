from __future__ import annotations

from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.pipelines.models import PipelineContext


def create_bucket_id(dataset_id: str, level: int, child_ids: list[str]) -> UUID:
    child_key = ",".join(sorted(child_ids))
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:{level}:{child_key}")


def create_root_summary_id(dataset_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:root")


def create_document_bucket_id(dataset_id: str, document_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:doc:{document_id}")


def dataset_id_from_context(ctx: PipelineContext | None) -> str:
    dataset = ctx.dataset if ctx else None
    dataset_id = getattr(dataset, "id", dataset)
    return str(dataset_id) if dataset_id is not None else ""
