from __future__ import annotations

from uuid import NAMESPACE_OID, UUID, uuid5


def create_bucket_id(dataset_id: str, level: int, child_ids: list[str]) -> UUID:
    child_key = ",".join(sorted(child_ids))
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:{level}:{child_key}")


def create_root_summary_id(dataset_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:root")
