"""Helpers that compute the provenance refs stamped onto graph artifacts.

A **source ref** identifies the ingestion source of an artifact — the
``(dataset_id, data_id)`` pair. ``delete`` of a single data item removes that
data item's source ref from every node/edge; an artifact survives as long as
any other source ref remains. This is the graph-native equivalent of the
relational ledger's per-``(dataset_id, data_id)`` ownership rows.

A **source-run ref** identifies the pipeline run that produced an artifact —
the ``(dataset_id, pipeline_run_id)`` pair. ``rollback`` of a run removes that
run's source-run ref; an artifact is hard-deleted only when no source ref and
no other run keeps it alive. This mirrors the ledger's ``pipeline_run_id``
tagging, including its "only remove what this run introduced" intent.

Refs are opaque, deterministic, collision-resistant string tokens. Callers
treat them as keys and never parse them; the ``(dataset_id, data_id)`` and
``(dataset_id, pipeline_run_id)`` tuples are the source of truth.
"""

from uuid import NAMESPACE_OID, UUID, uuid5

# Distinct prefixes keep the two ref namespaces from ever colliding even if a
# data_id and a pipeline_run_id happened to share a value.
_SOURCE_REF_NAMESPACE = "cognee:source-ref:v1:"
_SOURCE_RUN_REF_NAMESPACE = "cognee:source-run-ref:v1:"


def make_source_ref(dataset_id: UUID, data_id: UUID) -> str:
    """Return the stable source ref for a ``(dataset_id, data_id)`` ingestion source.

    Deterministic: the same pair always yields the same ref, so re-ingesting a
    data item re-uses its ref instead of creating a duplicate.
    """
    return str(uuid5(NAMESPACE_OID, f"{_SOURCE_REF_NAMESPACE}{dataset_id}:{data_id}"))


def make_source_run_ref(dataset_id: UUID, pipeline_run_id: UUID) -> str:
    """Return the stable source-run ref for a ``(dataset_id, pipeline_run_id)`` run.

    Deterministic for the same pair. Rollback computes the same ref to find the
    artifacts a given run touched.
    """
    return str(uuid5(NAMESPACE_OID, f"{_SOURCE_RUN_REF_NAMESPACE}{dataset_id}:{pipeline_run_id}"))


__all__ = ["make_source_ref", "make_source_run_ref"]
