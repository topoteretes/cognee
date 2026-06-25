"""Pure source-ref state transitions for graph-native provenance (COG-5522).

Backend-agnostic logic shared by every graph adapter that implements the Part 0
provenance contract. It operates only on lists of contract strings — no storage,
connection, or query concerns — so the set-merge / dedupe / derive / run-ref
rules live in ONE place and stay identical across Ladybug, Neo4j, Postgres, etc.

Note on concurrency: these are pure functions over the values an adapter has
already read. They do not make an adapter's read-modify-write atomic — a caller
that stamps the same artifact from two concurrent operations can still lose an
update in the read→write window. See ``phase1_storage_capabilities.md``.
"""

from typing import Any, List, NamedTuple, Optional
from uuid import UUID

from .source_refs import (
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
    make_source_run_ref,
)


class ProvenanceColumns(NamedTuple):
    """The four provenance fields stored on a graph node or edge."""

    source_ref_keys: List[str]
    source_dataset_ids: List[str]
    source_run_ids: List[str]
    source_run_refs: List[str]


def coerce_run_uuid(pipeline_run_id: Any) -> UUID:
    """Accept a UUID or its string form (the contract passes ``str``)."""
    return pipeline_run_id if isinstance(pipeline_run_id, UUID) else UUID(str(pipeline_run_id))


def derive_dataset_ids(source_ref_keys: List[str]) -> List[str]:
    """Materialized dataset filter derived from source_ref_keys (sorted, unique)."""
    return sorted({str(get_dataset_id_from_source_ref_key(key)) for key in source_ref_keys})


def derive_run_ids(source_run_refs: List[str]) -> List[str]:
    """Materialized rollback filter derived from source_run_refs (sorted, unique)."""
    return sorted({str(get_pipeline_run_id_from_source_run_ref(ref)) for ref in source_run_refs})


def provenance_after_attach(
    current_keys: List[str],
    current_run_refs: List[str],
    add_keys: List[str],
    pipeline_run_id: Optional[str],
) -> ProvenanceColumns:
    """Return the four provenance columns after attaching ``add_keys``.

    Part 0 contract: source_ref_keys are set-merged (deduped, order preserved); a
    source_run_ref is recorded for every ``(run, key)`` pair when a
    pipeline_run_id is given — independent of whether the key was already present,
    so a later run re-touching an existing key stays rollbackable. A write without
    a pipeline_run_id records no run ref (non-rollbackable by run id).
    """
    keys = list(current_keys)
    for key in add_keys:
        if key not in keys:
            keys.append(key)

    run_refs = list(current_run_refs)
    if pipeline_run_id is not None:
        run_uuid = coerce_run_uuid(pipeline_run_id)
        for key in add_keys:
            run_ref = make_source_run_ref(run_uuid, key)
            if run_ref not in run_refs:
                run_refs.append(run_ref)

    return ProvenanceColumns(keys, derive_dataset_ids(keys), derive_run_ids(run_refs), run_refs)


def provenance_after_remove(
    current_keys: List[str],
    current_run_refs: List[str],
    remove_keys: List[str],
) -> ProvenanceColumns:
    """Return the four provenance columns after removing ``remove_keys``.

    Removing a key strips every source_run_ref that embeds it; source_dataset_ids
    and source_run_ids are re-derived from what remains, so the last ref/run for a
    dataset/run disappearing automatically drops that id.
    """
    removed = set(remove_keys)
    keys = [key for key in current_keys if key not in removed]
    run_refs = [
        ref
        for ref in current_run_refs
        if get_source_ref_key_from_source_run_ref(ref) not in removed
    ]
    return ProvenanceColumns(keys, derive_dataset_ids(keys), derive_run_ids(run_refs), run_refs)
