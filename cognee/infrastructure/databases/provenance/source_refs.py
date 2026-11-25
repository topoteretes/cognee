from collections.abc import Callable
from typing import NamedTuple
from uuid import UUID

from .constants import CHUNK_SOURCE_REF_PREFIX, SOURCE_REF_PREFIX, SOURCE_RUN_REF_PREFIX


def make_source_ref_key(dataset_id: UUID, data_id: UUID) -> str:
    """Build the stable key for one owning dataset/data item pair."""
    return f"{SOURCE_REF_PREFIX}:{dataset_id}:{data_id}"


def make_chunk_source_ref_key(dataset_id: UUID, data_id: UUID, chunk_id: UUID) -> str:
    """Build the v2 key for one owning chunk of one dataset document.

    Chunk-scoped ownership: graph/vector output produced from one chunk is
    stamped with the chunk that produced it, so deletion can operate at chunk
    scope (an incremental update replacing one chunk) while shared output
    survives until its LAST owner goes — the same refcounted planner, finer
    keys.
    """
    return f"{CHUNK_SOURCE_REF_PREFIX}:{dataset_id}:{data_id}:{chunk_id}"


def make_chunk_source_ref_key_prefix(dataset_id: UUID, data_id: UUID) -> str:
    """Build what every v2 chunk key of one dataset document starts with."""
    return f"{CHUNK_SOURCE_REF_PREFIX}:{dataset_id}:{data_id}:"


def data_source_ref_key_selector(
    dataset_id: UUID, data_id: UUID
) -> Callable[[list[str]], list[str]]:
    """Build the function that keeps, of a list of keys, the ones one dataset
    document owns: its v1 key and its v2 chunk keys.

    This is the rule a graph store applies when it selects a document's refs
    itself (``find_node_source_refs_by_data``), so an adapter that narrows the rows
    in its query and one that filters them in Python agree. The key and the prefix
    are built once here because the function runs for every node and edge.
    """
    document_key = make_source_ref_key(dataset_id, data_id)
    chunk_key_prefix = make_chunk_source_ref_key_prefix(dataset_id, data_id)

    def select(source_ref_keys: list[str]) -> list[str]:
        return [
            key
            for key in source_ref_keys
            if key == document_key or key.startswith(chunk_key_prefix)
        ]

    return select


class ParsedSourceRef(NamedTuple):
    """A source ref key decomposed: v1 doc-scope (chunk_id None) or v2 chunk-scope."""

    version: int
    dataset_id: UUID
    data_id: UUID
    chunk_id: UUID | None


def parse_source_ref_key(source_ref_key: str) -> ParsedSourceRef:
    """Decompose a v1 or v2 source ref key into its components.

    Raises ``ValueError("Unsupported source ref key format")`` for any key
    that does not decompose — unknown prefix, wrong segment count, malformed
    ids — so callers can treat every undecomposable key uniformly.
    """
    try:
        if source_ref_key.startswith(f"{SOURCE_REF_PREFIX}:"):
            _prefix, _version, dataset_id, data_id = source_ref_key.split(":", 3)
            return ParsedSourceRef(1, UUID(dataset_id), UUID(data_id), None)
        if source_ref_key.startswith(f"{CHUNK_SOURCE_REF_PREFIX}:"):
            _prefix, _version, dataset_id, data_id, chunk_id = source_ref_key.split(":", 4)
            return ParsedSourceRef(2, UUID(dataset_id), UUID(data_id), UUID(chunk_id))
    except ValueError:
        raise ValueError("Unsupported source ref key format")
    raise ValueError("Unsupported source ref key format")


def get_dataset_id_from_source_ref_key(source_ref_key: str) -> UUID:
    """Extract the dataset id from a source ref key (v1 or v2)."""
    return parse_source_ref_key(source_ref_key).dataset_id


def get_data_id_from_source_ref_key(source_ref_key: str) -> UUID:
    """Extract the data item id from a source ref key (v1 or v2)."""
    return parse_source_ref_key(source_ref_key).data_id


def make_source_run_ref(pipeline_run_id: UUID, source_ref_key: str) -> str:
    """Build the rollback key for a run adding one source ref to an artifact."""
    return f"{SOURCE_RUN_REF_PREFIX}:{pipeline_run_id}:{source_ref_key}"


def get_pipeline_run_id_from_source_run_ref(source_run_ref: str) -> UUID:
    """Extract the pipeline run id from a source run ref."""
    prefix, version, pipeline_run_id, _source_ref_key = source_run_ref.split(":", 3)
    if f"{prefix}:{version}" != SOURCE_RUN_REF_PREFIX:
        raise ValueError("Unsupported source run ref format")
    return UUID(pipeline_run_id)


def get_source_ref_key_from_source_run_ref(source_run_ref: str) -> str:
    """Extract the source ref key from a source run ref."""
    prefix, version, _pipeline_run_id, source_ref_key = source_run_ref.split(":", 3)
    if f"{prefix}:{version}" != SOURCE_RUN_REF_PREFIX:
        raise ValueError("Unsupported source run ref format")
    return source_ref_key
