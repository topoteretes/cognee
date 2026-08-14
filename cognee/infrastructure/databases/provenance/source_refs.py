from uuid import UUID

from .constants import SOURCE_REF_PREFIX, SOURCE_RUN_REF_PREFIX


def make_source_ref_key(dataset_id: UUID, data_id: UUID) -> str:
    """Build the stable key for one owning dataset/data item pair."""
    return f"{SOURCE_REF_PREFIX}:{dataset_id}:{data_id}"


def get_dataset_id_from_source_ref_key(source_ref_key: str) -> UUID:
    """Extract the dataset id from a source ref key."""
    prefix, version, dataset_id, _data_id = source_ref_key.split(":", 3)
    if f"{prefix}:{version}" != SOURCE_REF_PREFIX:
        raise ValueError("Unsupported source ref key format")
    return UUID(dataset_id)


def get_data_id_from_source_ref_key(source_ref_key: str) -> UUID:
    """Extract the data item id from a source ref key."""
    prefix, version, _dataset_id, data_id = source_ref_key.split(":", 3)
    if f"{prefix}:{version}" != SOURCE_REF_PREFIX:
        raise ValueError("Unsupported source ref key format")
    return UUID(data_id)


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
