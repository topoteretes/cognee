"""Google names for the provider-neutral ingestion helpers.

Everything lives in ``cognee.modules.integrations.ingestion``; these names keep
existing callers working and are the same objects.
"""

from cognee.modules.integrations.ingestion import (
    _running_syncs,
    add_source_counts,
    build_service,
    dataset_summary,
    empty_resource,
    extraction_checkpoint,
    require_active_credential,
    resource_name,
    retire_resources,
    run_sync,
    source_counts,
    source_factory,
    sync_is_running,
)
