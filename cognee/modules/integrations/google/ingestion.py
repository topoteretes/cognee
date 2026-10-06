"""Names hosts import from before the helpers became provider-neutral.

The Cloud pod loads this module by its Google name, so it stays until the pod
switches to ``cognee.modules.integrations.ingestion``. Everything lives there.
"""

from cognee.modules.integrations.ingestion import (
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
