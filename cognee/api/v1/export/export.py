"""SDK entry point: export a dataset's memory to a portable format."""

from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from cognee.modules.migration.export import ExportResult, export_dataset
from cognee.modules.observability import new_span, COGNEE_DATASET_NAME


async def export(
    dataset: Union[str, UUID] = "main_dataset",
    format: str = "cmif",
    destination: Optional[Union[str, Path]] = None,
    user=None,
) -> ExportResult:
    """Export a dataset's knowledge graph to a portable format.

    Args:
        dataset: Dataset name or id. Requires read permission.
        format: One of ``"cmif"`` (re-importable archive, default),
            ``"json"``, ``"graphml"``, or ``"cypher"``.
        destination: Output path (a directory for ``cmif``, a file otherwise).
            Defaults to ``<dataset>_export.<ext>`` in the working directory.
        user: User context; defaults to the default user.

    Returns:
        ExportResult with the destination path and node/edge counts.

    Example::

        result = await cognee.export("main_dataset", format="graphml")
        print(result)  # ExportResult(format='graphml', nodes=1234, edges=5678, ...)
    """
    from cognee.shared.utils import send_telemetry

    with new_span("cognee.api.export") as span:
        span.set_attribute(COGNEE_DATASET_NAME, str(dataset))
        send_telemetry(
            "cognee.export",
            user or "sdk",
            additional_properties={"format": format, "dataset": str(dataset)},
        )
        return await export_dataset(
            dataset=dataset, format=format, destination=destination, user=user
        )
