"""SDK entry point: export a dataset's memory to a portable format."""

from pathlib import Path
from typing import Literal, Optional, Union, overload
from uuid import UUID

from cognee.modules.migration.export import ExportResult, export_dataset
from cognee.modules.migration.snapshot import GraphSnapshot
from cognee.modules.observability import new_span, COGNEE_DATASET_NAME

_FileFormat = Literal["cogx", "json", "graphml", "cypher", "governance"]


@overload
async def export(
    dataset: Union[str, UUID] = "main_dataset",
    format: Literal["pydantic"] = "pydantic",
    destination: Optional[Union[str, Path]] = None,
    user=None,
    link_relations: bool = False,
) -> GraphSnapshot: ...


@overload
async def export(
    dataset: Union[str, UUID],
    format: _FileFormat,
    destination: Optional[Union[str, Path]] = None,
    user=None,
    link_relations: bool = False,
) -> ExportResult: ...


@overload
async def export(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    format: _FileFormat,
    destination: Optional[Union[str, Path]] = None,
    user=None,
    link_relations: bool = False,
) -> ExportResult: ...


async def export(
    dataset: Union[str, UUID] = "main_dataset",
    format: str = "pydantic",
    destination: Optional[Union[str, Path]] = None,
    user=None,
    link_relations: bool = False,
) -> Union[ExportResult, GraphSnapshot]:
    """Export a dataset's knowledge graph.

    Args:
        dataset: Dataset name or id. Requires read permission.
        format: One of:
            - ``"pydantic"`` (default) — return a :class:`GraphSnapshot` of
              typed DataPoint instances (real ``Entity``/``DocumentChunk``/
              custom-model objects). Losslessly serializable:
              ``snapshot.model_dump_json()`` /
              ``GraphSnapshot.model_validate_json()``.
            - ``"cogx"`` — re-importable archive directory
              (``cognee.remember(COGXArchiveSource(path))`` restores it)
            - ``"json"`` / ``"graphml"`` / ``"cypher"`` — file exports
        destination: Output path. Optional for ``pydantic`` (saves JSON when
            given); a directory for ``cogx``; a file otherwise. File formats
            default to ``<dataset>_export.<ext>`` in the working directory.
        user: User context; defaults to the default user.
        link_relations: ``pydantic`` only — re-attach edges as object
            references on declared relation fields (e.g. ``entity.is_a``),
            turning the snapshot into a traversable object graph.

    Returns:
        GraphSnapshot for ``format="pydantic"``, ExportResult otherwise.

    Example::

        snapshot = await cognee.export("main_dataset")          # typed objects
        people = snapshot.find(Entity)                           # isinstance-real
        alice = snapshot.find(name="Alice")[0]
        snapshot.save("memory.json")                             # lossless dump

        result = await cognee.export("main_dataset", format="graphml")
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
            dataset=dataset,
            format=format,
            destination=destination,
            user=user,
            link_relations=link_relations,
        )

