"""CSV ingestion through the DLT pipeline, packaged as a loader.

Registered above the plain ``csv_loader`` in the engine's priority order, so
when the ``dlt`` extra is installed every CSV takes the structured route:
staging ingestion via dlt, one manifest per file with a stable data_id, and
the no-LLM DLT cognify route (one DltRow node per row). Without the extra
this module fails to import, the loader is never registered, and CSVs fall
back to ``csv_loader``'s text flattening.

The loader returns a ``LoaderResult`` instead of a plain path: the manifest
text is the derived file, and the result carries the manifest's stable
``data_id`` plus the ``system_metadata`` route stamp that cognify's per-item
routing keys on. Per-call dlt options (``primary_key``, ``write_disposition``,
``max_rows_per_table``, ``column_value_columns``) ride the standard
``preferred_loaders`` config channel:
``preferred_loaders={"dlt_csv_loader": {"primary_key": "id"}}``.
"""

import hashlib
from typing import Any, Optional

import dlt  # noqa: F401 — hard gate: without the extra this loader must not register

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface, LoaderResult
from cognee.modules.ingestion.exceptions import IngestionError


class DltCsvLoader(LoaderInterface):
    """Structured CSV loader: dlt staging ingestion + manifest emission."""

    loader_name = "dlt_csv_loader"

    @property
    def supported_extensions(self) -> list[str]:
        return ["csv"]

    @property
    def supported_mime_types(self) -> list[str]:
        return ["text/csv"]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        return extension in self.supported_extensions and mime_type in self.supported_mime_types

    async def load(
        self,
        file_path: str,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[Any] = None,
        user: Optional[Any] = None,
        original_file_name: Optional[str] = None,
        primary_key: Optional[str] = None,
        write_disposition: str = "replace",
        max_rows_per_table: Optional[int] = None,
        column_value_columns: Optional[dict] = None,
        **kwargs: Any,
    ) -> LoaderResult:
        # Task-layer imports are lazy: the loaders package must not depend on
        # cognee.tasks at import time (loaders are infrastructure).
        from cognee.tasks.ingestion.create_dlt_source import (
            create_dlt_source_from_csv,
            csv_source_name,
        )
        from cognee.tasks.ingestion.ingest_dlt_source import ingest_dlt_source
        from cognee.tasks.ingestion.resolve_dlt_sources import _build_source_manifest_item

        if dataset_name is None or user is None:
            raise IngestionError(
                message="dlt_csv_loader runs only inside ingestion: dataset_name and "
                "user must be provided by the ingest pipeline."
            )

        # The bytes may be a localized copy (s3 temp download, stored upload)
        # whose basename is meaningless — the manifest identity derives from
        # the ORIGINAL file name so it stays stable across runs.
        source_name = csv_source_name(original_file_name or file_path)
        local_path = get_data_file_path(file_path)

        resource = create_dlt_source_from_csv(local_path, source_name=source_name)
        rows = await ingest_dlt_source(
            resource,
            dataset_name,
            primary_key=primary_key,
            write_disposition=write_disposition,
            max_rows_per_table=max_rows_per_table,
        )

        manifest_item = await _build_source_manifest_item(
            rows,
            source_name,
            dataset_name,
            user,
            column_value_columns=column_value_columns,
        )
        if manifest_item is None:
            raise IngestionError(
                message=f"CSV source '{source_name}' produced no rows — nothing to ingest."
            )

        manifest_text = manifest_item.data
        storage_file_name = (
            "dlt_manifest_" + hashlib.md5(manifest_text.encode()).hexdigest() + ".txt"
        )
        storage = get_file_storage(get_storage_config()["data_root_directory"])
        stored_path = await storage.store(storage_file_name, manifest_text)

        return LoaderResult(
            file_path=stored_path,
            data_id=manifest_item.data_id,
            system_metadata=manifest_item.system_metadata,
        )
