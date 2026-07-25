"""SDK entry point: push a local dataset's knowledge graph to Cognee Cloud.

Bridges ``cognee.export(format="cogx")`` and the remote remember endpoint:
the dataset's graph is exported to a COGX archive, packed as a tarball, and
uploaded via :class:`CloudClient`, where the receiving instance imports it
through :class:`COGXArchiveSource` — preserving the local graph instead of
re-deriving it from raw files (which is what ``sync`` does).

Authentication reuses the ``serve`` stack: run ``cognee-cli serve`` (or
``await cognee.serve()``) once, and ``push`` picks up the saved credentials.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("push")


@dataclass
class PushResult:
    """Outcome of a ``cognee.push()`` call."""

    status: str
    dataset_name: str
    target_dataset: str
    num_nodes: int
    num_edges: int
    remote_response: dict
    # Remote migration pipeline run id; poll it when run_in_background=True.
    pipeline_run_id: Optional[str] = None

    def __repr__(self):
        return (
            f"PushResult(status={self.status!r}, dataset={self.dataset_name!r}, "
            f"target={self.target_dataset!r}, nodes={self.num_nodes}, edges={self.num_edges}"
            + (f", pipeline_run_id={self.pipeline_run_id!r}" if self.pipeline_run_id else "")
            + ")"
        )


def _resolve_client(url: Optional[str], api_key: Optional[str]):
    """Build or reuse a CloudClient. Returns ``(client, created)``.

    Precedence (matching ``serve()``): explicit ``url`` argument → live
    ``serve()`` connection → ``COGNEE_SERVICE_URL`` / ``COGNEE_API_KEY``
    environment variables → saved credentials from ``cognee-cli serve``.
    """
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import load_credentials
    from cognee.api.v1.serve.state import get_remote_client

    if url:
        return CloudClient(url, api_key or os.getenv("COGNEE_API_KEY", "")), True

    client = get_remote_client()
    if client is not None:
        return client, False

    env_url = os.getenv("COGNEE_SERVICE_URL")
    if env_url:
        return CloudClient(env_url, api_key or os.getenv("COGNEE_API_KEY", "")), True

    credentials = load_credentials()
    if credentials and credentials.service_url:
        return CloudClient(credentials.service_url, api_key or credentials.api_key), True

    raise RuntimeError(
        "No Cognee Cloud connection configured. Run `cognee-cli serve` to log in, "
        "pass url/api_key, or set COGNEE_SERVICE_URL and COGNEE_API_KEY."
    )


def _verify_migration_import(response: dict) -> None:
    """Ensure the remote server actually ran a COGX migration import.

    Servers that predate COGX archive support accept the upload but ingest the
    tarball as a regular file; their response lacks the ``migration_import``
    item the import path always emits.
    """
    items = response.get("items") or []
    if not any(isinstance(item, dict) and item.get("kind") == "migration_import" for item in items):
        raise RuntimeError(
            "The remote server did not perform a COGX archive import — it likely runs an "
            "older Cognee version without migration support. The uploaded archive was NOT "
            "imported as a knowledge graph (it may have been ingested as a plain file). "
            "Upgrade the remote instance, or use cognee.sync()/remember() with raw data."
        )


async def push(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    target_dataset: Optional[str] = None,
    mode: str = "preserve",
    run_in_background: bool = False,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    user=None,
) -> PushResult:
    """Upload a local dataset's knowledge graph to a Cognee Cloud instance.

    Args:
        dataset: Local dataset name or id. Requires read permission.
        target_dataset: Dataset name on the remote instance; defaults to the
            local dataset's name.
        mode: Import fidelity on the remote side —
            ``"preserve"`` (default) maps the exported entities and facts
            directly into the remote graph with zero LLM calls;
            ``"hybrid"`` also cognifies the raw content;
            ``"re-derive"`` ignores the exported graph and rebuilds from
            raw content.
        run_in_background: If True, the remote import is scheduled and the
            call returns once the upload completes; poll the returned
            ``pipeline_run_id`` for progress. Recommended for large graphs.
        url: Remote instance URL; falls back to the active ``serve()``
            connection, ``COGNEE_SERVICE_URL``, or saved credentials.
        api_key: API key; falls back like ``url``.
        user: Local user context for the export; defaults to the default user.

    Returns:
        A :class:`PushResult` with the export counts, the remote pipeline run
        id (when available), and the raw remote remember response.
    """
    from cognee.api.v1.export.export import export
    from cognee.modules.migration.archive import ARCHIVE_SUFFIX, pack_archive
    from cognee.modules.migration.sources.base import IMPORT_MODES
    from cognee.modules.observability import COGNEE_DATASET_NAME, new_span
    from cognee.shared.utils import send_telemetry

    if mode not in IMPORT_MODES:
        raise ValueError(f"Unknown push mode {mode!r}. Expected one of {IMPORT_MODES}.")

    client, created_client = _resolve_client(url, api_key)

    with new_span("cognee.api.push") as span:
        span.set_attribute(COGNEE_DATASET_NAME, str(dataset))
        send_telemetry(
            "cognee.push",
            user or "sdk",
            additional_properties={"mode": mode, "dataset": str(dataset)},
        )
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                archive_dir = Path(temporary_directory) / "cogx"
                result = await export(dataset, format="cogx", destination=archive_dir, user=user)

                if result.num_nodes == 0:
                    raise ValueError(
                        f"Dataset {result.dataset_name!r} exported 0 nodes — nothing to push. "
                        "Run cognee.cognify() (or cognee.remember()) on the dataset first to "
                        "build its knowledge graph."
                    )

                tar_path = Path(temporary_directory) / f"{result.dataset_name}{ARCHIVE_SUFFIX}"
                pack_archive(archive_dir, tar_path)

                logger.info(
                    "Pushing dataset %s (%d nodes, %d edges) to %s",
                    result.dataset_name,
                    result.num_nodes,
                    result.num_edges,
                    client.service_url,
                )

                with open(tar_path, "rb") as archive_file:
                    response = await client.remember(
                        archive_file,
                        dataset_name=target_dataset or result.dataset_name,
                        content_type="cogx-archive",
                        import_mode=mode,
                        run_in_background=run_in_background,
                    )

            _verify_migration_import(response)

            return PushResult(
                status=str(response.get("status", "unknown")),
                dataset_name=result.dataset_name,
                target_dataset=target_dataset or result.dataset_name,
                num_nodes=result.num_nodes,
                num_edges=result.num_edges,
                remote_response=response,
                pipeline_run_id=response.get("pipeline_run_id"),
            )
        finally:
            if created_client:
                await client.close()
