"""SDK entry point: push a local dataset's knowledge graph to Cognee Cloud.

Bridges ``cognee.export(format="cmif")`` and the remote remember endpoint:
the dataset's graph is exported to a CMIF archive, packed as a tarball, and
uploaded via :class:`CloudClient`, where the receiving instance imports it
through :class:`CMIFArchiveSource` — preserving the local graph instead of
re-deriving it from raw files (which is what ``sync`` does).

Authentication reuses the ``serve`` stack: run ``cognee-cli serve`` (or
``await cognee.serve()``) once, and ``push`` picks up the saved credentials.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("push")


def _resolve_client(url: Optional[str], api_key: Optional[str]):
    """Build or reuse a CloudClient. Returns ``(client, created)``.

    Precedence: explicit ``url`` argument → live ``serve()`` connection →
    saved credentials from ``cognee-cli serve`` → ``COGNEE_SERVICE_URL`` /
    ``COGNEE_API_KEY`` environment variables.
    """
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import load_credentials
    from cognee.api.v1.serve.state import get_remote_client

    if url:
        return CloudClient(url, api_key or os.getenv("COGNEE_API_KEY", "")), True

    client = get_remote_client()
    if client is not None:
        return client, False

    credentials = load_credentials()
    if credentials and credentials.service_url:
        return CloudClient(credentials.service_url, api_key or credentials.api_key), True

    env_url = os.getenv("COGNEE_SERVICE_URL")
    if env_url:
        return CloudClient(env_url, api_key or os.getenv("COGNEE_API_KEY", "")), True

    raise RuntimeError(
        "No Cognee Cloud connection configured. Run `cognee-cli serve` to log in, "
        "pass url/api_key, or set COGNEE_SERVICE_URL and COGNEE_API_KEY."
    )


async def push(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    target_dataset: Optional[str] = None,
    mode: str = "preserve",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    user=None,
) -> dict:
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
            raw content (works against older cloud versions).
        url: Remote instance URL; falls back to the active ``serve()``
            connection, saved credentials, or ``COGNEE_SERVICE_URL``.
        api_key: API key; falls back like ``url``.
        user: Local user context for the export; defaults to the default user.

    Returns:
        The remote remember response, with ``num_nodes``/``num_edges``
        export counts added.
    """
    from cognee.api.v1.export.export import export
    from cognee.modules.migration.archive import ARCHIVE_SUFFIX, pack_archive
    from cognee.modules.migration.export import ExportResult
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
                archive_dir = Path(temporary_directory) / "cmif"
                result = await export(dataset, format="cmif", destination=archive_dir, user=user)
                if not isinstance(result, ExportResult):  # format="cmif" always returns one
                    raise RuntimeError("CMIF export did not return an ExportResult.")

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
                        content_type="cmif-archive",
                        import_mode=mode,
                    )

            response["num_nodes"] = result.num_nodes
            response["num_edges"] = result.num_edges
            return response
        finally:
            if created_client:
                await client.close()
