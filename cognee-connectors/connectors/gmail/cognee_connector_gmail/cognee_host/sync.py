"""Moved from cognee.modules.integrations.gmail.sync (cognee dev, 2026-10-26).

Sync Gmail through the SDK connector's history and tombstone path.
"""

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

from cognee_connector_gmail.source import build_gmail_service_from_access_token, gmail_source
from cognee_connector_google.cognee_host.selected_sync import sync_selected_resources

GMAIL_DATASET_PREFIX = "gmail"


def _dataset_name(credential: IntegrationCredential) -> str:
    return ingestion.dataset_name_for_credential(GMAIL_DATASET_PREFIX, credential)


async def sync_gmail(credential: IntegrationCredential) -> None:
    """Run an initial or manual sync and record its outcome."""
    await ingestion.run_sync("gmail", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee_connector_gmail.cognee_host.adapter import access_token_for

    labels = (credential.provider_metadata or {}).get("selected_label_ids", [])
    if labels is not None and (
        not isinstance(labels, list)
        or not all(isinstance(label, str) and label for label in labels)
    ):
        raise ValueError("Gmail selection must be a list of label IDs or null")
    # This package's source, not cognee's registry: on cognee 1.6.x that names
    # cognee's own bundled copy of it (cognee_connector_google.cognee_host.selected_sync).
    return await sync_selected_resources(
        "gmail",
        credential,
        counts,
        make_source=gmail_source,
        build_service=build_gmail_service_from_access_token,
        selection=labels,
        selection_kwarg="label_ids",
        dataset_name=_dataset_name(credential),
        access_token_for=access_token_for,
    )
