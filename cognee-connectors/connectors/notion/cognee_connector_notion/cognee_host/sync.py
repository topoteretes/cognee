"""Sync Notion through the SDK source's incremental, deletion-aware path.

Moved from cognee.modules.integrations.notion.sync (cognee dev, 2026-10-27). The
helpers that need no host (``NOTION_DATASET_PREFIX``, ``SELECTION_KEY``,
``resource_id``, ``split_root`` and ``without_nested_roots``) live in
:mod:`cognee_connector_notion.workspace`, shared with the connector's ``sync_scope``
and ``dataset_name``.

Core owns OAuth, root selection and dataset ownership. ``notion_source`` owns the
walk, rendering, cursors and delete tombstones. A root is stored as ``page:<id>`` or
``data_source:<id>``, because ``notion_source`` takes the two kinds as separate lists.
"""

import logging

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import revoke_credential_if_current
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

from cognee_connector_notion import client
from cognee_connector_notion.source import notion_source
from cognee_connector_notion.workspace import (
    NOTION_DATASET_PREFIX,
    SELECTION_KEY,
    resource_id,
    split_root,
    without_nested_roots,
)

logger = logging.getLogger(__name__)

SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


def dataset_name_for_credential(credential: IntegrationCredential) -> str:
    """The dataset a workspace syncs into, named from the workspace id alone.

    The workspace name is not part of it: users rename workspaces, and a name that
    followed the rename would send every later sync to a new dataset and orphan the old.
    """
    return ingestion.dataset_name_for_account(
        NOTION_DATASET_PREFIX, "", str(credential.provider_account_id)
    )


def _classify_error(error: BaseException) -> str | None:
    """The count key a failed root is reported under, so a rate limit can be resumed."""
    return "failed_rate_limit" if getattr(error, "code", None) == "rate_limited" else None


async def sync_notion(credential: IntegrationCredential) -> None:
    """Run a manual sync and record its outcome."""
    await ingestion.run_sync("notion", credential, _sync_source)


async def _require_accepted_token(credential: IntegrationCredential, access_token: str) -> None:
    """Stop the sync, and revoke the connection, when Notion no longer accepts the token.

    A token stored with no expiry never goes through a refresh, so this is the only place
    a workspace that removed the integration at Notion shows up: without it the connection
    stays active and every sync fails.
    """
    try:
        await client.verify_token(access_token)
    except client.NotionError as error:
        if error.code in client.UNAUTHORIZED_CODES:
            await revoke_credential_if_current(credential)
            logger.warning(
                "Notion access for workspace %s was removed at the provider; "
                "local credential revoked",
                credential.provider_account_id,
            )
        raise


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee_connector_notion.cognee_host.adapter import access_token_for

    selected = (credential.provider_metadata or {}).get(SELECTION_KEY)
    if selected == []:
        await ingestion.retire_resources(
            "notion", credential, dataset_name_for_credential(credential), set()
        )
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}
    if not isinstance(selected, list) or not all(isinstance(root, str) for root in selected):
        raise ValueError("Notion selection must be a list of page:<id> or data_source:<id>")
    # One spelling per root, so the same page picked twice is one source and one table.
    roots = list(dict.fromkeys(resource_id(*split_root(root)) for root in selected))

    # This package's source, not ``ingestion.source_factory("notion")``: on cognee 1.6.x
    # that name resolves to cognee's own bundled copy of it.
    source_factory = notion_source
    access_token = await access_token_for(credential)
    await _require_accepted_token(credential, access_token)
    roots = await without_nested_roots(access_token, roots)

    def make_source(root: str, resource_name: str, check_active):
        kind, object_id = split_root(root)
        return source_factory(
            token=access_token,
            root_page_ids=[object_id] if kind == "page" else [],
            root_data_source_ids=[object_id] if kind == "data_source" else [],
            workspace_id=str(credential.provider_account_id),
            resource_name=resource_name,
            check_active=check_active,
        )

    retained = await ingestion.sync_scopes(
        "notion",
        credential,
        counts,
        scopes=roots,
        dataset_name=dataset_name_for_credential(credential),
        make_source=make_source,
        classify_error=_classify_error,
    )
    if not counts["failed"]:
        await ingestion.retire_resources(
            "notion", credential, dataset_name_for_credential(credential), retained
        )
    return (SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK), counts
