"""Sync Notion through the SDK source's incremental, deletion-aware path.

Core owns OAuth, root selection and dataset ownership. ``notion_source`` owns the
walk, rendering, cursors and delete tombstones. A root is stored as ``page:<id>`` or
``data_source:<id>``, because ``notion_source`` takes the two kinds as separate lists.
"""

import re

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import revoke_credential_if_current
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.notion import client
from cognee.shared.logging_utils import get_logger

logger = get_logger("notion_integration")

NOTION_DATASET_PREFIX = "notion"
SELECTION_KEY = "selected_roots"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"
# What Notion says about an object the integration was not shared (or that is gone):
# the walk up from a root ends there, and the root is kept.
_NOT_REACHABLE = frozenset({"object_not_found", "restricted_resource", "http_403", "http_404"})
_ROOT_KINDS = ("page", "data_source")
_ID = re.compile(r"[0-9a-f]{32}")


def resource_id(kind: str, object_id: str) -> str:
    """A selection entry, with the id in one spelling.

    The entry is hashed into the dlt table name, so two spellings of one page (dashes,
    case) would be two tables and the page would be ingested twice.
    """
    return f"{kind}:{client.canonical_id(object_id)}"


def _split(root: str) -> tuple[str, str]:
    """``(kind, id)`` of a selection entry, with the id in its canonical spelling."""
    kind, _, object_id = root.partition(":")
    object_id = client.canonical_id(object_id)
    if kind not in _ROOT_KINDS or not _ID.fullmatch(object_id):
        raise ValueError(f"Notion selection entry {root!r} is not page:<id> or data_source:<id>")
    return kind, object_id


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


async def without_nested_roots(access_token: str, roots: list[str]) -> list[str]:
    """The roots whose parent chain does not reach another selected root.

    The outer root's walk already reads everything under it, so syncing the inner root
    too would ingest and cognify its pages twice under two node sets. The chain is
    followed one request per level (a page's parent can be a block, and a data
    source's parent is its database). A chain that cannot be followed because Notion
    reports an object as not shared is left alone and the root is kept. Any other
    failure raises, so a flaky lookup never silently changes what is synced.
    """
    # ``_split`` refuses an entry that is not page:<id> or data_source:<id>, before any request.
    selected = {client.canonical_id(_split(root)[1]): root for root in roots}
    parents: dict[tuple[str, str], tuple[str, str] | None] = {}

    async def parent_of(kind: str, object_id: str) -> tuple[str, str] | None:
        key = (kind, client.canonical_id(object_id))
        if key not in parents:
            try:
                parents[key] = await client.get_parent(access_token, kind, object_id)
            except client.NotionError as error:
                if error.code not in _NOT_REACHABLE:
                    raise
                parents[key] = None
        return parents[key]

    kept = []
    for root in roots:
        kind, object_id = _split(root)
        seen = {client.canonical_id(object_id)}
        parent = await parent_of(kind, object_id)
        nested = False
        while parent is not None and client.canonical_id(parent[1]) not in seen:
            if client.canonical_id(parent[1]) in selected:
                nested = True
                break
            seen.add(client.canonical_id(parent[1]))
            parent = await parent_of(*parent)
        if not nested:
            kept.append(root)
    return kept


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
    from cognee.modules.integrations.notion.adapter import access_token_for

    selected = (credential.provider_metadata or {}).get(SELECTION_KEY)
    if selected == []:
        await ingestion.retire_resources(
            "notion", credential, dataset_name_for_credential(credential), set()
        )
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}
    if not isinstance(selected, list) or not all(isinstance(root, str) for root in selected):
        raise ValueError("Notion selection must be a list of page:<id> or data_source:<id>")
    # One spelling per root, so the same page picked twice is one source and one table.
    roots = list(dict.fromkeys(resource_id(*_split(root)) for root in selected))

    source_factory = ingestion.source_factory("notion")
    access_token = await access_token_for(credential)
    await _require_accepted_token(credential, access_token)
    roots = await without_nested_roots(access_token, roots)

    def make_source(root: str, resource_name: str, check_active):
        kind, object_id = _split(root)
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
