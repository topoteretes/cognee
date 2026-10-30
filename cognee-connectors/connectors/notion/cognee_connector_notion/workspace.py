"""The connected workspace of a Notion connection: the dataset it syncs into and its root
selection, the entries' spelling and their nesting.

Split out of cognee.modules.integrations.notion.sync (cognee dev, 2026-10-27): the part
of that module that needs no host, so the connector's ``sync_scope`` and cognee's own
sync read a selection the same way. A root is stored as ``page:<id>`` or
``data_source:<id>``, because ``notion_source`` takes the two kinds as separate lists.
The dataset name replicates cognee's ``ingestion.dataset_name_for_account`` for a
connection that stores no email, so the contract object and the cognee adapter name the
same dataset without this package importing cognee.
"""

import re
from hashlib import sha256

from cognee_connector_notion import client

NOTION_DATASET_PREFIX = "notion"
SELECTION_KEY = "selected_roots"
# What Notion says about an object the integration was not shared (or that is gone):
# the walk up from a root ends there, and the root is kept.
_NOT_REACHABLE = frozenset({"object_not_found", "restricted_resource", "http_403", "http_404"})
_ROOT_KINDS = ("page", "data_source")
_ID = re.compile(r"[0-9a-f]{32}")


def dataset_name_for_workspace(workspace_id: str) -> str:
    """The dataset a workspace syncs into: ``notion_account_<10 hex of the workspace id>``.

    Named from the workspace id alone. The workspace name is not part of it: users
    rename workspaces, and a name that followed the rename would send every later sync
    to a new dataset and orphan the old. The formula is cognee's
    ``ingestion.dataset_name_for_account("notion", "", workspace_id)`` (the empty email
    slug, then a 10 hex suffix of the id), replicated here so a cognee server and the
    cloud land the same workspace in the same dataset. The name is a stored identifier,
    so neither side may change it once a workspace has synced.
    """
    name = f"{NOTION_DATASET_PREFIX}_account"
    return f"{name}_{sha256(workspace_id.encode()).hexdigest()[:10]}" if workspace_id else name


def resource_id(kind: str, object_id: str) -> str:
    """A selection entry, with the id in one spelling.

    The entry is hashed into the dlt table name, so two spellings of one page (dashes,
    case) would be two tables and the page would be ingested twice.
    """
    return f"{kind}:{client.canonical_id(object_id)}"


def split_root(root: str) -> tuple[str, str]:
    """``(kind, id)`` of a selection entry, with the id in its canonical spelling."""
    kind, _, object_id = root.partition(":")
    object_id = client.canonical_id(object_id)
    if kind not in _ROOT_KINDS or not _ID.fullmatch(object_id):
        raise ValueError(f"Notion selection entry {root!r} is not page:<id> or data_source:<id>")
    return kind, object_id


async def without_nested_roots(access_token: str, roots: list[str]) -> list[str]:
    """The roots whose parent chain does not reach another selected root.

    The outer root's walk already reads everything under it, so syncing the inner root
    too would ingest and cognify its pages twice under two node sets. The chain is
    followed one request per level (a page's parent can be a block, and a data
    source's parent is its database). A chain that cannot be followed because Notion
    reports an object as not shared is left alone and the root is kept. Any other
    failure raises, so a flaky lookup never silently changes what is synced.
    """
    # ``split_root`` refuses an entry that is not page:<id> or data_source:<id>, before any request.
    selected = {client.canonical_id(split_root(root)[1]): root for root in roots}
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
        kind, object_id = split_root(root)
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


__all__ = [
    "NOTION_DATASET_PREFIX",
    "SELECTION_KEY",
    "dataset_name_for_workspace",
    "resource_id",
    "split_root",
    "without_nested_roots",
]
