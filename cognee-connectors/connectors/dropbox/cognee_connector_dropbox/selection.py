"""The folder selection and the dataset name of a Dropbox connection.

Copied from the host-free parts of cognee.modules.integrations.dropbox.sync (cognee dev,
2026-11-12): the selection key, the selection checks and the dataset name. The sync runner
itself stays in cognee. ``refuse_nested_selection`` drops the credential check cognee's
version starts with (``ingestion.require_active_credential``): that is the host's, and a host
runs it before it hands out the token.

The dataset name replicates cognee's ``ingestion.dataset_name_for_account`` for an account's
stored email and id, so a cognee server and any other host land the same account in the same
dataset. The name is a stored identifier: neither copy may change it once an account synced.
"""

import re
from hashlib import sha256
from itertools import pairwise

from cognee_connector_dropbox import client
from cognee_connector_dropbox.source import valid_folder_id

DROPBOX_DATASET_PREFIX = "dropbox"
SELECTION_KEY = "selected_folder_ids"


class SelectedFolderNotFoundError(ValueError):
    """A selected folder id Dropbox cannot resolve (deleted, or not visible to the account)."""


class NestedSelectionError(ValueError):
    """One selected folder is inside another, so its files would be two documents."""


def dataset_name_for_account(email: str, account_id: str) -> str:
    """``dropbox_{email slug}_{10 hex of the account id}``, as cognee names it.

    The slug rule and the suffix length are cognee's ``ingestion.dataset_name_for_account``;
    the suffix keeps accounts apart whose slugs collide (``a.b`` and ``a_b``).
    """
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", email or "").strip("_").lower()
    name = f"{DROPBOX_DATASET_PREFIX}_{slug or 'account'}"
    return f"{name}_{sha256(account_id.encode()).hexdigest()[:10]}" if account_id else name


def check_selection_shape(selected: object) -> list[str]:
    """The selection as sorted folder ids, or ValueError when it is not a list of ``id:...``."""
    if not isinstance(selected, list) or not all(valid_folder_id(folder) for folder in selected):
        raise ValueError("Dropbox selection must be a list of folder ids (id:...)")
    return sorted(set(selected))


def _has_path(meta: dict) -> bool:
    """Whether a folder's metadata carries the path the nesting check compares; one without
    (unmounted) cannot be placed in the tree."""
    return isinstance(meta.get("path_lower"), str) and bool(meta["path_lower"])


async def refuse_nested_selection(
    access_token: str, folder_ids: list[str], *, refuse_missing: bool = False
) -> None:
    """Raise NestedSelectionError when a selected folder sits inside another one.

    One HTTP session for the whole selection. Paths are compared sorted with a trailing slash:
    a folder inside another sorts right after it (or after a folder that is itself inside it),
    so checking each against its neighbour is enough. A folder without a path (unmounted)
    takes no part: its "/" would be a prefix of all.

    ``refuse_missing`` (when a selection is saved) raises SelectedFolderNotFoundError for an
    id Dropbox cannot resolve. A sync lets it through: the source counts it as gone.
    """
    metas = await client.get_folders(access_token, folder_ids)
    if refuse_missing and not all(meta is not None and _has_path(meta) for meta in metas):
        raise SelectedFolderNotFoundError(
            "A selected Dropbox folder could not be found; pick it again from the list"
        )
    paths = sorted(
        meta["path_lower"] + "/" for meta in metas if meta is not None and _has_path(meta)
    )
    if any(path.startswith(previous) for previous, path in pairwise(paths)):
        raise NestedSelectionError("A selected Dropbox folder is inside another selected folder")
