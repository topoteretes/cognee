"""The Google Drive connector on the shared contract: ``CONNECTOR``.

Tier 2: the Google descriptor (:func:`cognee_connector_google.google_descriptor`) with
Drive's read-only scope, on :class:`~cognee_connector_google.GoogleConnector` for the
userinfo lookup that keys the credential on the account's Google subject, plus what a
Drive connection can enumerate: the folder picker (``list_resources``: My Drive, the
shared drives, then every folder the account can see) and the sync scope
(``sync_scope``: the folders to sync, one scope each, the shared drives among them
marked, since the source lists a shared drive as a corpus rather than walking it as a
folder). Drive sends no webhooks on this path, so the descriptor declares an hourly poll
interval for hosts that schedule syncs; cognee's own adapter syncs on the user's Refresh
action only. Nothing in this module imports cognee.
"""

from collections.abc import Mapping
from typing import Any

from cognee_connector_google import GoogleConnector, google_descriptor
from cognee_connector_google_drive import resources as drive_resources
from cognee_connector_google_drive.settings import google_drive_settings
from cognee_connectors_core.base import ConnectorResource, Credential, SyncScope

# drive.readonly is a restricted scope and needs Google's OAuth verification before a
# public launch; openid and email are what identify the account the credential is keyed on.
SCOPE = "openid email https://www.googleapis.com/auth/drive.readonly"

#: The My Drive root: the first entry of the picker and the first scope of a sync that
#: was given no selection.
ROOT_RESOURCE_ID = "root"

DESCRIPTOR = google_descriptor("google_drive", "Google Drive", SCOPE)


class GoogleDriveConnector(GoogleConnector):
    """Drive on the contract: the Google OAuth flow plus folders and shared drives."""

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """My Drive, the shared drives and the folders the account can see, for a picker.

        The ids are what :meth:`sync_scope` takes back as ``resources``: ``root`` for
        My Drive, a shared drive's id for the drive as a whole, a folder's id otherwise.
        """
        token = self._bearer(credential, tokens)
        return [
            ConnectorResource(
                id=str(item["id"]),
                name=str(item["name"]),
                description=item.get("description"),
                attributes=dict(item.get("attributes") or {}),
            )
            for item in await drive_resources.list_resources(token)
        ]

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """The folders to sync, one scope each, with the shared drives among them marked.

        ``resources`` names folder ids, or shared-drive ids for a drive as a whole;
        ``None`` (the connect path and a poll) is the whole account: My Drive plus every
        shared drive, the same scopes cognee's own sync runs for a connection with no
        selection. An empty list is refused, as the contract requires: a webhook that
        implies no work is skipped by the caller, never passed through here.

        ``options["shared_drive_ids"]`` says which of the items are shared drives, so the
        source can pass each one as ``shared_drive_id`` and list it as a corpus. The items
        are the resource ids themselves, so ``item_resource_ids`` repeats them.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )
        token = self._bearer(credential, tokens)
        shared = await drive_resources.shared_drive_ids(token)
        if resources is None:
            items = [ROOT_RESOURCE_ID, *shared]
        else:
            # One scope per folder, however often the host named it.
            items = list(dict.fromkeys(str(folder) for folder in resources))
        return SyncScope(
            dataset_name=self.dataset_name(credential),
            content_type=self.content_type,
            items=items,
            item_resource_ids=list(items),
            credential=token,
            credential_expires_at=credential.token_expires_at,
            credential_refreshable=bool(tokens.get("refresh_token")),
            options={
                "folder_ids": list(items),
                "shared_drive_ids": [item for item in items if item in shared],
            },
        )


CONNECTOR = GoogleDriveConnector(DESCRIPTOR, settings=google_drive_settings)

__all__ = ["CONNECTOR", "DESCRIPTOR", "ROOT_RESOURCE_ID", "SCOPE", "GoogleDriveConnector"]
