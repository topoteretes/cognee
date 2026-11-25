"""The Gmail connector on the shared contract: ``CONNECTOR``.

Tier 2: the Google descriptor (:func:`cognee_connector_google.google_descriptor`) with
Gmail's read-only scope, on :class:`~cognee_connector_google.GoogleConnector` for the
userinfo lookup that keys the credential on the account's Google subject, plus what a
mailbox can enumerate: the label picker (``list_resources``: every label of the mailbox,
system and user-made) and the sync scope (``sync_scope``: the whole mailbox, or the labels
a host names). Gmail sends no webhooks on this path (push notifications would be a Pub/Sub
subscription, a different entry point), so the descriptor declares an hourly poll interval
for hosts that schedule syncs; cognee's own adapter syncs on the user's Refresh action only.
Nothing in this module imports cognee.
"""

from collections.abc import Mapping
from typing import Any

from cognee_connector_gmail import resources as gmail_resources
from cognee_connector_gmail.settings import google_gmail_settings
from cognee_connector_google import GoogleConnector, google_descriptor
from cognee_connectors_core.base import ConnectorResource, Credential, SyncScope

# gmail.readonly is a restricted scope and needs Google's OAuth verification before a
# public launch; openid and email are what identify the account the credential is keyed on.
SCOPE = "openid email https://www.googleapis.com/auth/gmail.readonly"

#: The one item of a scope that covers the whole mailbox, which the source syncs with
#: ``label_ids=None``. Named as the cloud's Gmail connector names it, so a runner reads
#: both the same way.
MAILBOX_RESOURCE_ID = "mailbox"

DESCRIPTOR = google_descriptor("gmail", "Gmail", SCOPE)


class GmailConnector(GoogleConnector):
    """Gmail on the contract: the Google OAuth flow plus the mailbox's labels."""

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """Every label of the mailbox, for a picker.

        The ids are what :meth:`sync_scope` takes back as ``resources``: Gmail label ids
        (``INBOX``, ``Label_12``), which the source passes to ``messages.list``.
        """
        token = self._bearer(credential, tokens)
        return [
            ConnectorResource(
                id=str(item["id"]),
                name=str(item["name"]),
                description=item.get("description"),
                attributes=dict(item.get("attributes") or {}),
            )
            for item in await gmail_resources.list_resources(token)
        ]

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """The mailbox, or the labels to sync, with the stored bearer token.

        ``resources`` names label ids; ``None`` (the connect path and a poll) is the whole
        mailbox, one item named :data:`MAILBOX_RESOURCE_ID` with ``label_ids`` left null
        for the source, the same scope the cloud's Gmail connector builds. A host that keeps
        a label selection narrows before calling, as the contract has it; cognee's own
        adapter stores one under ``selected_label_ids`` and its sync reads it there. An
        empty list is refused, as the contract requires: a delivery that implies no work is
        skipped by the caller, never passed through here.

        ``options["label_ids"]`` is what the source takes: the labels named, or ``None``
        for the mailbox. The items are the resource ids themselves, so
        ``item_resource_ids`` repeats them.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )
        token = self._bearer(credential, tokens)
        if resources is None:
            items = [MAILBOX_RESOURCE_ID]
            label_ids = None
        else:
            # One scope per label, however often the host named it.
            items = list(dict.fromkeys(str(label) for label in resources))
            label_ids = list(items)
        return SyncScope(
            dataset_name=self.dataset_name(credential),
            content_type=self.content_type,
            items=items,
            item_resource_ids=list(items),
            credential=token,
            credential_expires_at=credential.token_expires_at,
            credential_refreshable=bool(tokens.get("refresh_token")),
            options={"label_ids": label_ids},
        )


CONNECTOR = GmailConnector(DESCRIPTOR, settings=google_gmail_settings)

__all__ = ["CONNECTOR", "DESCRIPTOR", "MAILBOX_RESOURCE_ID", "SCOPE", "GmailConnector"]
