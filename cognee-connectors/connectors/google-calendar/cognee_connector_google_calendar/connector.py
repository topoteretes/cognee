"""The Google Calendar connector on the shared contract: ``CONNECTOR``.

Tier 2: the Google descriptor (:func:`cognee_connector_google.google_descriptor`) with
Calendar's read-only scope, on :class:`~cognee_connector_google.GoogleConnector` for the
userinfo lookup that keys the credential on the account's Google subject, plus what an
account can enumerate: the calendar picker (``list_resources``: every calendar the
account can see, hidden ones included) and the sync scope (``sync_scope``: the calendars
to sync, always named explicitly). Google Calendar sends no webhooks on this path (push
channels are a different entry point), so the descriptor declares an hourly poll interval
for hosts that schedule syncs; cognee's own adapter syncs on the user's Refresh action
only. Nothing in this module imports cognee.
"""

from collections.abc import Mapping
from typing import Any

from cognee_connector_google import GoogleConnector, google_descriptor
from cognee_connector_google_calendar import resources as calendar_resources
from cognee_connector_google_calendar.settings import google_calendar_settings
from cognee_connectors_core.base import ConnectorResource, Credential, SyncScope

# calendar.readonly is a sensitive scope and needs Google's OAuth verification before a
# public launch; openid and email are what identify the account the credential is keyed on.
SCOPE = "openid email https://www.googleapis.com/auth/calendar.readonly"

DESCRIPTOR = google_descriptor("google_calendar", "Google Calendar", SCOPE)


class GoogleCalendarConnector(GoogleConnector):
    """Calendar on the contract: the Google OAuth flow plus the account's calendars."""

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """Every calendar the account can see, for a picker.

        The ids are what :meth:`sync_scope` takes back as ``resources``: calendar ids
        (``primary`` resolves to the account's own calendar, the rest are usually email
        addresses), which the source passes to ``events.list``.
        """
        token = self._bearer(credential, tokens)
        return [
            ConnectorResource(
                id=str(item["id"]),
                name=str(item["name"]),
                description=item.get("description"),
                attributes=dict(item.get("attributes") or {}),
            )
            for item in await calendar_resources.list_resources(token)
        ]

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """The calendars to sync, always named explicitly, with the stored bearer token.

        ``resources`` names calendar ids; ``None`` (the connect path and a poll) is every
        calendar the account can see, listed here and named one by one. Never the source's
        own default: it reads ``calendar_ids=None`` as the primary calendar or whatever
        ``GOOGLE_CALENDAR_IDS`` holds in the process it runs in, which is no answer for a
        hosted account. Events carry attendees and meeting details, so a host that keeps
        calendars opt-in narrows before calling, as the contract has it: cognee's own
        adapter refuses a null selection outright (``resource_selection_allows_all`` is
        false) and its sync reads the chosen calendars from ``selected_calendar_ids``. An
        empty list is refused, as the contract requires: a delivery that implies no work is
        skipped by the caller, never passed through here.

        ``options["calendar_ids"]`` is what the source takes. The items are the resource
        ids themselves, so ``item_resource_ids`` repeats them.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )
        token = self._bearer(credential, tokens)
        if resources is None:
            items = await calendar_resources.calendar_ids(token)
        else:
            # One scope per calendar, however often the host named it.
            items = list(dict.fromkeys(str(calendar) for calendar in resources))
        return SyncScope(
            dataset_name=self.dataset_name(credential),
            content_type=self.content_type,
            items=items,
            item_resource_ids=list(items),
            credential=token,
            credential_expires_at=credential.token_expires_at,
            credential_refreshable=bool(tokens.get("refresh_token")),
            options={"calendar_ids": list(items)},
        )


CONNECTOR = GoogleCalendarConnector(DESCRIPTOR, settings=google_calendar_settings)

__all__ = ["CONNECTOR", "DESCRIPTOR", "SCOPE", "GoogleCalendarConnector"]
