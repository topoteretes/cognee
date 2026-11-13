"""The Dropbox connector on the shared contract: ``CONNECTOR``.

Tier 2, mirroring cognee's ``cognee.modules.integrations.dropbox.adapter``. The token response
names the account (``account_id``, ``dbid:...``), but the label and the dataset name come from
the account's email, which takes one more call (``users/get_current_account``). Dropbox's
token endpoint is called the way cognee's adapter calls it, with the app key and secret as
HTTP Basic, through :mod:`cognee_connector_dropbox.client`, so the code exchange, the refresh
grant, revocation, the folder picker and the nesting check of a sync never go through the
core's ``HttpClient``.

One authorization is one Dropbox account. ``token_access_type=offline`` makes Dropbox issue a
refresh token next to the four-hour access token; the refresh token does not expire or
rotate, and revoking the access token revokes both. Nothing syncs until the user picks
folders: a whole Dropbox costs LLM tokens. Dropbox sends no webhooks on this path. Nothing in
this module imports cognee.
"""

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from cognee_connector_dropbox import client
from cognee_connector_dropbox.selection import (
    SELECTION_KEY,
    check_selection_shape,
    dataset_name_for_account,
    refuse_nested_selection,
)
from cognee_connector_dropbox.settings import dropbox_settings
from cognee_connectors_core.base import (
    ConnectorInstallation,
    ConnectorResource,
    Credential,
    RefreshedTokens,
    SyncScope,
)
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

logger = logging.getLogger(__name__)

SCOPES = "files.metadata.read files.content.read account_info.read"
# How often a host without webhooks re-syncs a connection. Dropbox pushes nothing on this
# path; cognee syncs on Sync now or on a sync policy. The contract asks a webhook-less OAuth2
# provider for a cadence. A run that finds nothing new costs three Dropbox calls per folder at
# the source (users/get_current_account, files/get_metadata, list_folder/continue, through one
# HTTP client per folder), plus one files/get_metadata per folder in sync_scope's nesting check
# when two or more folders are selected.
POLL_INTERVAL_SECONDS = 3600
# How long before its expiry a stored access token is renewed before a revoke: cognee's
# ``credentials.TOKEN_REFRESH_MARGIN``.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)

DESCRIPTOR = ConnectorDescriptor(
    provider="dropbox",
    display_name="Dropbox",
    auth="oauth2",
    authorize_url="https://www.dropbox.com/oauth2/authorize",
    token_url=client.TOKEN_URL,
    scopes=SCOPES,
    authorize_params=(("response_type", "code"), ("token_access_type", "offline")),
    account_id_path="account_id",
    # Merged in by exchange_callback from the account lookup.
    account_label_path="email",
    scopes_path="scope",
    metadata_paths=(("email", "email"),),
    poll_interval_seconds=POLL_INTERVAL_SECONDS,
)


class DropboxConnector(DescriptorConnector):
    """Dropbox on the contract: the descriptor plus the calls a descriptor cannot make."""

    # ---- Connect -------------------------------------------------------

    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Exchange the code, then read the account's email for the label and dataset name.

        A token whose ``account_id`` is not the account the lookup answers for is refused with
        ``ValueError``: the connection would be keyed on one account and read another.
        """
        token_response = await client.exchange_code(
            code,
            client_id=self._setting("client_id"),
            client_secret=self._setting("client_secret"),
            redirect_uri=self._setting("redirect_uri"),
        )
        account = await client.get_current_account(token_response["access_token"])
        if account.get("account_id") != token_response["account_id"]:
            raise ValueError("Dropbox token and account lookup name different accounts")
        return {**token_response, "email": account.get("email")}

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        account_id = account_data.get("account_id")
        if not isinstance(account_id, str) or not account_id.startswith("dbid:"):
            raise ValueError("Dropbox token response carries no account_id")
        installation = super().parse_installation(account_data)
        # cognee's adapter reads only a positive integer lifetime; anything else is no expiry.
        installation.token_expires_at = _expires_at(account_data.get("expires_in"))
        # A new connection must opt into a selection before importing data. A host that
        # merges metadata on reconnect (cognee's complete_installation) keeps the choice.
        installation.provider_metadata[SELECTION_KEY] = []
        return installation

    # ---- Refresh and revoke --------------------------------------------

    async def refresh(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> RefreshedTokens | None:
        """Renew the access token; Dropbox does not rotate the refresh token, so it is kept.

        ``None`` when the payload holds no refresh token (the account must reconnect). An
        account that removed the app answers ``invalid_grant``, raised as the
        :class:`~.client.DropboxIntegrationError` it is so the host revokes the connection on
        it, as cognee's adapter does. An answer without ``expires_in`` is refused
        (``no_expires_in``): stored without an expiry, the token would never be renewed.
        """
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            return None
        refreshed = await client.refresh_access_token(
            str(refresh_token),
            client_id=self._setting("client_id"),
            client_secret=self._setting("client_secret"),
        )
        expires_at = _expires_at(refreshed.get("expires_in"))
        if expires_at is None:
            raise client.DropboxIntegrationError("token refresh", "no_expires_in")
        return RefreshedTokens(
            token_payload={
                "access_token": refreshed["access_token"],
                "refresh_token": str(refresh_token),
            },
            token_expires_at=expires_at,
            scopes=refreshed.get("scope") or credential.scopes,
        )

    async def revoke_remote(self, credential: Credential, tokens: Mapping[str, Any]) -> None:
        """Best-effort revoke; the host's local revoke is the actual cut-off.

        An access token past or near its expiry cannot revoke itself, so a fresh one is taken
        from the refresh token, in memory only, and revoking it revokes the refresh token with
        it. A Dropbox failure is logged by its code and never raised; a missing app key or
        secret is a configuration error and raises, as in cognee's adapter.
        """
        token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        try:
            if refresh_token and (not token or _is_expiring(credential.token_expires_at)):
                refreshed = await client.refresh_access_token(
                    str(refresh_token),
                    client_id=self._setting("client_id"),
                    client_secret=self._setting("client_secret"),
                )
                token = refreshed["access_token"]
            if token:
                await client.revoke_token(str(token))
        except client.DropboxIntegrationError as error:
            # The app stays listed in the user's Dropbox settings until they remove it.
            logger.warning(
                "Dropbox token revoke failed for account %s: %s",
                credential.provider_account_id,
                error.code,
            )

    # ---- Resources and sync --------------------------------------------

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """The folders at the top of the account, by id, with their path, for a picker.

        One level, as cognee's picker starts: an account can hold too many folders to list at
        once. The contract's ``list_resources`` takes no parent, so the deeper levels cognee's
        picker opens (``client.list_folders(token, parent=folder_id)``) are not offered here.
        """
        folders = await client.list_folders(self._bearer(credential, tokens))
        return [
            ConnectorResource(
                id=folder["id"],
                name=folder.get("path_display") or folder.get("name") or "Untitled folder",
                description=None,
                attributes={"path": folder.get("path_display")},
            )
            for folder in sorted(folders, key=lambda f: str(f.get("path_lower")))
            if isinstance(folder.get("id"), str)
        ]

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """The selected folders, one scope each, plus the token and the source's options.

        ``resources`` names folder ids; ``None`` (the connect path and a poll) reads the
        selection stored under ``selected_folder_ids``, where an empty selection is an empty
        scope: nothing syncs until folders are picked. An explicit empty ``resources`` is
        refused, as the contract requires. A selection that is not a list of ``id:...`` raises
        ``ValueError`` before any request; a folder inside another selected folder raises
        :class:`~.selection.NestedSelectionError` (one ``get_metadata`` per folder when two or
        more are selected), as cognee's sync does. With one folder nothing is asked here, so a
        token Dropbox refuses reaches the source; with two or more the lookup raises it.
        ``options`` carries what ``dropbox_source`` needs besides the folder and the token:
        ``account_id`` and ``max_file_size_mb``.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list: a dispatch "
                "that names no folders implies no work and must be skipped before calling "
                "sync_scope, not passed through here."
            )
        selected: Any = resources
        if selected is None:
            selected = (credential.provider_metadata or {}).get(SELECTION_KEY) or []
        folder_ids = check_selection_shape(selected)
        token = self._bearer(credential, tokens)
        await refuse_nested_selection(token, folder_ids)
        return SyncScope(
            dataset_name=self.dataset_name(credential),
            content_type=self.content_type,
            items=folder_ids,
            credential=token,
            credential_expires_at=credential.token_expires_at,
            credential_refreshable=bool(tokens.get("refresh_token")),
            options={
                "account_id": str(credential.provider_account_id),
                # The settings object a host built the connector with, read at call time.
                "max_file_size_mb": self.settings.max_file_size_mb,
            },
        )

    def dataset_name(self, credential: Credential) -> str:
        """One dataset per account, named as cognee's adapter names it.

        ``dropbox_<email slug>_<10 hex of the account id>``
        (:func:`~.selection.dataset_name_for_account`), not the core's ``dataset_name_for``,
        so a cognee server and any other host land the same account in the same dataset.
        """
        email = (credential.provider_metadata or {}).get("email") or ""
        return dataset_name_for_account(str(email), str(credential.provider_account_id))

    def _bearer(self, credential: Credential, tokens: Mapping[str, Any]) -> str:
        """The stored access token, refusing a payload that holds none."""
        token = self._access_token(credential, tokens)
        if not token:
            raise RuntimeError(
                f"Dropbox credential for account {credential.provider_account_id} "
                "holds no access token"
            )
        return token


def _expires_at(expires_in: Any) -> datetime | None:
    if not isinstance(expires_in, int) or isinstance(expires_in, bool) or expires_in <= 0:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in)


def _is_expiring(expires_at: datetime | None) -> bool:
    """True within ``TOKEN_REFRESH_MARGIN`` of the expiry; ``None`` never expires. A naive
    timestamp is UTC (SQLite stores no timezone)."""
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at - TOKEN_REFRESH_MARGIN <= datetime.now(timezone.utc)


CONNECTOR = DropboxConnector(DESCRIPTOR, settings=dropbox_settings)

__all__ = ["CONNECTOR", "DESCRIPTOR", "POLL_INTERVAL_SECONDS", "SCOPES", "DropboxConnector"]
