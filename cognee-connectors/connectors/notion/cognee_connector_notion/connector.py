"""The Notion connector on the shared contract: ``CONNECTOR``.

Tier 2: a descriptor for what is standard (the consent URL, and where the workspace
id, name and icon sit in the token response) plus the hooks Notion needs beyond it.
Notion's token endpoints authenticate the app with HTTP Basic and take a JSON body,
which the descriptor's form-encoded exchange cannot send (the ``token_auth_style``
limit :mod:`cognee_connectors_core.oauth2` names), so the code exchange, the refresh
grant, revocation and the token check a sync starts with go through
:mod:`cognee_connector_notion.client`, the same calls cognee's own adapter makes, and
never through ``HttpClient.post_form``. The token response names the workspace that
authorized, so no account lookup is needed and ``resolve_account`` stays untouched.

One authorization covers exactly one workspace, so a second workspace is a second
connection, keyed on the workspace id, and its dataset is named as cognee's adapter
names it (``notion_account_<10 hex of the workspace id>``,
:func:`~cognee_connector_notion.workspace.dataset_name_for_workspace`), so a server
and the cloud land the same workspace in the same dataset. Notion sends no webhooks on
this path, and nothing syncs until the user picks roots (pages or data sources): a
workspace cannot be walked whole, and ingesting one would cost LLM tokens the user
never asked for. Nothing in this module imports cognee.
"""

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from cognee_connector_notion import client
from cognee_connector_notion.settings import notion_settings
from cognee_connector_notion.workspace import (
    SELECTION_KEY,
    dataset_name_for_workspace,
    resource_id,
    split_root,
    without_nested_roots,
)
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

# How often a host without webhooks re-syncs a connection. Notion pushes nothing on
# this path, and cognee's own adapter syncs on the user's Refresh action only; the
# contract asks a webhook-less OAuth2 provider for a cadence, and an hour is cheap
# enough at Notion's rate limit for the selected trees to be walked again.
POLL_INTERVAL_SECONDS = 3600

DESCRIPTOR = ConnectorDescriptor(
    provider="notion",
    display_name="Notion",
    authorize_url=f"{client.API_BASE}/oauth/authorize",
    token_url=client.TOKEN_URL,
    # Notion has no scope parameter: capabilities are set on the integration itself.
    scopes="",
    authorize_params=(("response_type", "code"), ("owner", "user")),
    # Notion's token response names the workspace that authorized.
    account_id_path="workspace_id",
    account_label_path="workspace_name",
    metadata_paths=(("bot_id", "bot_id"), ("workspace_icon", "workspace_icon")),
    poll_interval_seconds=POLL_INTERVAL_SECONDS,
)


class NotionConnector(DescriptorConnector):
    """Notion on the contract: the descriptor plus the calls a descriptor cannot make.

    Tokens: Notion's docs describe a rotating refresh token and show no
    ``expires_in``. A response that carries no expiry is stored without one and used
    as is; one that carries an expiry gets ``token_expires_at`` for the host to
    refresh ahead of. Every refresh returns a new refresh token and the one just
    spent stops working, so the stored pair is replaced each time. Notion has a
    revoke endpoint, so a disconnect revokes both tokens.
    """

    # ---- Connect -------------------------------------------------------

    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Exchange the code at Notion's token endpoint (HTTP Basic, JSON body).

        ``params`` is ignored: Notion's callback carries nothing but ``code`` and the
        state the host has already checked. The answer, normalized by
        :func:`cognee_connector_notion.client.exchange_code`, names the workspace, so
        the descriptor's paths read it like any other token response.
        """
        return await client.exchange_code(
            code,
            client_id=self._setting("client_id"),
            client_secret=self._setting("client_secret"),
            redirect_uri=self._setting("redirect_uri"),
        )

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        installation = super().parse_installation(account_data)
        # A new connection must opt into a selection before importing data. A host
        # that merges metadata on reconnect (cognee's complete_installation) keeps
        # the existing choices.
        installation.provider_metadata[SELECTION_KEY] = []
        return installation

    # ---- Refresh and revoke --------------------------------------------

    async def refresh(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> RefreshedTokens | None:
        """Rotate the pair through Notion's refresh grant.

        ``None`` when the stored payload holds no refresh token: a workspace that
        connected before Notion issued them must reconnect. Notion rotates, so the
        answer carries the pair to keep and the host stores it; an answer without a
        new refresh token keeps the old one, which may still work, where storing
        nothing would lose it. A workspace that removed the integration at Notion
        answers ``invalid_grant``, raised as the :class:`~.client.NotionError` it is
        so the host can revoke the connection on it.
        """
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            return None
        refreshed = await client.refresh_access_token(
            str(refresh_token),
            client_id=self._setting("client_id"),
            client_secret=self._setting("client_secret"),
        )
        return RefreshedTokens(
            token_payload={
                **tokens,
                "access_token": refreshed["access_token"],
                "refresh_token": refreshed.get("refresh_token") or str(refresh_token),
            },
            token_expires_at=_expires_at(refreshed.get("expires_in")),
            scopes=credential.scopes,
        )

    async def revoke_remote(self, credential: Credential, tokens: Mapping[str, Any]) -> None:
        """Best-effort revoke of both tokens.

        The host's local revoke is the actual access cut-off, and a network blip must
        never block a disconnect, so one token's failure does not stop the other's
        revoke and nothing here raises.
        """
        for name in ("access_token", "refresh_token"):
            token = tokens.get(name)
            if not token:
                continue
            try:
                await client.revoke_token(
                    str(token),
                    client_id=self._setting("client_id"),
                    client_secret=self._setting("client_secret"),
                )
            except Exception:  # disconnect must not be blocked by Notion
                logger.exception(
                    "Notion %s revoke failed for workspace %s", name, credential.provider_account_id
                )

    # ---- Resources and sync --------------------------------------------

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """The roots a user can pick: shared pages and data sources whose parent is not shared.

        A page or data source inside another shared one is read by that one's walk, so
        offering it too would have the sync read it twice. Search only knows the parents
        it returns: a page inside a block (a toggle or a column) of a shared page is
        offered here, and the sync drops it when it walks the parent chain.
        """
        hits = await client.search_shared(self._bearer(credential, tokens))
        shared = {client.canonical_id(hit["id"]) for hit in hits}
        return [
            ConnectorResource(
                id=resource_id(hit["kind"], hit["id"]),
                name=hit["title"] or "Untitled",
                description=None,
                attributes={
                    "kind": hit["kind"],
                    "parent_kind": hit["parent_kind"],
                    "parent_id": hit["parent_id"],
                },
            )
            for hit in hits
            if not hit["parent_id"] or client.canonical_id(hit["parent_id"]) not in shared
        ]

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """The selected roots, one spelling each and none nested in another, plus the token.

        ``resources`` names the roots to sync; ``None`` (the connect path and a poll)
        reads the selection stored on the credential under ``selected_roots``, where
        no selection, or an empty one, is an empty scope rather than an error: a
        Notion workspace cannot be walked whole, so "everything" never means anything
        here and nothing syncs until roots are picked. An explicit empty ``resources``
        is refused as the base class refuses it: it can only be a no-work dispatch the
        caller should have skipped. A selection that is not a list of ``page:<id>`` or
        ``data_source:<id>`` entries raises ``ValueError`` before any request. A root
        nested in another selected root is dropped
        (:func:`~cognee_connector_notion.workspace.without_nested_roots`), which asks
        Notion for one parent per level of each root's chain.

        The stored token is checked first (``GET /users/me``,
        :func:`~cognee_connector_notion.client.verify_token`), on an empty selection
        too: a token stored with no expiry never goes through a refresh, so this is the
        only place a workspace that removed the integration at Notion shows up, and a
        host that polls gets its revoke signal from here alone. The
        :class:`~cognee_connector_notion.client.NotionError` propagates, and a host
        revokes the connection when ``error.code`` is in
        :data:`~cognee_connector_notion.client.UNAUTHORIZED_CODES`, the way it does on
        the ``invalid_grant`` :meth:`refresh` raises.

        ``options["workspace_id"]`` carries the workspace for the source's node sets,
        so a sync never has to look it up. The items are selection entries themselves,
        so ``item_resource_ids`` is not needed.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list: a dispatch "
                "that names no roots implies no work and must be skipped before calling "
                "sync_scope, not passed through here."
            )
        selected: Any = resources
        if selected is None:
            selected = (credential.provider_metadata or {}).get(SELECTION_KEY) or []
        if not isinstance(selected, list) or not all(isinstance(root, str) for root in selected):
            raise ValueError("Notion selection must be a list of page:<id> or data_source:<id>")
        # One spelling per root, so the same page picked twice is one source and one table.
        roots = list(dict.fromkeys(resource_id(*split_root(root)) for root in selected))
        token = self._bearer(credential, tokens)
        # The only place a workspace that removed the integration at Notion shows up for
        # a token stored with no expiry; raised as is, so the host revokes on its code.
        await client.verify_token(token)
        if roots:
            roots = await without_nested_roots(token, roots)
        return SyncScope(
            dataset_name=self.dataset_name(credential),
            content_type=self.content_type,
            items=roots,
            credential=token,
            credential_expires_at=credential.token_expires_at,
            credential_refreshable=bool(tokens.get("refresh_token")),
            options={"workspace_id": str(credential.provider_account_id)},
        )

    def dataset_name(self, credential: Credential) -> str:
        """One dataset per workspace, named as cognee's adapter names it.

        ``notion_account_<10 hex of the workspace id>``, not the core's
        ``dataset_name_for``: cognee's adapter keyed the dataset on a hash of the
        workspace id before this contract existed, and a server and the cloud must
        land the same workspace in the same dataset.
        """
        return dataset_name_for_workspace(str(credential.provider_account_id))

    def _bearer(self, credential: Credential, tokens: Mapping[str, Any]) -> str:
        """The stored access token, refusing a payload that holds none."""
        token = self._access_token(credential, tokens)
        if not token:
            raise RuntimeError(
                f"Notion credential for workspace {credential.provider_account_id} "
                "holds no access token"
            )
        return token


def _expires_at(expires_in: Any) -> datetime | None:
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in) if expires_in else None


CONNECTOR = NotionConnector(DESCRIPTOR, settings=notion_settings)

__all__ = ["CONNECTOR", "DESCRIPTOR", "POLL_INTERVAL_SECONDS", "NotionConnector"]
