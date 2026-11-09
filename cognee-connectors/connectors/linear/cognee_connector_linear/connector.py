"""The Linear connector on the shared contract: ``CONNECTOR``.

Tier 2, derived from the cognee adapter (``cognee_host/adapter.py``): a
:class:`ConnectorDescriptor` for the agent-app OAuth flow plus the one hook Linear needs.
Its token response says nothing about *which* workspace authorized the app, while every
webhook delivery routes by its ``organizationId`` envelope field, so
:meth:`LinearConnector.resolve_account` spends the fresh token on one GraphQL query for
``viewer`` (the app user) and ``organization``, and the descriptor's paths read the merged
result. The endpoints, scopes and that query are defined here once; the cognee adapter
imports them and keeps cognee's own refresh, revoke and sync machinery around them.

Beyond that hook the connector overrides what a descriptor cannot say about Linear:

* a delivery is a bare hex HMAC plus a ``webhookTimestamp`` replay window inside the signed
  body (:mod:`cognee_connector_linear.webhooks`), and the event type sits in the body, not
  in a header, so :meth:`LinearConnector.handle_webhook` reads it there;
* a sync is per team: :meth:`LinearConnector.sync_scope` and
  :meth:`LinearConnector.list_resources` enumerate the teams the app was granted, and the
  dataset is named after the workspace's url key, as cognee's is;
* :meth:`LinearConnector.revoke_remote` kills the workspace's tokens at Linear, which is
  cheap and non-destructive there.

Nothing in this module imports cognee.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from cognee_connector_linear import client as linear_client
from cognee_connector_linear.settings import linear_settings
from cognee_connector_linear.webhooks import SIGNATURE_HEADER, LinearWebhookVerifier
from cognee_connector_linear.workspace import (
    TEAMS_QUERY,
    dataset_name_for_credential,
    describe_team,
    is_team_event,
    list_teams_with,
    selected_team_ids,
    team_ids_from_event,
)
from cognee_connectors_core.base import (
    ConnectorResource,
    Credential,
    SyncScope,
    WebhookDispatch,
    WebhookVerifier,
)
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.http import HttpClient
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.signatures import SignatureScheme

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://linear.app/oauth/authorize"
TOKEN_URL = "https://api.linear.app/oauth/token"
REVOKE_URL = "https://api.linear.app/oauth/revoke"

# app:assignable lets members delegate issues to the agent; app:mentionable
# lets them @mention it. Both require the actor=app install (and actor=app
# cannot be combined with the admin scope).
SCOPES = "read,write,app:assignable,app:mentionable"

# viewer is the freshly installed app user (the agent identity in that
# workspace); organization.id is what every webhook envelope routes by.
INSTALL_CONTEXT_QUERY = """
query InstallContext {
  viewer { id name }
  organization { id name urlKey }
}
"""

#: A GraphQL call as :func:`cognee_connector_linear.client.graphql` makes one:
#: ``(access_token, query, variables=None) -> data``.
GraphQL = Callable[..., Awaitable[dict[str, Any]]]

DESCRIPTOR = ConnectorDescriptor(
    provider="linear",
    display_name="Linear",
    authorize_url=AUTHORIZE_URL,
    token_url=TOKEN_URL,
    scopes=SCOPES,
    # The agent install: actor=app puts an app user into the workspace instead
    # of acting as the authorizing human.
    authorize_params=(("response_type", "code"), ("actor", "app")),
    # Read off the token response once resolve_account() has merged the install
    # context into it: the organization id is what every webhook routes by.
    account_id_path="organization.id",
    account_label_path="organization.name",
    scopes_path="scope",
    metadata_paths=(
        ("app_user_id", "viewer.id"),
        ("app_user_name", "viewer.name"),
        ("organization_name", "organization.name"),
        ("organization_url_key", "organization.urlKey"),
        ("scope", "scope"),
    ),
    signature_scheme=SignatureScheme.BARE_HEX,
    signature_header=SIGNATURE_HEADER,
    webhook_account_id_path="organizationId",
    # No webhook_delivery_id_path on purpose: the body's ``webhookId`` is the id of
    # the webhook itself, the same on every delivery it sends, and nothing else in
    # the body identifies one delivery. Replay protection is the ``webhookTimestamp``
    # window the verifier enforces, and the handling is idempotent.
)


class LinearConnector(DescriptorConnector):
    """Linear on the shared contract: the descriptor plus the hooks the module describes.

    ``graphql`` is the coroutine function the install-context and teams queries go
    through, ``(access_token, query, variables=None) -> data``; tests inject a recording
    one, hosts take :func:`cognee_connector_linear.client.graphql`. The token endpoint is
    reached through the core's ``http`` client like any other descriptor's.
    """

    def __init__(
        self,
        descriptor: ConnectorDescriptor = DESCRIPTOR,
        settings: Any = None,
        http: HttpClient | None = None,
        *,
        graphql: GraphQL | None = None,
    ) -> None:
        super().__init__(descriptor, settings, http)
        self._graphql_override = graphql

    async def _graphql(
        self, access_token: str, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Resolved per call, so a monkeypatched ``client.graphql`` is seen as well.
        graphql = self._graphql_override or linear_client.graphql
        return await graphql(access_token, query, variables)

    # ---- Connect -------------------------------------------------------

    async def resolve_account(self, token_response: dict[str, Any]) -> dict[str, Any] | None:
        """Spend the fresh token on the install context the token response omits.

        The raw token response carries no workspace or app-user identity, but the
        credential must be keyed on the organization id, because webhooks route back
        by their ``organizationId`` envelope field. So the token is spent here, once,
        and ``viewer`` and ``organization`` ride along for the descriptor's paths.
        """
        access_token = token_response.get("access_token")
        if not access_token:
            raise RuntimeError("Linear code exchange returned no access_token")
        context = await self._graphql(str(access_token), INSTALL_CONTEXT_QUERY)
        return {"viewer": context.get("viewer"), "organization": context.get("organization")}

    # ---- Webhooks ------------------------------------------------------

    def webhook_verifier(self) -> WebhookVerifier | None:
        # The descriptor's bare-hex HMAC plus Linear's replay window; the secret is
        # read off the settings per delivery, as the base class reads it.
        return LinearWebhookVerifier(secret_provider=self._webhook_secret)

    async def handle_webhook(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookDispatch:
        """Reduce a verified delivery to the organization and the teams it names.

        Linear puts the event type and action in the body, so the descriptor's
        header-driven event lists cannot express them: ``OAuthApp``/``revoked`` revokes,
        an ``Issue``, ``Comment`` or ``Project`` create/update/remove names the teams to
        re-sync, and everything else — agent session events included — is understood but
        implies no sync. A host answers those sessions itself; cognee's does in
        ``cognee_host/handle_linear_event.py``.
        """
        dispatch = await super().handle_webhook(raw_body, headers)
        if dispatch.provider_account_id is None:
            return dispatch
        payload = json.loads(raw_body)  # decoded by the base class already, so it parses
        if payload.get("type") == "OAuthApp" and payload.get("action") == "revoked":
            dispatch.revoke = True
        elif is_team_event(payload):
            dispatch.resources = team_ids_from_event(payload)
        return dispatch

    # ---- Sync ----------------------------------------------------------

    def _stored_access_token(self, credential: Credential, tokens: Mapping[str, Any]) -> str:
        access_token = self._access_token(credential, tokens)
        if not access_token:
            raise RuntimeError(
                f"Linear credential for organization {credential.provider_account_id} "
                "holds no access token"
            )
        return access_token

    async def _teams(self, access_token: str) -> list[dict[str, Any]]:
        return await list_teams_with(
            lambda variables: self._graphql(access_token, TEAMS_QUERY, variables)
        )

    async def _team_ids(self, access_token: str) -> list[str]:
        return [str(team["id"]) for team in await self._teams(access_token)]

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """The teams to ingest, one source each, with the stored bearer token.

        ``None`` (the connect path) means the stored selection — ``selected_team_ids`` in
        the credential's metadata, absent meaning every team the app was granted, which is
        listed from Linear. A webhook's ``resources`` are narrowed the way cognee's event
        handler narrows them: to the selection, or to the granted teams when there is
        none, so a delivery naming a team the app never had cannot fail the sync. ``items``
        is therefore always the explicit list of team ids, and an empty list means there is
        nothing to ingest: the source is per team, there is no "everything" call for it to
        collapse to.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )
        access_token = self._stored_access_token(credential, tokens)
        selected = selected_team_ids(credential)
        if resources is None:
            teams = selected if selected is not None else await self._team_ids(access_token)
        elif selected is not None:
            teams = [team_id for team_id in resources if team_id in selected]
        else:
            granted = set(await self._team_ids(access_token))
            teams = [team_id for team_id in resources if team_id in granted]
        return SyncScope(
            dataset_name=dataset_name_for_credential(credential),
            content_type=self.descriptor.content_type,
            items=list(dict.fromkeys(teams)),
            credential=access_token,
            # Linear's tokens last 24 hours and refresh() rotates them, so the runner
            # knows when to ask for a replacement.
            credential_expires_at=credential.token_expires_at,
            credential_refreshable=bool(tokens.get("refresh_token")),
        )

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """The teams the app was granted, for the picker."""
        access_token = self._stored_access_token(credential, tokens)
        return [
            ConnectorResource(**describe_team(team)) for team in await self._teams(access_token)
        ]

    def dataset_name(self, credential: Credential) -> str:
        """``linear_<url key>``: one dataset per workspace, named as cognee names it.

        Not the core's ``dataset_name_for``: cognee's adapter keyed the dataset on the
        workspace's url key before this contract existed, and a server and the cloud
        must land the same workspace in the same dataset.
        """
        return dataset_name_for_credential(credential)

    # ---- Disconnect ----------------------------------------------------

    async def revoke_remote(self, credential: Credential, tokens: Mapping[str, Any]) -> None:
        """Best-effort remote revoke of the workspace's agent token.

        Cheap and non-destructive on Linear's side (it kills only this
        token, not the app install), so worth doing — but still best-effort:
        the local revoke is the actual access cut-off, and a network blip
        here must never block a disconnect. Linear documents the ``token``
        field as the form to use, but not whether revoking the refresh token
        also ends the access token, so both are revoked, refresh token first.
        """
        for token_type in ("refresh_token", "access_token"):
            token = tokens.get(token_type)
            if not token:
                continue
            try:
                status, _body = await self.http.post_form(
                    REVOKE_URL, {"token": str(token), "token_type_hint": token_type}
                )
            except Exception:  # one failed revoke must not stop the other or the disconnect
                logger.exception(
                    "Linear %s revoke for organization %s failed",
                    token_type,
                    credential.provider_account_id,
                )
                continue
            if status != 200:
                logger.warning(
                    "Linear %s revoke for organization %s failed: HTTP %s",
                    token_type,
                    credential.provider_account_id,
                    status,
                )


CONNECTOR = LinearConnector(settings=linear_settings)

__all__ = [
    "AUTHORIZE_URL",
    "CONNECTOR",
    "DESCRIPTOR",
    "INSTALL_CONTEXT_QUERY",
    "REVOKE_URL",
    "SCOPES",
    "TOKEN_URL",
    "LinearConnector",
]
