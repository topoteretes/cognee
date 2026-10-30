"""The Google OAuth contract Drive, Gmail and Calendar build on, and ``CONNECTOR``.

Every Google product connects one Google account, personal or Workspace, through an
ordinary per-user consent at the same endpoints; the products differ in their scope and
in what they sync. :func:`google_descriptor` writes that shared OAuth shape down once and
:class:`GoogleConnector` adds the one thing a descriptor cannot express: Google's token
response carries no account identity, so the fresh access token is spent on a userinfo
call (:meth:`GoogleConnector.resolve_account`) and the descriptor's paths read the
account from the result (``userinfo.sub``, ``userinfo.email``, ``userinfo.hd``). That is
what makes the Google connectors Tier 2. The cognee host adapter does the same in
:mod:`cognee_connector_google.cognee_host.oauth_integration`; this module is the
host-agnostic contract, built on the core's form-encoded token client.

Two parameters on the authorize URL carry the design, as in cognee's adapter:

* ``access_type=offline`` with ``prompt=consent`` is what makes Google issue a refresh
  token. Without both, an account that has consented before gets an access token only,
  and the connector goes dark an hour after install with no way back short of a reconnect.
* ``include_granted_scopes`` is deliberately absent. Incremental authorization would fold
  one product's grant together with another's, and since Google's revoke endpoint kills a
  whole grant, disconnecting one product would silently disconnect the others.

``CONNECTOR`` is the base the three service connectors are derived from: provider
``google`` with the identity scopes only, built without settings. It is what this
package advertises as its contract object; a host that registers it gets a provider that
connects a Google account and syncs nothing. The service packages (``google_drive``,
``gmail``, ``google_calendar``) carry the connectors that sync data. Nothing in this
module imports cognee.
"""

import logging
from collections.abc import Mapping
from typing import Any

from cognee_connector_google.client import REVOKE_URL, TOKEN_URL, USERINFO_URL
from cognee_connectors_core.base import ConnectorInstallation, Credential
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"

#: ``openid email`` identify the account the credential is keyed on, so every Google
#: product asks for them ahead of its own scope.
IDENTITY_SCOPES = "openid email"

#: The fixed query parameters of every Google authorize URL (see the module docstring).
AUTHORIZE_PARAMS = (
    ("response_type", "code"),
    ("access_type", "offline"),
    ("prompt", "consent"),
)

#: Google sends no webhooks on this path, so a host that schedules syncs polls a
#: connection this often; cognee's own adapter syncs on the user's Refresh action only.
POLL_INTERVAL_SECONDS = 3600


def google_descriptor(
    provider: str,
    display_name: str,
    scopes: str = IDENTITY_SCOPES,
    *,
    poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
) -> ConnectorDescriptor:
    """A descriptor for one Google product: the shared OAuth shape plus its scope.

    ``scopes`` is the full space-separated string the product asks for, identity scopes
    included (``"openid email https://www.googleapis.com/auth/drive.readonly"``). The
    account paths read the ``userinfo`` object :meth:`GoogleConnector.resolve_account`
    attaches to the token response.
    """
    return ConnectorDescriptor(
        provider=provider,
        display_name=display_name,
        authorize_url=AUTHORIZE_URL,
        token_url=TOKEN_URL,
        scopes=scopes,
        authorize_params=AUTHORIZE_PARAMS,
        # The stable Google subject: an email can change, a subject cannot.
        account_id_path="userinfo.sub",
        account_label_path="userinfo.email",
        scopes_path="scope",
        # hd is present only for Workspace accounts, so its absence is what marks a
        # personal one (see GoogleConnector.parse_installation).
        metadata_paths=(
            ("email", "userinfo.email"),
            ("hosted_domain", "userinfo.hd"),
            ("scope", "scope"),
        ),
        poll_interval_seconds=poll_interval_seconds,
    )


class GoogleConnector(DescriptorConnector):
    """A Google product on the contract: the descriptor plus the userinfo lookup.

    Tier 2: :meth:`resolve_account` is the one hook the descriptor needs. Refresh goes
    through the descriptor's own refresh grant (Google returns no new refresh token, and
    the core keeps the stored one); revocation posts to Google's revoke endpoint, which
    kills the whole grant of this product's OAuth client and nothing else, since every
    product has its own client and its own consent.
    """

    async def resolve_account(self, token_response: dict[str, Any]) -> dict[str, Any] | None:
        """Attach the identity of who authorized, under ``userinfo``.

        The token response carries no account identity, but ``parse_installation`` is
        sync and has to derive a stable ``provider_account_id`` from this response alone.
        So the fresh token is spent here, in the async leg, on one userinfo call and the
        result rides along. A failure names the HTTP status only: the body of a failed
        call is never logged or raised.
        """
        access_token = token_response.get("access_token")
        if not access_token:
            raise RuntimeError(f"{self.provider} token exchange returned no access token")
        status, body = await self.http.get_json(
            USERINFO_URL, {"Authorization": f"Bearer {access_token}"}
        )
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"{self.provider} userinfo lookup failed: HTTP {status}")
        return {"userinfo": body}

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        """The descriptor's reading of the response, with the account type worked out.

        ``provider_metadata`` always carries the same four keys (``email``,
        ``hosted_domain``, ``account_type``, ``scope``), null when Google sent nothing,
        so a host that merges metadata on reconnect never keeps a stale value.
        """
        installation = super().parse_installation(account_data)
        metadata = installation.provider_metadata
        metadata.setdefault("email", None)
        metadata.setdefault("scope", None)
        hosted_domain = metadata.setdefault("hosted_domain", None)
        metadata["account_type"] = "workspace" if hosted_domain else "personal"
        if "refresh_token" not in installation.token_payload:
            # Recoverable only by reconnecting: Google withholds the refresh token when
            # a prior consent is reused, which is what prompt=consent exists to prevent.
            # Better named here than as a 401 an hour from now.
            logger.warning(
                "%s account %s connected without a refresh token; "
                "access will stop working once the access token expires",
                self.display_name,
                installation.provider_account_id,
            )
        return installation

    async def revoke_remote(self, credential: Credential, tokens: Mapping[str, Any]) -> None:
        """Best-effort revoke of this account's grant.

        The grant is kept separate per product (see the module docstring), so this
        cannot take a sibling Google connector down with it. Still best-effort: the
        local revoke is the actual access cut-off and a network blip must never block
        a disconnect, so nothing here raises.
        """
        token = tokens.get("refresh_token") or tokens.get("access_token")
        if not token:
            return
        try:
            status, _body = await self.http.post_form(REVOKE_URL, {"token": str(token)})
        except Exception:  # disconnect must not be blocked by Google
            logger.exception(
                "%s token revoke failed for account %s",
                self.display_name,
                credential.provider_account_id,
            )
            return
        if status != 200:
            logger.warning("%s token revoke returned HTTP %s", self.display_name, status)

    def _bearer(self, credential: Credential, tokens: Mapping[str, Any]) -> str:
        """The stored access token, refusing a payload that holds none.

        The host refreshes ahead of a sync (:meth:`refresh`); an expired token is handed
        out with a warning, as the core does, never silently.
        """
        token = self._access_token(credential, tokens)
        if not token:
            raise RuntimeError(
                f"{self.display_name} credential for account {credential.provider_account_id} "
                "holds no access token"
            )
        return token


#: The base every Google product descriptor is derived from: Google's endpoints with the
#: identity scopes only.
DESCRIPTOR = google_descriptor("google", "Google")

#: The identity-only base connector, built without settings: it connects a Google account
#: and syncs nothing. A service package builds its own with its settings, for example
#: ``GoogleDriveConnector(google_descriptor("google_drive", "Google Drive", scope), settings)``.
CONNECTOR = GoogleConnector(DESCRIPTOR)

__all__ = [
    "AUTHORIZE_PARAMS",
    "AUTHORIZE_URL",
    "CONNECTOR",
    "DESCRIPTOR",
    "IDENTITY_SCOPES",
    "POLL_INTERVAL_SECONDS",
    "GoogleConnector",
    "google_descriptor",
]
