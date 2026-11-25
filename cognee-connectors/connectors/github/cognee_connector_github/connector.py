"""GitHub as a Tier 3 connector on the shared contract: ``CONNECTOR``.

GitHub connects a whole org (or user account) through a GitHub App
installation rather than a plain OAuth app: the org admin installs the app
once, picks which repositories it covers, and the host stores only the
installation id — short-lived access tokens are minted on demand from the
app's private key (:mod:`cognee_connector_github.app_auth`), so the stored
credential carries an empty token payload. That is what makes it Tier 3: the
auth model is not OAuth2-with-a-bearer-token, so this subclasses
:class:`~cognee_connectors_core.base.Connector` directly instead of
describing itself as a descriptor. The shape follows the cloud-backend
connector of the same provider and the ``OAuthIntegration`` adapter the
cognee host keeps under ``cognee_host/``.

The install callback differs from the plain-OAuth shape in two ways, both
absorbed by :meth:`GithubConnector.exchange_callback`:

* GitHub's redirect carries an ``installation_id`` alongside the ``code``.
* That id arrives on an unauthenticated endpoint and is a small guessable
  integer, so it is never trusted directly: the ``code`` is first exchanged
  for a user token ("Request user authorization during installation" must
  be enabled on the app) and the user's access to the installation is
  confirmed, then the authoritative installation record is fetched with the
  app JWT. Without that check, anyone could bind another org's installation
  — and read access to its private repositories — to their own account.

Repositories are identified by their full name (``owner/repo``) everywhere
this connector talks about one — :attr:`WebhookDispatch.resources`, the
``resources`` given to :meth:`GithubConnector.sync_scope` and the
``item_resource_ids`` it returns, :attr:`ConnectorResource.id` — the same
handle cognee's own host code (``cognee_host/sync.py``) syncs by. A rename
on GitHub's side therefore changes a repository's resource id; a host that
stores a selection keys it on the name the repository was picked under.

``revoke_remote`` stays the default no-op on purpose: the GitHub-side
equivalent would be deleting the installation, i.e. uninstalling the app
from the whole org — too destructive for a cognee-side disconnect. Once the
local credential is revoked no further tokens are minted, which is the
actual access cut-off. ``refresh`` stays ``None`` for the same reason: there
is nothing stored to refresh.

Nothing in this module imports cognee.
"""

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import urlencode

from cognee_connector_github import app_auth
from cognee_connector_github.settings import github_settings
from cognee_connector_github.webhooks import github_webhook_verifier
from cognee_connectors_core.base import (
    Connector,
    ConnectorInstallation,
    ConnectorResource,
    Credential,
    SyncScope,
    WebhookDispatch,
    WebhookVerifier,
    dataset_name_for,
)
from cognee_connectors_core.http import AiohttpClient, HttpClient
from cognee_connectors_core.settings import require

logger = logging.getLogger(__name__)

_INSTALL_URL = "https://github.com/apps/{app_slug}/installations/new"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

# A GitHub installation id is a small sequential integer (eight digits at the
# time of writing). This is not a correctness bound — GitHub decides what is
# real and user_can_access_installation asks it — just a sanity ceiling that
# keeps the guard below, rather than int()'s own 4300-digit conversion limit,
# the thing that refuses an absurd value.
_MAX_INSTALLATION_ID_LENGTH = 20

# ``owner/repo``, the two path segments GitHub allows in a repository's full
# name: an owner is alphanumerics and single hyphens (never leading or
# trailing), a repository name alphanumerics, ``-``, ``_`` and ``.`` but never
# dots alone. A resource that does not look like one never becomes a clone
# URL: it arrives from a webhook payload or a host's stored selection, and
# the URL it would build is handed to git.
_FULL_NAME = re.compile(
    r"^[A-Za-z0-9](?:-?[A-Za-z0-9])*/(?![.]+$)[A-Za-z0-9_.-]+$",
)

# Events that mean the installation is gone on GitHub's side for good.
# ``suspend`` is deliberately not here: GitHub suspension is reversible — an
# org admin toggles it, and GitHub itself suspends installations for
# billing/abuse — and ``installation.unsuspend`` is a real, documented event.
# A suspended install mints no tokens and must not read as healthy, but
# treating it as deleted would revoke the credential with no way back except
# re-running the whole install flow; :attr:`WebhookDispatch.suspend` is the
# softer signal the contract has for it.
_REVOKE_ACTIONS = ("deleted",)
_SUSPEND_ACTIONS = ("suspend",)
_UNSUSPEND_ACTIONS = ("unsuspend",)


class GithubConnector(Connector):
    """GitHub App installations on the shared connector contract."""

    provider: ClassVar[str] = "github"
    display_name: ClassVar[str] = "GitHub"

    def __init__(self, settings: Any = None, http: HttpClient | None = None) -> None:
        # Settings are read at call time, never captured: a value changed on
        # the settings object takes effect on the next call. ``settings`` is
        # any object with the GithubSettings fields (the package's own
        # instance by default, a FakeSettings in tests).
        self.settings = github_settings if settings is None else settings
        self.http = http if http is not None else AiohttpClient()

    def _setting(self, field_name: str) -> str:
        return require(self.settings, field_name)

    # ---- Connect -------------------------------------------------------

    def authorize_url(self, state: str) -> str:
        """The app-installation URL, not an OAuth consent screen.

        GitHub passes ``state`` through to the app's configured callback URL
        along with the installation id (and a code, with
        request-user-authorization enabled).
        """
        slug = self._setting("app_slug")
        return f"{_INSTALL_URL.format(app_slug=slug)}?{urlencode({'state': state})}"

    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Verify the installation id, then fetch the authoritative record.

        The three steps are the confused-deputy mitigation described in the
        module docstring, and their order is load-bearing: nothing from the
        query string is used until the caller's access to it is proven.
        """
        installation_id = str(params.get("installation_id", "") or "")
        # isascii() as well as isdigit(): str.isdigit() is true for characters
        # int() cannot parse at all (superscripts, circled digits) and for
        # non-ASCII decimal digits it parses into a DIFFERENT number than the
        # string appears to name. Both are rejected here rather than
        # incidentally below. The length bound is the same idea: int()
        # refuses a string over 4300 digits, so without it a long-enough
        # numeric string also escapes this guard.
        if not (installation_id.isascii() and installation_id.isdigit()):
            raise ValueError("GitHub callback carried no installation_id")
        if len(installation_id) > _MAX_INSTALLATION_ID_LENGTH:
            raise ValueError("GitHub callback installation_id is implausibly long")

        user_token_response = await self._exchange_user_token(code)
        if not await app_auth.user_can_access_installation(
            user_token_response["access_token"], int(installation_id)
        ):
            raise PermissionError(
                f"GitHub user completing the install has no access to "
                f"installation {installation_id}"
            )

        # The authoritative record, fetched as the app — nothing from the
        # unauthenticated query string beyond the (now verified) id is used.
        return await app_auth.get_installation(int(installation_id), settings=self.settings)

    async def _exchange_user_token(self, code: str) -> dict[str, Any]:
        """Exchange the OAuth code for a *user* token.

        Only used to prove who completed the install flow — the user token
        is checked against the installation and discarded, never stored.
        GitHub returns errors as HTTP 200 with an ``error`` field, so the
        body, not the status, is what's checked.
        """
        status, payload = await self.http.post_form(
            _ACCESS_TOKEN_URL,
            {
                "client_id": self._setting("client_id"),
                "client_secret": self._setting("client_secret"),
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"GitHub code exchange returned a non-JSON body (HTTP {status})")
        if payload.get("error") or not payload.get("access_token"):
            raise RuntimeError(
                f"GitHub code exchange failed: {payload.get('error', 'no access_token')}"
            )
        return payload

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        installation_id = account_data.get("id")
        if installation_id is None:
            raise RuntimeError("GitHub installation record carries no id")

        account = account_data.get("account") or {}
        return ConnectorInstallation(
            # The installation id, which is also what every webhook carries —
            # so this is what routes a delivery back to its connection.
            provider_account_id=str(installation_id),
            # No durable per-installation secret exists — tokens are minted
            # on demand from the app private key (see module docstring).
            token_payload={},
            provider_metadata={
                "account_login": account.get("login"),
                "account_id": account.get("id"),
                "account_type": account.get("type"),
                "repository_selection": account_data.get("repository_selection"),
                "app_id": account_data.get("app_id"),
            },
            account_label=account.get("login"),
            auth_type="github_app",
        )

    # ---- Sync ----------------------------------------------------------

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """Mint an installation token and resolve the repositories to index.

        ``tokens`` is ignored: the stored payload is empty by design, and the
        credential a sync runs with is minted here from the app private key,
        which never leaves this process.

        ``resources`` (repository full names) is what makes an incremental
        sync incremental: a push names one repository, so enumeration is
        skipped entirely. Only the connect path (``None``) and an unfiltered
        re-sync list the whole installation. An empty list is refused, as the
        contract requires: a webhook that implies no work is skipped by the
        caller, never passed through here.

        The returned token carries its expiry so the runner can ask for a
        fresh one before it gets too close; ``credential_refreshable`` says
        the host may do so by calling this method again.
        """
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )

        installation_id = int(credential.provider_account_id)
        token, expires_at = await app_auth.mint_installation_token(
            installation_id, settings=self.settings
        )

        if resources is None:
            records = await app_auth.list_installation_repository_records(token)
            full_names = [str(record["full_name"]) for record in records if record]
        else:
            full_names = []
            for full_name in resources:
                if not _FULL_NAME.match(str(full_name)):
                    logger.warning(
                        "GitHub resource %r is not an owner/repo full name; skipping", full_name
                    )
                    continue
                full_names.append(str(full_name))

        return SyncScope(
            # One dataset per installation, not per repository: backend access
            # control isolates per dataset, so per-repo datasets would multiply
            # that cost without buying isolation anyone asked for. Keyed on
            # the installation id, the immutable identity on the credential —
            # an account login is display data GitHub lets an org rename.
            dataset_name=dataset_name_for(self.provider, credential.provider_account_id),
            content_type="code",
            items=[app_auth.clone_url(name) for name in full_names],
            # The full names, positionally matching ``items`` — what a picker
            # and a per-item outcome key on.
            item_resource_ids=list(full_names),
            credential=token,
            credential_expires_at=expires_at,
            credential_refreshable=True,
        )

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """The repositories this installation covers, for a picker.

        Deliberately not filtered by any stored selection: a picker needs the
        full set to draw unticked boxes next to. The id is the repository's
        full name — the same identity :meth:`sync_scope` filters on and
        stamps into ``item_resource_ids``.
        """
        installation_id = int(credential.provider_account_id)
        token, _expires_at = await app_auth.mint_installation_token(
            installation_id, settings=self.settings
        )
        repositories = await app_auth.list_installation_repository_records(token)

        return [
            ConnectorResource(
                id=str(repository["full_name"]),
                name=str(repository["full_name"]),
                description=repository.get("description"),
                attributes={
                    "id": repository.get("id"),
                    "private": repository.get("private"),
                    "archived": repository.get("archived"),
                    "default_branch": repository.get("default_branch"),
                    "language": repository.get("language"),
                },
            )
            for repository in repositories
            if repository and repository.get("full_name")
        ]

    def dataset_name(self, credential: Credential) -> str | None:
        return dataset_name_for(self.provider, credential.provider_account_id)

    @property
    def content_type(self) -> str | None:
        return "code"

    # ---- Webhooks ------------------------------------------------------

    def webhook_verifier(self) -> WebhookVerifier | None:
        """GitHub's ``X-Hub-Signature-256`` — the core ``sha256_prefix`` scheme.

        No GitHub-specific verification code: the HMAC check is written and
        tested once in ``cognee_connectors_core.signatures``.
        """
        return github_webhook_verifier(self.settings)

    def webhook_delivery_key(self, raw_body: bytes, headers: Mapping[str, str]) -> str | None:
        """The same digest ``handle_webhook`` reports, computable before it runs.

        GitHub's ``X-GitHub-Delivery`` header is not part of the HMAC input, so
        it cannot be a replay key: a captured, still-validly-signed body could
        be replayed under any header value. The body *is* authenticated by the
        time this is called, so its digest is a safe, deterministic key — and
        a pure function of the body, so a host can claim it before calling
        ``handle_webhook`` (whose ``unsuspend`` branch mints a token and lists
        the installation) and pay for that provider call once, not per replay.
        """
        return f"body-sha256:{hashlib.sha256(raw_body).hexdigest()}"

    async def handle_webhook(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookDispatch:
        """Reduce a verified delivery to an installation and the repositories to re-sync.

        Returns a decision rather than acting: the host owns resolving an
        installation id to the connections that hold it and running the sync
        (on cognee that is ``cognee_host/handle_github_event.py``).

        * ``installation`` deleted -> revoke the stored credential; suspend ->
          mark it suspended; unsuspend -> re-sync every repository.
        * ``installation_repositories`` added -> index the new repos. Removed
          repos are logged only — deleting indexed data on a webhook would be
          a silent, destructive surprise; ``forget()`` stays a human decision.
        * ``push`` to the default branch -> re-index that repo (a clone is a
          shallow copy of the default branch, so other refs change nothing).
        """
        lowered = {str(name).lower(): value for name, value in headers.items()}
        event = lowered.get("x-github-event", "")
        delivery_id = self.webhook_delivery_key(raw_body, lowered)
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("GitHub delivery with unparseable body (event=%s)", event)
            return WebhookDispatch(provider_account_id=None)
        if not isinstance(payload, dict):
            logger.warning("GitHub delivery whose body is not an object (event=%s)", event)
            return WebhookDispatch(provider_account_id=None)

        installation = payload.get("installation")
        installation_id = (
            str(installation.get("id") or "") if isinstance(installation, dict) else ""
        )
        if not installation_id:
            logger.info("GitHub %s delivery without installation id; ignoring", event)
            return WebhookDispatch(provider_account_id=None)

        action = payload.get("action")

        if event == "installation" and action in _REVOKE_ACTIONS:
            return WebhookDispatch(
                provider_account_id=installation_id, revoke=True, delivery_id=delivery_id
            )

        if event == "installation" and action in _SUSPEND_ACTIONS:
            return WebhookDispatch(
                provider_account_id=installation_id, suspend=True, delivery_id=delivery_id
            )

        if event == "installation" and action in _UNSUSPEND_ACTIONS:
            # No narrower signal than "the install works again": GitHub's
            # unsuspend payload names nothing that changed while paused, and
            # ``resources`` cannot mean "everything" (only sync_scope's
            # ``None`` can), so the installation is enumerated here and every
            # repository named explicitly.
            token, _expires_at = await app_auth.mint_installation_token(
                int(installation_id), settings=self.settings
            )
            records = await app_auth.list_installation_repository_records(token)
            return WebhookDispatch(
                provider_account_id=installation_id,
                resources=[str(record["full_name"]) for record in records if record],
                delivery_id=delivery_id,
            )

        if event == "installation_repositories":
            return WebhookDispatch(
                provider_account_id=installation_id,
                resources=self._added_repositories(payload, installation_id),
                delivery_id=delivery_id,
            )

        if event == "push":
            return WebhookDispatch(
                provider_account_id=installation_id,
                resources=self._pushed_repository(payload),
                delivery_id=delivery_id,
            )

        logger.debug("GitHub %s delivery for installation %s ignored", event, installation_id)
        return WebhookDispatch(provider_account_id=installation_id, delivery_id=delivery_id)

    @staticmethod
    def _added_repositories(payload: dict[str, Any], installation_id: str) -> list[str]:
        """Repositories added to an installation. Removals are logged, not acted on."""
        removed = [
            repo["full_name"]
            for repo in payload.get("repositories_removed") or []
            if repo and repo.get("full_name")
        ]
        if removed:
            logger.info(
                "GitHub installation %s removed repos %s; indexed data retained "
                "(use forget() to drop it)",
                installation_id,
                removed,
            )
        return [
            str(repo["full_name"])
            for repo in payload.get("repositories_added") or []
            if repo and repo.get("full_name")
        ]

    @staticmethod
    def _pushed_repository(payload: dict[str, Any]) -> list[str]:
        """The pushed repository, but only for a push to its default branch."""
        repository = payload.get("repository") or {}
        full_name = repository.get("full_name")
        default_branch = repository.get("default_branch")
        if not full_name or payload.get("ref") != f"refs/heads/{default_branch}":
            return []
        return [str(full_name)]


CONNECTOR = GithubConnector()

__all__ = ["CONNECTOR", "GithubConnector"]
