"""Handle one verified GitHub webhook delivery.

Runs detached from the request (the generic events route acks first), so
handlers here may clone repositories and run pipelines. Every delivery
carries an ``installation.id`` that routes back to the connecting cognee
user via the credential store — an unknown or revoked installation is
logged and dropped, never an error: GitHub retries deliveries and gives no
ordering guarantee, so an ``installation.created`` racing ahead of the
OAuth callback's credential upsert is a normal, self-healing state (the
post-install hook covers the initial sync).

Deliberately narrow event coverage for the first cut:

* ``installation`` deleted/suspend -> revoke the stored credential.
* ``installation_repositories`` added -> index the new repos. Removed repos
  are logged only — deleting indexed data on a webhook would be a silent,
  destructive surprise; ``forget()`` stays a human decision.
* ``push`` to the default branch -> re-index that repo (the clone is a
  shallow copy of the default branch, so other refs change nothing we hold).
"""

import json
import logging

from cognee.modules.integrations.credentials import (
    STATUS_ACTIVE,
    get_credential_by_account,
    revoke_credential_by_account,
)
from cognee.modules.integrations.github.sync import sync_repositories

logger = logging.getLogger(__name__)

_PROVIDER = "github"


async def handle_github_event(raw_body: bytes, headers: dict[str, str]) -> None:
    event = headers.get("x-github-event", "")
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.warning("GitHub delivery with unparseable body (event=%s)", event)
        return

    installation_id = str((payload.get("installation") or {}).get("id") or "")
    if not installation_id:
        logger.info("GitHub %s delivery without installation id; ignoring", event)
        return

    action = payload.get("action")

    if event == "installation" and action in ("deleted", "suspend"):
        # The org removed (or suspended) the app on GitHub's side — the
        # local credential must stop minting tokens. Idempotent, so
        # redelivery is harmless.
        await revoke_credential_by_account(_PROVIDER, installation_id)
        logger.info("GitHub installation %s %s; credential revoked", installation_id, action)
        return

    credential = await get_credential_by_account(_PROVIDER, installation_id)
    if credential is None or credential.status != STATUS_ACTIVE:
        logger.info(
            "GitHub %s delivery for unconnected installation %s; ignoring",
            event,
            installation_id,
        )
        return

    if event == "installation_repositories":
        added = [repo["full_name"] for repo in payload.get("repositories_added", []) if repo]
        removed = [repo["full_name"] for repo in payload.get("repositories_removed", []) if repo]
        if removed:
            logger.info(
                "GitHub installation %s removed repos %s; indexed data retained "
                "(use forget() to drop it)",
                installation_id,
                removed,
            )
        if added:
            await sync_repositories(credential, added)
        return

    if event == "push":
        repository = payload.get("repository") or {}
        full_name = repository.get("full_name")
        default_branch = repository.get("default_branch")
        if not full_name or payload.get("ref") != f"refs/heads/{default_branch}":
            return
        await sync_repositories(credential, [full_name])
        return

    logger.debug("GitHub %s delivery for installation %s ignored", event, installation_id)
