"""Handle one verified Linear webhook delivery.

Runs detached from the request (the generic events route acks first, well
inside Linear's 5-second response window), so handlers here may run searches
and pipelines. Every delivery's envelope carries an ``organizationId`` that
routes back to the connecting cognee user via the credential store — an
unknown or revoked organization is logged and dropped, never an error:
Linear retries deliveries and gives no ordering guarantee, so a delivery
racing ahead of the OAuth callback's credential upsert is a normal,
self-healing state (the post-install hook covers the initial sync).

Deliberately narrow event coverage for the first cut:

* ``AgentSessionEvent`` created/prompted -> answer the session from cognee
  memory (the agent loop, the point of this integration).
* ``Issue`` / ``Comment`` / ``Project`` create/update/remove -> sync the teams
  the delivery names through the DLT source, once the connection has finished
  a full sync. Linear must be set to send these categories as well as the
  agent session events.
* ``OAuthApp`` revoked -> revoke the stored credential; the workspace
  removed the app on Linear's side, so the local token must stop being used.

Everything else is logged and dropped. A ``remove`` delivery only triggers a
sync: deleting indexed data on a webhook would be a silent, destructive
surprise, and the source does not tombstone deletions yet.
"""

import json
import logging

from cognee.modules.integrations.credentials import (
    STATUS_ACTIVE,
    get_credential_by_account,
    revoke_credential_by_account,
)
from cognee.modules.integrations.linear.agent_session import handle_agent_session
from cognee.modules.integrations.linear.sync import (
    SEEDED_KEY,
    SELECTION_KEY,
    is_team_event,
    request_sync,
    team_ids_from_event,
)

logger = logging.getLogger(__name__)

_PROVIDER = "linear"


async def handle_linear_event(raw_body: bytes, headers: dict[str, str]) -> None:
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Linear delivery with unparseable body")
        return

    event_type = payload.get("type", "")
    action = payload.get("action", "")

    organization_id = str(payload.get("organizationId") or "")
    if not organization_id:
        logger.info("Linear %s delivery without organizationId; ignoring", event_type)
        return

    if event_type == "OAuthApp" and action == "revoked":
        # The workspace revoked the app on Linear's side — the local
        # credential must stop being used. Idempotent, so redelivery is
        # harmless.
        await revoke_credential_by_account(_PROVIDER, organization_id)
        logger.info("Linear app revoked for organization %s; credential revoked", organization_id)
        return

    credential = await get_credential_by_account(_PROVIDER, organization_id)
    if credential is None or credential.status != STATUS_ACTIVE:
        logger.info(
            "Linear %s delivery for unconnected organization %s; ignoring",
            event_type,
            organization_id,
        )
        return

    if event_type == "AgentSessionEvent" and action in ("created", "prompted"):
        # May be slow (search + LLM) — fine, we are already off the request
        # path, and the session handler acks Linear's own 10-second window
        # itself before doing that work.
        await handle_agent_session(credential, payload)
        return

    if is_team_event(payload):
        await _sync_teams_named_by(credential, payload)
        return

    logger.debug(
        "Linear %s/%s delivery for organization %s ignored", event_type, action, organization_id
    )


async def _sync_teams_named_by(credential, payload: dict) -> None:
    """Sync the teams an Issue, Comment or Project delivery is about.

    Skipped until the connection has finished a full pass, so an upgraded
    install never starts a backfill from a webhook, and for teams the user did
    not select. A delivery that arrives while a sync runs is queued.
    """
    metadata = credential.provider_metadata or {}
    if not metadata.get(SEEDED_KEY):
        logger.info(
            "Linear %s delivery for organization %s before its first full sync; ignoring",
            payload.get("type"),
            credential.provider_account_id,
        )
        return
    team_ids = team_ids_from_event(payload)
    selected = metadata.get(SELECTION_KEY)
    if selected is not None:
        team_ids = [team_id for team_id in team_ids if team_id in selected]
    if team_ids:
        await request_sync(credential, team_ids)
