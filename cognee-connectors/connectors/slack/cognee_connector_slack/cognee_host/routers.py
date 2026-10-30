"""Slack HTTP routes for a cognee server.

Moved from cognee.api.v1.slack.routers (get_slack_router, get_slack_channels_router,
get_slack_history_router and the package ``__init__``; cognee dev, 2026-10-27) into
one module. :func:`routers` lists what a host mounts, with the prefix and tags
cognee's ``api/client.py`` used; each factory keeps its original module docstring as
its own.
"""

import logging
from typing import Annotated

from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    get_active_credential_for_user,
    update_provider_metadata,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from fastapi import APIRouter, Depends, HTTPException

from cognee_connector_slack.channels import list_channels
from cognee_connector_slack.cognee_host.handle_slack_command import handle_slack_command
from cognee_connector_slack.cognee_host.handle_slack_event import handle_slack_event
from cognee_connector_slack.cognee_host.handle_slack_interactive import handle_slack_interactive
from cognee_connector_slack.cognee_host.handle_slack_link import confirm_link
from cognee_connector_slack.cognee_host.history_sync import (
    configure_slack_sync,
    run_history_import,
    slack_history_lifespan,
)
from cognee_connector_slack.cognee_host.persistence import PROVIDER, get_by_team, is_active
from cognee_connector_slack.cognee_host.verify_slack_signature import verify_slack_request
from cognee_connector_slack.history_models import (
    SlackHistoryError,
    SlackHistoryRequest,
    SlackHistoryResult,
    SlackSyncSettings,
)
from cognee_connector_slack.settings import slack_settings

logger = logging.getLogger(__name__)

__all__ = ["get_slack_channels_router", "get_slack_history_router", "get_slack_router", "routers"]


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags).

    The same two routers cognee's ``api/client.py`` mounted, with the same prefix
    and tags: the public, signature-verified webhook receiver (``/commands``,
    ``/events``, ``/interactive``) and the authenticated settings routes (the
    channel allowlist, ``/link``, the history import and its sync controls).
    """
    return [
        (get_slack_router(), "/api/v1/slack", ["slack"]),
        (get_slack_channels_router(), "/api/v1/slack", ["slack"]),
    ]


def get_slack_router():
    """Public Slack webhook receiver.

    Three unauthenticated-by-design endpoints — Slack cannot send a bearer token
    or cookie, so the ``X-Slack-Signature`` HMAC (enforced by the
    ``verify_slack_request`` dependency) is the entire auth model. Handlers
    receive the verified raw bytes and parse them themselves; adding a parsed
    body parameter here would break verification (see verify_slack_signature's
    module docstring).

    These are shared, multi-workspace URLs: every payload carries a ``team_id``
    that the modules-layer handlers resolve back to the connecting cognee user.
    """
    slack_router = APIRouter()

    @slack_router.post("/commands", include_in_schema=False)
    async def slack_commands(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_command(raw_body)

    @slack_router.post("/events", include_in_schema=False)
    async def slack_events(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_event(raw_body)

    @slack_router.post("/interactive", include_in_schema=False)
    async def slack_interactive(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_interactive(raw_body)

    return slack_router


def get_slack_history_router():
    """Authenticated Slack history selection and sync controls."""
    router = APIRouter(lifespan=slack_history_lifespan)

    @router.post("/history/{team_id}/import", response_model=SlackHistoryResult)
    async def import_history(
        team_id: str,
        selection: SlackHistoryRequest,
        user: Annotated[User, Depends(get_authenticated_user)],
    ):
        """Fetch and index the selection; returns only after native indexing succeeds.

        This is a long-running request. For interactive use, the Slack import
        dialog acknowledges immediately and reports its background result.
        """
        try:
            return await run_history_import(team_id, selection, user=user)
        except SlackHistoryError as error:
            raise HTTPException(400, str(error)) from error
        except PermissionDeniedError as error:
            raise HTTPException(
                403, "Read, write and delete access to the target dataset is required."
            ) from error

    @router.put("/history/{team_id}/sync", response_model=SlackSyncSettings)
    async def set_sync(
        team_id: str,
        settings: SlackSyncSettings,
        user: Annotated[User, Depends(get_authenticated_user)],
    ):
        try:
            return await configure_slack_sync(team_id, settings, user=user)
        except (SlackHistoryError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        except PermissionDeniedError as error:
            raise HTTPException(
                403, "Read, write and delete access to the target dataset is required."
            ) from error

    @router.get("/history/{team_id}")
    async def history_status(team_id: str, user: Annotated[User, Depends(get_authenticated_user)]):
        credential = await get_by_team(team_id)
        if not is_active(credential) or credential.user_id != user.id:
            raise HTTPException(404, "Slack connection not found.")
        metadata = credential.provider_metadata or {}
        return {
            "sync": metadata.get("history_sync") or {},
            "reports": metadata.get("history_reports") or {},
            "worker_enabled": bool(
                slack_settings.history_sync_enabled and slack_settings.client_id
            ),
        }

    return router


class ChannelDTO(OutDTO):
    id: str
    name: str
    is_private: bool
    allowed: bool


class ChannelListDTO(OutDTO):
    channels: list[ChannelDTO]


class SetAllowedChannelsPayload(InDTO):
    channel_ids: list[str]


class SetAllowedChannelsResultDTO(OutDTO):
    allowed_channel_ids: list[str]


class ConfirmLinkPayload(InDTO):
    code: str


class ConfirmLinkResultDTO(OutDTO):
    linked: bool


def _connected_credential_or_404(credential) -> IntegrationCredential:
    if credential is None:
        raise HTTPException(status_code=404, detail="Slack is not connected.")
    return credential


def get_slack_channels_router():
    """Slack channel-allowlist management — authenticated settings endpoints.

    Separate from get_slack_router's public, signature-verified webhooks on
    purpose: these need a real cognee session (Depends(get_authenticated_user)),
    not an HMAC over the raw body, and "which channels can run slash commands"
    is a Slack-specific concept that has no equivalent for a future Notion/GitHub
    integration — it does not belong on the generic integrations router either.
    """
    router = APIRouter()
    router.include_router(get_slack_history_router())

    @router.get("/channels")
    async def get_channels(
        user: Annotated[User, Depends(get_authenticated_user)],
    ) -> ChannelListDTO:
        """List the connected workspace's public channels, flagging the current allowlist."""
        credential = _connected_credential_or_404(
            await get_active_credential_for_user(user.id, PROVIDER)
        )
        access_token = decrypt_token_payload(credential).get("access_token")

        try:
            channels = await list_channels(access_token)
        except RuntimeError:
            # Most likely: the workspace connected before channels:read was
            # added to the app's scopes, and Slack never retroactively grants
            # new scopes to an existing installation — reconnecting is the fix.
            logger.exception(
                "Slack conversations.list failed for team %s", credential.provider_account_id
            )
            raise HTTPException(
                status_code=502,
                detail="Could not fetch Slack channels. Try disconnecting and reconnecting Slack.",
            )

        allowed_ids = set((credential.provider_metadata or {}).get("allowed_channel_ids") or [])
        return ChannelListDTO(
            channels=[
                ChannelDTO(
                    id=channel["id"],
                    name=channel["name"],
                    is_private=channel["is_private"],
                    allowed=channel["id"] in allowed_ids,
                )
                for channel in channels
            ]
        )

    @router.put("/channels")
    async def set_allowed_channels(
        payload: SetAllowedChannelsPayload, user: Annotated[User, Depends(get_authenticated_user)]
    ) -> SetAllowedChannelsResultDTO:
        """Restrict slash commands to exactly these channel ids.

        An empty list means unrestricted (the default) — channel scoping is
        opt-in, so a workspace that never visits this settings screen keeps
        working everywhere, exactly as before this feature existed.

        ## Request Parameters
        - **channelIds** (List[str]): Slack channel IDs allowed to run slash commands; an
          empty list removes all channel restrictions.
        """
        credential = _connected_credential_or_404(
            await get_active_credential_for_user(user.id, PROVIDER)
        )
        updated = await update_provider_metadata(
            PROVIDER,
            credential.provider_account_id,
            {"allowed_channel_ids": payload.channel_ids},
        )
        return SetAllowedChannelsResultDTO(
            allowed_channel_ids=updated.provider_metadata["allowed_channel_ids"]
        )

    @router.post("/link")
    async def link(
        payload: ConfirmLinkPayload, user: Annotated[User, Depends(get_authenticated_user)]
    ) -> ConfirmLinkResultDTO:
        """Confirm a ``/cognee-link`` magic-link code for the authenticated caller.

        Backs the ``/link-slack`` frontend page — the browser session here
        (not anything typed into Slack) is what proves which cognee account
        the invoking Slack member should be linked to.

        ## Request Parameters
        - **code** (str): Magic-link code issued by /cognee-link, confirmed to bind the
          Slack member to this account.
        """
        linked = await confirm_link(payload.code, user.id)
        if not linked:
            raise HTTPException(status_code=400, detail="This link is invalid or has expired.")
        return ConfirmLinkResultDTO(linked=True)

    return router
