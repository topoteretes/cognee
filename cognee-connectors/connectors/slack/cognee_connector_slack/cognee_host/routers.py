"""Slack HTTP routes for a cognee server.

Copied from cognee.api.v1.slack.routers (get_slack_router, get_slack_channels_router and
the package ``__init__``; cognee dev, 2026-11-12) into one module. :func:`routers` lists
what a host mounts, with the prefix and tags cognee's ``api/client.py`` used; each factory
keeps its original module docstring as its own.
"""

import logging
from typing import Annotated

from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    get_active_credential_for_user,
    update_provider_metadata,
)
from cognee.modules.integrations.ingestion import RateLimitedError
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from fastapi import APIRouter, Depends, HTTPException, Request

from cognee_connector_slack.cognee_host.adapter import visible_channels
from cognee_connector_slack.cognee_host.handle_slack_event import handle_slack_event
from cognee_connector_slack.cognee_host.handle_slack_interactive import handle_slack_interactive
from cognee_connector_slack.cognee_host.persistence import PROVIDER
from cognee_connector_slack.cognee_host.sync import FORGET_UNAVAILABLE_KEY, slack_sync_lifespan
from cognee_connector_slack.cognee_host.verify_slack_signature import verify_slack_request

logger = logging.getLogger(__name__)

__all__ = ["get_slack_channels_router", "get_slack_router", "routers"]


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags).

    The same two routers cognee's ``api/client.py`` mounted, with the same prefix and
    tags: the public, signature-verified webhook receiver (``/events``, ``/interactive``)
    and the authenticated settings routes (the command allowlist, ``/settings``), whose
    lifespan runs the channel sync tick.
    """
    return [
        (get_slack_router(), "/api/v1/slack", ["slack"]),
        (get_slack_channels_router(), "/api/v1/slack", ["slack"]),
    ]


def get_slack_router():
    """Public Slack webhook receiver.

    Two unauthenticated-by-design endpoints — Slack cannot send a bearer token
    or cookie, so the ``X-Slack-Signature`` HMAC (enforced by the
    ``verify_slack_request`` dependency) is the entire auth model. Handlers
    receive the verified raw bytes and parse them themselves; adding a parsed
    body parameter here would break verification (see verify_slack_signature's
    module docstring).

    These are shared, multi-workspace URLs: every payload carries a ``team_id``
    that the modules-layer handlers resolve back to the connecting cognee user.
    """
    slack_router = APIRouter()

    @slack_router.post("/events", include_in_schema=False)
    async def slack_events(request: Request, raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_event(
            raw_body, retry_reason=request.headers.get("x-slack-retry-reason")
        )

    @slack_router.post("/interactive", include_in_schema=False)
    async def slack_interactive(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_interactive(raw_body)

    return slack_router


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


class SlackSettingsPayload(InDTO):
    forget_unavailable_channels: bool


class SlackSettingsDTO(OutDTO):
    forget_unavailable_channels: bool


def _connected_credential_or_404(credential) -> IntegrationCredential:
    if credential is None:
        raise HTTPException(status_code=404, detail="Slack is not connected.")
    return credential


async def _offered_channels(credential: IntegrationCredential) -> list[dict]:
    """The channels the allowlist offers, or a 502 when Slack refuses the list.

    An allowed private channel whose members cannot be read right now is still offered, so
    the editor neither drops it nor refuses it when its list is saved back.
    """
    access_token = decrypt_token_payload(credential).get("access_token")
    allowed = (credential.provider_metadata or {}).get("allowed_channel_ids") or []
    try:
        return await visible_channels(credential, access_token, selected=allowed)
    except RateLimitedError as error:
        # Never waited out inside the request: the client is told when to try again.
        raise HTTPException(
            status_code=429,
            detail="Slack is limiting requests; try again shortly.",
            headers={"Retry-After": str(error.retry_after)} if error.retry_after else None,
        )
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


def get_slack_channels_router():
    """Slack channel-allowlist management — authenticated settings endpoints.

    Separate from :func:`get_slack_router`'s public, signature-verified webhooks on
    purpose: these need a real cognee session (Depends(get_authenticated_user)),
    not an HMAC over the raw body, and "which channels Cognee's commands work in"
    is a Slack-specific concept that has no equivalent for a future Notion/GitHub
    integration — it does not belong on the generic integrations router either.
    """
    # The channel sync tick runs for the life of the app; channel selection and manual
    # syncs go through the generic /api/v1/integrations/slack routes.
    router = APIRouter(lifespan=slack_sync_lifespan)

    @router.get("/channels")
    async def get_channels(
        user: Annotated[User, Depends(get_authenticated_user)],
    ) -> ChannelListDTO:
        """List the channels Cognee's commands can be allowed in, flagging the current allowlist.

        Every public channel, and the private channels the app is in that the person who
        connected the workspace is in too: a private channel they cannot see is never named.
        """
        credential = _connected_credential_or_404(
            await get_active_credential_for_user(user.id, PROVIDER)
        )
        channels = await _offered_channels(credential)

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
        """Restrict Cognee's commands and "Remember this" to exactly these channel ids.

        The allowlist decides where Cognee can be used in Slack, never which channels are
        synced into memory: that is the channel selection's alone. An empty list means
        unrestricted (the default) — channel scoping is opt-in, so a workspace that never
        visits this settings screen keeps working everywhere.

        ## Request Parameters
        - **channelIds** (List[str]): Slack channel IDs Cognee may be used in, each one the
          channels listing offers; an empty list removes all channel restrictions.

        ## Error Codes
        - **422**: an id the channels listing does not offer (a name instead of an id, a
          private channel the connecting user is not in); nothing is saved.
        """
        credential = _connected_credential_or_404(
            await get_active_credential_for_user(user.id, PROVIDER)
        )
        if payload.channel_ids:
            offered = {channel["id"] for channel in await _offered_channels(credential)}
            unknown = [
                channel_id for channel_id in payload.channel_ids if channel_id not in offered
            ]
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail=f"Not a channel Cognee can be allowed in: {', '.join(unknown)}",
                )
        updated = await update_provider_metadata(
            PROVIDER,
            credential.provider_account_id,
            {"allowed_channel_ids": payload.channel_ids},
        )
        return SetAllowedChannelsResultDTO(
            allowed_channel_ids=updated.provider_metadata["allowed_channel_ids"]
        )

    @router.get("/settings")
    async def get_settings(
        user: Annotated[User, Depends(get_authenticated_user)],
    ) -> SlackSettingsDTO:
        """The connection owner's Slack sync settings."""
        credential = _connected_credential_or_404(
            await get_active_credential_for_user(user.id, PROVIDER)
        )
        metadata = credential.provider_metadata or {}
        return SlackSettingsDTO(
            forget_unavailable_channels=metadata.get(FORGET_UNAVAILABLE_KEY) is True
        )

    @router.put("/settings")
    async def set_settings(
        payload: SlackSettingsPayload, user: Annotated[User, Depends(get_authenticated_user)]
    ) -> SlackSettingsDTO:
        """Decide what happens to a selected channel that was deleted in Slack or that the
        app was removed from.

        Off (the default), the channel keeps its memory and every sync reports it under
        ``failed_channel_unavailable`` until it is deselected. On, the next sync removes its
        memory like a deselected channel's. Only the connection's owner can change it.

        ## Request Parameters
        - **forgetUnavailableChannels** (bool): Remove the memory of unavailable channels.
        """
        credential = _connected_credential_or_404(
            await get_active_credential_for_user(user.id, PROVIDER)
        )
        updated = await update_provider_metadata(
            PROVIDER,
            credential.provider_account_id,
            {FORGET_UNAVAILABLE_KEY: payload.forget_unavailable_channels},
        )
        return SlackSettingsDTO(
            forget_unavailable_channels=updated.provider_metadata[FORGET_UNAVAILABLE_KEY]
        )

    return router
