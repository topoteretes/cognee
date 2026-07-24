"""Slack channel-allowlist management — authenticated settings endpoints.

Separate from get_slack_router.py's public, signature-verified webhooks on
purpose: these need a real cognee session (Depends(get_authenticated_user)),
not an HMAC over the raw body, and "which channels can run slash commands"
is a Slack-specific concept that has no equivalent for a future Notion/GitHub
integration — it does not belong on the generic integrations router either.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    get_active_credential_for_user,
    update_provider_metadata,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.slack.channels import list_channels
from cognee.modules.integrations.slack.handle_slack_link import confirm_link
from cognee.modules.integrations.slack.persistence import PROVIDER
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User

logger = logging.getLogger(__name__)


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
    router = APIRouter()

    @router.get("/channels")
    async def get_channels(user: User = Depends(get_authenticated_user)) -> ChannelListDTO:
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
        payload: SetAllowedChannelsPayload, user: User = Depends(get_authenticated_user)
    ) -> SetAllowedChannelsResultDTO:
        """Restrict slash commands to exactly these channel ids.

        An empty list means unrestricted (the default) — channel scoping is
        opt-in, so a workspace that never visits this settings screen keeps
        working everywhere, exactly as before this feature existed.
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
        payload: ConfirmLinkPayload, user: User = Depends(get_authenticated_user)
    ) -> ConfirmLinkResultDTO:
        """Confirm a ``/cognee-link`` magic-link code for the authenticated caller.

        Backs the ``/link-slack`` frontend page — the browser session here
        (not anything typed into Slack) is what proves which cognee account
        the invoking Slack member should be linked to.
        """
        linked = await confirm_link(payload.code, user.id)
        if not linked:
            raise HTTPException(status_code=400, detail="This link is invalid or has expired.")
        return ConfirmLinkResultDTO(linked=True)

    return router
