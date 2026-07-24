"""Handle ``/cognee-link`` — link the invoking Slack member's own cognee
account via a magic link, not a pasted API key.

One Slack OAuth install serves an entire workspace, but ``/cognee-ask`` and
"Remember this" need to know *which* cognee account each individual member
should read from/write to. Typical SDK/local usage has no standing API key
to paste, so this mints a short-lived, signed code instead — the member
opens it in a browser where they're already logged in and confirms there
(see the ``/link-slack`` frontend page and ``POST /api/v1/slack/link``,
which :func:`confirm_link` backs); nothing is ever typed or pasted into
Slack itself.

The link itself reuses the generic ``integration_credentials`` table rather
than a new one: a member-link is stored under ``provider="slack_member"``
(distinct from ``"slack"``, the workspace-level bot credential, so the two
never collide on the ``UNIQUE(provider, provider_account_id)`` index) with
``provider_account_id="{team_id}:{slack_user_id}"`` and an empty token
payload — there is no secret to keep here, only an identity pointer, so
this rides the existing credential machinery for free instead of adding a
migration.

The command's own reply is ephemeral, same reasoning as everywhere else in
this integration: only the invoking member should ever see their own link.
"""

import time
from typing import Any, Optional, Tuple
from urllib.parse import parse_qs

from cognee.modules.integrations.credentials import upsert_credential
from cognee.modules.integrations.oauth_flow import sign_state_payload
from cognee.modules.integrations.slack.slack_settings import require

# Distinct from PROVIDER ("slack") on purpose — see module docstring.
MEMBER_LINK_PROVIDER = "slack_member"

# Long enough to open Slack, switch to the browser, and click Confirm;
# short enough that a leaked link (pasted somewhere, sitting in a browser
# history) is stale soon after.
LINK_CODE_TTL_SECONDS = 60 * 10


def _ephemeral(text: str) -> dict[str, Any]:
    return {"response_type": "ephemeral", "text": text}


def member_link_account_id(team_id: str, slack_user_id: str) -> str:
    """The synthetic ``provider_account_id`` a member-link is stored/looked up under."""
    return f"{team_id}:{slack_user_id}"


def make_link_code(team_id: str, slack_user_id: str) -> str:
    """Mint the signed, expiring code the ``/link-slack`` page trades in for a link."""
    expires = int(time.time()) + LINK_CODE_TTL_SECONDS
    payload = f"{team_id}:{slack_user_id}:{expires}"
    signature = sign_state_payload(payload, signing_secret=require("signing_secret"))
    return f"{payload}:{signature}"


def validate_link_code(code: str) -> Optional[Tuple[str, str]]:
    """Return ``(team_id, slack_user_id)`` for a valid, unexpired code; ``None`` otherwise.

    Verifies the HMAC before reading any field, so a forged or tampered
    code never influences behavior.
    """
    parts = (code or "").split(":")
    if len(parts) != 4:
        return None
    team_id, slack_user_id, expires_str, signature = parts

    payload = f"{team_id}:{slack_user_id}:{expires_str}"
    if sign_state_payload(payload, signing_secret=require("signing_secret")) != signature:
        return None

    try:
        if int(expires_str) < time.time():
            return None
    except ValueError:
        return None

    return team_id, slack_user_id


async def handle_cognee_link(raw_body: bytes) -> dict[str, Any]:
    """Mint a magic link for the invoking Slack member to confirm in a browser."""
    form = parse_qs(raw_body.decode())
    team_id = (form.get("team_id") or [""])[0]
    invoking_slack_user_id = (form.get("user_id") or [""])[0]

    code = make_link_code(team_id, invoking_slack_user_id)
    link = f"{require('frontend_base_url').rstrip('/')}/link-slack?code={code}"

    return _ephemeral(
        f"Open this in your browser (where you're already logged in) to link your Cognee "
        f"account: {link}\nThis link expires in 10 minutes."
    )


async def confirm_link(code: str, cognee_user_id) -> bool:
    """Resolve ``code`` and link it to ``cognee_user_id`` — the authenticated caller
    of ``POST /api/v1/slack/link``. Returns ``False`` for an invalid/expired code.
    """
    resolved = validate_link_code(code)
    if resolved is None:
        return False

    team_id, invoking_slack_user_id = resolved
    await upsert_credential(
        provider=MEMBER_LINK_PROVIDER,
        provider_account_id=member_link_account_id(team_id, invoking_slack_user_id),
        user_id=cognee_user_id,
        token_payload={},
        auth_type="api_key",
    )
    return True
