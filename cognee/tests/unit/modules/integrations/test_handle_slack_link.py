"""Unit tests for cognee.modules.integrations.slack.handle_slack_link.

Two halves: /cognee-link mints a magic link (no API key involved at all),
and confirm_link (backing POST /api/v1/slack/link) resolves that link's
code and performs the actual member-link write.
"""

import time
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.modules.integrations.oauth_flow import sign_state_payload
from cognee.modules.integrations.slack.handle_slack_link import (
    MEMBER_LINK_PROVIDER,
    confirm_link,
    handle_cognee_link,
    make_link_code,
    member_link_account_id,
    validate_link_code,
)

MODULE = "cognee.modules.integrations.slack.handle_slack_link"

_SETTINGS = {"signing_secret": "test-signing-secret", "frontend_base_url": "http://localhost:3000"}


def _require(name: str) -> str:
    return _SETTINGS[name]


def _body(team_id="T123", user_id="U100"):
    return urlencode({"team_id": team_id, "user_id": user_id}).encode()


# ── handle_cognee_link: mint the magic link ─────────────────────────────────


@pytest.mark.asyncio
async def test_mints_a_link_pointing_at_the_frontend():
    with patch(f"{MODULE}.require", side_effect=_require):
        response = await handle_cognee_link(_body(team_id="T123", user_id="U100"))

    assert response["response_type"] == "ephemeral"
    assert "http://localhost:3000/link-slack?code=" in response["text"]
    assert "expires in 10 minutes" in response["text"]


@pytest.mark.asyncio
async def test_minted_link_code_resolves_back_to_the_invoking_member():
    with patch(f"{MODULE}.require", side_effect=_require):
        response = await handle_cognee_link(_body(team_id="T123", user_id="U100"))
        code = response["text"].split("code=")[1].split("\n")[0]
        resolved = validate_link_code(code)

    assert resolved == ("T123", "U100")


# ── make_link_code / validate_link_code ─────────────────────────────────────


def test_validate_link_code_rejects_tampered_signature():
    with patch(f"{MODULE}.require", side_effect=_require):
        code = make_link_code("T123", "U100")
        team_id, slack_user_id, expires, _signature = code.split(":")
        tampered = f"{team_id}:{slack_user_id}:{expires}:deadbeef"

        assert validate_link_code(tampered) is None


def test_validate_link_code_rejects_expired_code():
    # A genuinely, correctly-signed code whose expiry has already passed —
    # distinct from tampered-signature rejection above.
    payload = f"T123:U100:{int(time.time()) - 1}"
    with patch(f"{MODULE}.require", side_effect=_require):
        signature = sign_state_payload(payload, signing_secret=_SETTINGS["signing_secret"])
        expired = f"{payload}:{signature}"

        assert validate_link_code(expired) is None


def test_validate_link_code_rejects_malformed_code():
    with patch(f"{MODULE}.require", side_effect=_require):
        assert validate_link_code("not-enough-parts") is None
        assert validate_link_code("") is None


def test_member_link_account_id_combines_team_and_user():
    assert member_link_account_id("T123", "U100") == "T123:U100"


# ── confirm_link: POST /api/v1/slack/link ───────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_link_writes_the_credential_for_a_valid_code():
    cognee_user_id = uuid4()
    with (
        patch(f"{MODULE}.require", side_effect=_require),
        patch(f"{MODULE}.upsert_credential", new=AsyncMock()) as upsert,
    ):
        code = make_link_code("T123", "U100")
        linked = await confirm_link(code, cognee_user_id)

    assert linked is True
    upsert.assert_awaited_once_with(
        provider=MEMBER_LINK_PROVIDER,
        provider_account_id=member_link_account_id("T123", "U100"),
        user_id=cognee_user_id,
        token_payload={},
        auth_type="api_key",
    )


@pytest.mark.asyncio
async def test_confirm_link_rejects_invalid_code_without_writing():
    with patch(f"{MODULE}.upsert_credential", new=AsyncMock()) as upsert:
        linked = await confirm_link("garbage", uuid4())

    assert linked is False
    upsert.assert_not_awaited()
