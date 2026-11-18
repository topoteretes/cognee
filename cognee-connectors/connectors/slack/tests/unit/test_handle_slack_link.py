"""Linking a Slack member: ``/cognee-link`` points at Sign in with Slack, and the
``slack_member`` provider turns Slack's answer into the member link (G42).

Copied from cognee.tests.unit.modules.integrations.test_handle_slack_link (cognee dev, 2026-11-12).

The flow through the real routes, including the attacks, is in
``cognee/tests/integration/tasks/connectors/test_slack_member_link.py``.
"""

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from cognee_connector_slack.cognee_host import member_link
from cognee_connector_slack.cognee_host.handle_slack_link import (
    MEMBER_LINK_PROVIDER,
    link_message,
    member_link_account_id,
)
from cognee_connector_slack.cognee_host.member_link import SlackMemberIntegration

_SETTINGS = {
    "signing_secret": "test-signing-secret",
    "frontend_base_url": "http://localhost:3000",
    "client_id": "client-id",
    "client_secret": "client-secret",
    "member_redirect_uri": "https://cognee.example/api/v1/integrations/slack_member/callback",
}


def _require(name: str) -> str:
    return _SETTINGS[name]


@pytest.fixture(autouse=True)
def settings():
    with (
        patch("cognee_connector_slack.cognee_host.handle_slack_link.require", side_effect=_require),
        patch.object(member_link, "require", side_effect=_require),
    ):
        yield


def test_the_link_message_points_at_the_server_route_that_starts_sign_in_and_carries_no_code():
    """Without a frontend page, the link opens the server's start route for the member
    provider, beside the callback Slack already redirects to."""
    text = link_message()

    assert "https://cognee.example/api/v1/integrations/slack_member/start" in text
    assert "/link-slack" not in text
    assert "code=" not in text


def test_member_link_account_id_combines_team_and_user():
    assert member_link_account_id("T123", "U100") == "T123:U100"


def test_the_member_provider_is_its_own_connection():
    assert SlackMemberIntegration.provider == MEMBER_LINK_PROVIDER == "slack_member"


def test_sign_in_with_slack_asks_for_the_member_id_only():
    url = urlparse(SlackMemberIntegration().authorize_url("the-state"))
    query = parse_qs(url.query)

    assert (url.netloc, url.path) == ("slack.com", "/openid/connect/authorize")
    assert query["scope"] == ["openid"], "no email or profile"
    assert query["response_type"] == ["code"]
    assert query["state"] == ["the-state"]
    assert query["client_id"] == ["client-id"]
    assert query["redirect_uri"] == [_SETTINGS["member_redirect_uri"]]
    assert query["nonce"][0]


def test_the_link_is_the_member_slack_named_and_keeps_no_token():
    installation = SlackMemberIntegration().parse_installation(
        {
            "ok": True,
            "sub": "U100",
            "https://slack.com/user_id": "U100",
            "https://slack.com/team_id": "T123",
        }
    )

    assert installation.provider_account_id == "T123:U100"
    assert installation.token_payload == {}


@pytest.mark.parametrize(
    "identity",
    [
        {"https://slack.com/user_id": "U100"},
        {"https://slack.com/team_id": "T123"},
        {"sub": "U100"},
    ],
)
def test_an_answer_without_the_member_or_the_workspace_links_nothing(identity):
    with pytest.raises(ValueError):
        SlackMemberIntegration().parse_installation(identity)


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


class _Session:
    def __init__(self, answers):
        self.answers = answers
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, *, data=None, headers=None, allow_redirects=True):
        # SDK-832: neither call may follow a redirect.
        assert allow_redirects is False
        self.posts.append((url, data, headers))
        return _Response(self.answers[url])


@pytest.mark.asyncio
async def test_the_code_is_exchanged_and_slack_asked_who_signed_in():
    identity = {"ok": True, "https://slack.com/user_id": "U100", "https://slack.com/team_id": "T"}
    session = _Session(
        {
            member_link._TOKEN_URL: {"ok": True, "access_token": "xoxp-user"},
            member_link._USER_INFO_URL: identity,
        }
    )
    with patch.object(member_link.aiohttp, "ClientSession", lambda **kwargs: session):
        answer = await SlackMemberIntegration().exchange_code("the-code")

    assert answer == identity
    token_post, user_info_post = session.posts
    assert token_post[1]["code"] == "the-code"
    assert token_post[1]["redirect_uri"] == _SETTINGS["member_redirect_uri"]
    assert user_info_post[2] == {"Authorization": "Bearer xoxp-user"}


@pytest.mark.asyncio
async def test_a_refused_exchange_raises_without_echoing_the_code():
    session = _Session({member_link._TOKEN_URL: {"ok": False, "error": "invalid_code"}})
    with (
        patch.object(member_link.aiohttp, "ClientSession", lambda **kwargs: session),
        pytest.raises(RuntimeError, match="invalid_code") as caught,
    ):
        await SlackMemberIntegration().exchange_code("secret-code")

    assert "secret-code" not in str(caught.value)
