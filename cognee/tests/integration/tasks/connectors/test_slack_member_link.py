"""A Slack member reaches a cognee account's memory only after signing in with Slack in it.

Linking a Slack member to a cognee account gives the member's ``@cognee`` questions and
remembered items that account's memory (``persistence.resolve_owner_user_id``). The link
is made by Sign in with Slack, started by the logged-in cognee user: Slack proves which
member is signing in, the signed state and the browser's nonce cookie prove which cognee
user started it (G42). Real relational rows, the real integrations routes over ASGI and
the real owner lookup; Slack's OpenID exchange is the only stand-in.
"""

import base64
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

import cognee
from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.api.v1.slack.routers.get_slack_channels_router import get_slack_channels_router
from cognee.modules.engine.operations.setup import setup
from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.slack import member_link
from cognee.modules.integrations.slack import slack_settings as settings_module
from cognee.modules.integrations.slack.handle_slack_link import link_message
from cognee.modules.integrations.slack.persistence import resolve_owner_user_id
from cognee.modules.users.methods import create_user, get_authenticated_user

TEAM = "T0LINK"
INSTALLER = "U0INSTALLER"
VICTIM_SLACK = "U0VICTIM"
ATTACKER_SLACK = "U0ATTACKER"

# What Slack's openid.connect.userInfo answers for the member who signed in with a code.
SIGNED_IN = {
    f"code-{member}": {
        "sub": member,
        "https://slack.com/user_id": member,
        "https://slack.com/team_id": TEAM,
    }
    for member in (VICTIM_SLACK, ATTACKER_SLACK)
}


@pytest_asyncio.fixture
async def clean_cognee(tmp_path, monkeypatch):
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"k" * 32).decode())
    for name, value in {
        "signing_secret": "test-signing-secret",
        "frontend_base_url": "http://localhost:3000",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "member_redirect_uri": "http://test/api/v1/integrations/slack_member/callback",
    }.items():
        monkeypatch.setattr(settings_module.slack_settings, name, value)

    async def sign_in(self, code: str) -> dict:
        return SIGNED_IN[code]

    monkeypatch.setattr(member_link.SlackMemberIntegration, "exchange_code", sign_in)
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()
    await store.upsert_credential(
        provider="slack",
        provider_account_id=TEAM,
        user_id=(await _user()).id,
        token_payload={"access_token": "xoxb-test"},
        provider_metadata={"installed_by_slack_user_id": INSTALLER},
    )
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _user():
    return await create_user(f"slack-{uuid4().hex[:8]}@example.com", "pw-Test-123")


def _browser(user) -> httpx.AsyncClient:
    """One browser, logged in to cognee as ``user``; it keeps its own cookies."""
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.include_router(get_slack_channels_router(), prefix="/api/v1/slack")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _start_link(browser: httpx.AsyncClient) -> str:
    """The cognee page's "Link Slack" button: the state in Slack's sign-in URL."""
    response = await browser.post("/api/v1/integrations/slack_member/authorize")
    assert response.status_code == 200, response.text
    url = urlparse(response.json()["authorizeUrl"])
    assert url.netloc == "slack.com" and url.path == "/openid/connect/authorize"
    return parse_qs(url.query)["state"][0]


async def _finish_link(browser: httpx.AsyncClient, state: str, member: str) -> httpx.Response:
    """Slack sends the browser back after ``member`` signed in."""
    return await browser.get(
        "/api/v1/integrations/slack_member/callback",
        params={"code": f"code-{member}", "state": state},
    )


async def _owner_of(member: str):
    credential = await store.get_credential_by_account("slack", TEAM)
    return await resolve_owner_user_id(credential, TEAM, member)


@pytest.mark.asyncio
async def test_a_member_who_signs_in_with_slack_reaches_the_users_memory(clean_cognee):
    user = await _user()
    async with _browser(user) as browser:
        state = await _start_link(browser)
        await _finish_link(browser, state, VICTIM_SLACK)

    assert await _owner_of(VICTIM_SLACK) == user.id
    assert await _owner_of(ATTACKER_SLACK) is None


@pytest.mark.asyncio
async def test_a_link_started_in_one_browser_cannot_be_finished_in_another(clean_cognee):
    """The attacker starts a link and gets the victim to sign in with Slack through it:
    the victim's browser holds no nonce for that state, so nothing is linked."""
    attacker = await _user()
    async with _browser(attacker) as attackers_browser:
        state = await _start_link(attackers_browser)
    async with _browser(await _user()) as victims_browser:
        await _finish_link(victims_browser, state, VICTIM_SLACK)

    assert await _owner_of(VICTIM_SLACK) is None


@pytest.mark.asyncio
async def test_no_link_code_exists_that_someone_else_could_confirm(clean_cognee):
    """The former flow: ``/cognee-link`` answered a code naming the member, and anyone
    logged in to cognee who opened it got that member linked to their account (G42)."""
    assert "code=" not in link_message()

    async with _browser(await _user()) as victims_browser:
        response = await victims_browser.post("/api/v1/slack/link", json={"code": "anything"})
    assert response.status_code in (404, 405)
    assert await _owner_of(ATTACKER_SLACK) is None


@pytest.mark.asyncio
async def test_the_user_can_unlink_a_member(clean_cognee):
    user = await _user()
    async with _browser(user) as browser:
        await _finish_link(browser, await _start_link(browser), VICTIM_SLACK)
        assert await _owner_of(VICTIM_SLACK) == user.id

        response = await browser.delete("/api/v1/integrations/slack_member/connection")

    assert response.status_code == 200, response.text
    assert await _owner_of(VICTIM_SLACK) is None


@pytest.mark.asyncio
async def test_a_member_linked_to_one_user_is_not_moved_to_another(clean_cognee):
    first, second = await _user(), await _user()
    async with _browser(first) as browser:
        await _finish_link(browser, await _start_link(browser), VICTIM_SLACK)
    async with _browser(second) as browser:
        response = await _finish_link(browser, await _start_link(browser), VICTIM_SLACK)

    assert response.status_code in (302, 303, 307)
    assert "error_already_connected" in response.headers["location"]
    assert await _owner_of(VICTIM_SLACK) == first.id


@pytest.mark.asyncio
async def test_a_member_unlinks_from_slack_with_cognee_unlink(clean_cognee):
    from cognee.modules.integrations.slack.handle_cognee_mention import UNLINKED_MESSAGE, unlink

    user = await _user()
    async with _browser(user) as browser:
        await _finish_link(browser, await _start_link(browser), VICTIM_SLACK)
    replies = []

    async def reply(payload):
        replies.append(payload["text"])

    await unlink(reply, team_id=TEAM, member=VICTIM_SLACK)

    assert replies == [UNLINKED_MESSAGE]
    assert await _owner_of(VICTIM_SLACK) is None


async def _deactivate(user) -> None:
    """What an admin's PATCH /users/{id} with is_active=false leaves in the database."""
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    async with get_relational_engine().get_async_session() as session:
        await session.execute(update(User).where(User.id == user.id).values(is_active=False))
        await session.commit()


@pytest.mark.asyncio
async def test_a_deactivated_account_is_not_reached_from_slack(clean_cognee):
    """Every API route refuses an inactive user; a linked member, or the installer, must
    not keep reading and writing that account's memory through Slack (G53)."""
    user = await _user()
    async with _browser(user) as browser:
        await _finish_link(browser, await _start_link(browser), VICTIM_SLACK)
    assert await _owner_of(VICTIM_SLACK) == user.id

    await _deactivate(user)
    assert await _owner_of(VICTIM_SLACK) is None

    credential = await store.get_credential_by_account("slack", TEAM)
    assert await _owner_of(INSTALLER) == credential.user_id
    await _deactivate(await _get_user(credential.user_id))
    assert await _owner_of(INSTALLER) is None


async def _get_user(user_id):
    from cognee.modules.users.methods import get_user

    return await get_user(user_id)


@pytest.mark.asyncio
async def test_a_deleted_account_is_not_reached_from_slack_and_does_not_fail_it(clean_cognee):
    """G63: a linked member whose cognee account was deleted is refused like a deactivated
    one; the lookup must not raise, or "Remember this" answers 500 and Slack sends it again."""
    from cognee.modules.users.methods.delete_user import delete_user

    user = await _user()
    async with _browser(user) as browser:
        await _finish_link(browser, await _start_link(browser), VICTIM_SLACK)
    assert await _owner_of(VICTIM_SLACK) == user.id

    await delete_user(user.email)

    assert await _owner_of(VICTIM_SLACK) is None


@pytest.mark.asyncio
async def test_a_member_who_opens_the_start_link_is_sent_to_slack_and_linked(clean_cognee):
    """The ``@cognee link`` link opens ``GET /slack_member/start`` in the browser where the
    user is logged in to cognee: it sets the same nonce cookie as ``authorize`` and sends the
    browser to Sign in with Slack, and the callback links that member to that user."""
    user = await _user()
    async with _browser(user) as browser:
        response = await browser.get("/api/v1/integrations/slack_member/start")
        assert response.status_code == 302, response.text
        url = urlparse(response.headers["location"])
        assert url.netloc == "slack.com" and url.path == "/openid/connect/authorize"
        state = parse_qs(url.query)["state"][0]
        await _finish_link(browser, state, VICTIM_SLACK)

    assert await _owner_of(VICTIM_SLACK) == user.id


@pytest.mark.asyncio
async def test_the_start_link_needs_a_cognee_login(clean_cognee):
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as anonymous:
        response = await anonymous.get("/api/v1/integrations/slack_member/start")
    assert response.status_code == 401
