"""No connector call follows a redirect (SDK-832).

Every provider host is a constant, so a 3xx can only come from the provider
itself (or something between it and us). Followed, a 307/308 makes aiohttp
send the same body to the host the ``Location`` names: for the token
endpoints that body is the client secret plus a code or refresh token, for
the rest it is a request carrying a bearer. SDK-796 closed this for Slack's
``response_url``; this pins it for every other outbound call the
integrations make, the same way ``test_linear_redirects.py`` does for Linear.

Each case points one call at a local server that answers 307 to a second
server, which records whatever reaches it: nothing may.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web

from cognee.modules.integrations.github import adapter as github_adapter
from cognee.modules.integrations.github import app_auth as github_app_auth
from cognee.modules.integrations.github import sync as github_sync
from cognee.modules.integrations.google import client as google_client
from cognee.modules.integrations.google_drive import client as google_drive_client
from cognee.modules.integrations.linear import client as linear_client
from cognee.modules.integrations.notion import client as notion_client
from cognee.modules.integrations.slack import adapter as slack_adapter
from cognee.modules.integrations.slack import channels as slack_channels
from cognee.modules.integrations.slack import home as slack_home
from cognee.modules.integrations.slack import oauth as slack_oauth


async def _serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


# 301/302/303 turn a followed POST into a GET without its body; 307/308 keep
# method and body. Both must be refused: the first still sends the bearer
# along, the second the secrets in the body.
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


@pytest_asyncio.fixture(params=_REDIRECT_STATUSES, ids=[str(s) for s in _REDIRECT_STATUSES])
async def redirecting_origin(request):
    """A provider stand-in answering a 3xx to a second server that records what it gets."""
    received = []
    status = request.param

    async def elsewhere(request):
        received.append((request.method, await request.text()))
        return web.json_response({"ok": True, "access_token": "stolen", "token": "stolen"})

    other, other_port = await _serve(elsewhere)

    async def origin(request):
        # A fixed target: nothing from the request decides where this redirects.
        return web.Response(status=status, headers={"Location": f"http://localhost:{other_port}/x"})

    runner, port = await _serve(origin)
    yield SimpleNamespace(base=f"http://127.0.0.1:{port}", received=received, status=status)
    await runner.cleanup()
    await other.cleanup()


_INSTALLATION = {"id": 1, "account": {"login": "acme", "id": 9, "type": "Organization"}}


def _slack_code_exchange(mp, base):
    mp.setattr(slack_oauth, "_ACCESS_URL", f"{base}/api/oauth.v2.access")
    mp.setattr(slack_oauth, "require", lambda key: "x")
    return slack_oauth.exchange_code("the-code"), RuntimeError


def _slack_refresh(mp, base):
    mp.setattr(slack_oauth, "_ACCESS_URL", f"{base}/api/oauth.v2.access")
    mp.setattr(slack_adapter, "require", lambda key: "x")
    mp.setattr(slack_adapter, "decrypt_token_payload", lambda c: {"refresh_token": "r"})
    return slack_adapter.SlackIntegration().refresh(SimpleNamespace()), RuntimeError


def _slack_revoke(mp, base):
    mp.setattr(slack_adapter, "_REVOKE_URL", f"{base}/api/auth.revoke")
    mp.setattr(slack_adapter, "decrypt_token_payload", lambda c: {"access_token": "a"})
    credential = SimpleNamespace(provider_account_id="T1")
    return slack_adapter.SlackIntegration().revoke_remote(credential), None


def _slack_home(mp, base):
    mp.setattr(slack_home, "_VIEWS_PUBLISH_URL", f"{base}/api/views.publish")
    return slack_home.publish_home_view("tok", "U1"), RuntimeError


def _slack_channels(mp, base):
    mp.setattr(slack_channels, "_CHANNELS_URL", f"{base}/api/conversations.list")
    return slack_channels.list_channels("tok"), RuntimeError


def _google_token(mp, base):
    mp.setattr(google_client, "TOKEN_URL", f"{base}/token")
    return (
        google_client.exchange_code("c", client_id="i", client_secret="s", redirect_uri="r"),
        google_client.GoogleAuthError,
    )


def _google_revoke(mp, base):
    mp.setattr(google_client, "REVOKE_URL", f"{base}/revoke")
    return google_client.revoke_token("tok"), None


def _google_userinfo(mp, base):
    mp.setattr(google_client, "USERINFO_URL", f"{base}/userinfo")
    return google_client.fetch_userinfo("tok"), RuntimeError


def _google_drive_text(mp, base):
    return google_drive_client._authorized_text(
        "export", f"{base}/files/x", "tok", {}
    ), RuntimeError


def _github_code_exchange(mp, base):
    mp.setattr(github_adapter, "_ACCESS_TOKEN_URL", f"{base}/login/oauth/access_token")
    mp.setattr(github_adapter, "require", lambda key: "x")
    return github_adapter.GithubIntegration().exchange_code("the-code"), RuntimeError


def _github_installation(mp, base):
    mp.setattr(github_app_auth, "API_BASE_URL", base)
    mp.setattr(github_app_auth, "build_app_jwt", lambda: "jwt")
    return github_app_auth.get_installation(1), RuntimeError


def _github_mint(mp, base):
    mp.setattr(github_app_auth, "API_BASE_URL", base)
    mp.setattr(github_app_auth, "build_app_jwt", lambda: "jwt")
    return github_app_auth.mint_installation_token(1), RuntimeError


def _github_admin_check(mp, base):
    mp.setattr(github_app_auth, "API_BASE_URL", base)
    return github_app_auth.user_administers_installation("tok", _INSTALLATION), RuntimeError


async def _noop(**kwargs):
    return None


def _github_repositories(mp, base):
    mp.setattr(github_sync, "API_BASE_URL", base)
    mp.setattr(github_sync, "report_sync_progress", _noop)
    return github_sync.list_installation_repositories("tok"), RuntimeError


def _linear_graphql(mp, base):
    mp.setattr(linear_client, "GRAPHQL_URL", f"{base}/graphql")
    return linear_client.graphql("tok", "query A { x }"), RuntimeError


def _notion_grant(mp, base):
    mp.setattr(notion_client, "TOKEN_URL", f"{base}/v1/oauth/token")
    return (
        notion_client.exchange_code("c", client_id="i", client_secret="s", redirect_uri="r"),
        notion_client.NotionError,
    )


def _notion_revoke(mp, base):
    mp.setattr(notion_client, "REVOKE_URL", f"{base}/v1/oauth/revoke")
    return notion_client.revoke_token("tok", client_id="i", client_secret="s"), None


_CASES = {
    "slack code exchange": _slack_code_exchange,
    "slack token refresh": _slack_refresh,
    "slack revoke": _slack_revoke,
    "slack views.publish": _slack_home,
    "slack conversations.list": _slack_channels,
    "google token": _google_token,
    "google revoke": _google_revoke,
    "google userinfo": _google_userinfo,
    "google drive download": _google_drive_text,
    "github code exchange": _github_code_exchange,
    "github installation lookup": _github_installation,
    "github token mint": _github_mint,
    "github admin check": _github_admin_check,
    "github repository listing": _github_repositories,
    "linear graphql": _linear_graphql,
    "notion grant": _notion_grant,
    "notion revoke": _notion_revoke,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
async def test_no_integration_call_follows_a_redirect(case, redirecting_origin, monkeypatch):
    call, raises = _CASES[case](monkeypatch, redirecting_origin.base)

    if raises is None:
        await call  # best-effort calls log and return
    else:
        # The failure names the status it got, so the log says what happened.
        with pytest.raises(raises, match=str(redirecting_origin.status)):
            await call

    assert redirecting_origin.received == []


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
async def test_no_integration_call_opens_a_connection_while_outbound_http_is_disabled(
    case, redirecting_origin, monkeypatch
):
    """SDK-832: every provider call goes through ``client_session``, which is where
    ``ALLOW_HTTP_REQUESTS=false`` stops it, whatever path led there (OAuth callback,
    webhook, resource picker, refresh, sync)."""
    from cognee.modules.integrations.outbound import OutboundHttpDisabledError

    call, _raises = _CASES[case](monkeypatch, redirecting_origin.base)
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")

    if case == "slack revoke":
        await call  # revoke_remote swallows every failure so disconnect still completes
    else:
        with pytest.raises(OutboundHttpDisabledError):
            await call

    assert redirecting_origin.received == []
