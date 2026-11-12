"""No Slack call follows a redirect (SDK-832).

Copied from the Slack cases of cognee.tests.unit.modules.integrations.test_outbound_redirects
(cognee dev, 2026-11-12).

Every Slack host is a constant, so a 3xx can only come from Slack itself (or something
between it and us). Followed, a 307/308 makes aiohttp send the same body to the host the
``Location`` names: for the token endpoints that body is the client secret plus a code or
refresh token, for the rest it is a request carrying a bearer.

Each case points one call at a local server that answers 307 to a second server, which
records whatever reaches it: nothing may.
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web

from cognee_connector_slack.cognee_host import adapter as slack_adapter
from cognee_connector_slack.cognee_host import channels as slack_channels
from cognee_connector_slack.cognee_host import handle_cognee_mention as slack_mention
from cognee_connector_slack.cognee_host import home as slack_home
from cognee_connector_slack.cognee_host import member_link as slack_member_link
from cognee_connector_slack.cognee_host import oauth as slack_oauth


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


def _slack_member_sign_in(mp, base):
    mp.setattr(slack_member_link, "_TOKEN_URL", f"{base}/api/openid.connect.token")
    mp.setattr(slack_member_link, "require", lambda key: "x")
    return slack_member_link.SlackMemberIntegration().exchange_code("the-code"), RuntimeError


def _slack_web_api(mp, base):
    mp.setattr(slack_mention, "_SLACK_API", f"{base}/api/")

    async def token(credential):
        return "tok"

    mp.setattr(slack_mention, "access_token_for", token)
    return SimpleNamespace(provider_account_id="T1")


def _slack_post_ephemeral(mp, base):
    credential = _slack_web_api(mp, base)
    # Best effort like every reply: the failure is logged, nothing is raised.
    return slack_mention.post_ephemeral(credential, "C1", "U1", {"text": "hi"}), None


def _slack_post_message(mp, base):
    credential = _slack_web_api(mp, base)
    # An answer: the caller tells the asker, so the failure is raised.
    return slack_mention.post_message(credential, "C1", {"text": "hi"}), RuntimeError


_CASES = {
    "slack code exchange": _slack_code_exchange,
    "slack token refresh": _slack_refresh,
    "slack revoke": _slack_revoke,
    "slack views.publish": _slack_home,
    "slack conversations.list": _slack_channels,
    "slack sign in with slack": _slack_member_sign_in,
    "slack chat.postEphemeral": _slack_post_ephemeral,
    "slack chat.postMessage": _slack_post_message,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
async def test_no_slack_call_follows_a_redirect(case, redirecting_origin, monkeypatch):
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
async def test_no_slack_call_opens_a_connection_while_outbound_http_is_disabled(
    case, redirecting_origin, monkeypatch
):
    """SDK-832: every provider call goes through ``client_session``, which is where
    ``ALLOW_HTTP_REQUESTS=false`` stops it, whatever path led there (OAuth callback,
    webhook, resource picker, refresh, sync)."""
    from cognee.modules.integrations.outbound import OutboundHttpDisabledError

    call, _raises = _CASES[case](monkeypatch, redirecting_origin.base)
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")

    if case in ("slack revoke", "slack chat.postEphemeral"):
        # Both swallow every failure: a disconnect still completes, a reply is logged.
        await call
    else:
        with pytest.raises(OutboundHttpDisabledError):
            await call

    assert redirecting_origin.received == []
