"""Unit tests for the generic {provider} dispatch in get_integrations_router.

A fake OAuthIntegration is registered under provider "fake" so these tests
exercise the router's own generic logic (dispatch, error-to-redirect
mapping, unknown-provider 404) without depending on Slack or any real
network call. Slack's own OAuth mechanics are covered separately by
test_slack_adapter.py and test_oauth_state.py.
"""

import importlib
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.credentials import CrossUserConflictError
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()

# The router package's __init__.py does `from .get_integrations_router import
# get_integrations_router`, which rebinds that name on the package to the
# *function* — shadowing the submodule Python auto-attaches there on import.
# String-target patch("...routers.get_integrations_router.X") resolves that
# name via attribute traversal on some Python versions (3.10) and via
# importlib.import_module on others (3.12+), so it silently returns the
# function instead of the module on 3.10 and AttributeErrors. Importing the
# submodule explicitly sidesteps the shadowed attribute entirely.
_router_module = importlib.import_module(
    "cognee.api.v1.integrations.routers.get_integrations_router"
)


class _FakeUser:
    id = USER_ID


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def authorize_url(self, state):
        return f"https://fake.example/authorize?state={state}"

    async def exchange_code(self, code):
        return {"code": code}

    def parse_installation(self, token_response):
        return OAuthInstallation(provider_account_id="ACC1", token_payload={})

    def state_signing_secret(self):
        return "fake-secret"

    def frontend_base_url(self):
        return "https://app.example.com"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()

    before = dict(supported_integrations)
    supported_integrations.clear()
    use_integration(_FakeIntegration())

    yield TestClient(app)

    supported_integrations.clear()
    supported_integrations.update(before)


def test_unknown_provider_404s_on_every_route(client):
    assert client.post("/api/v1/integrations/notreal/authorize").status_code == 404
    assert client.get("/api/v1/integrations/notreal/callback").status_code == 404
    assert client.get("/api/v1/integrations/notreal/connection").status_code == 404
    assert client.delete("/api/v1/integrations/notreal/connection").status_code == 404


def start_install(client, provider: str = "fake") -> str:
    """Run the real authorize call and return the state it minted.

    Hand-minting a state skips the nonce cookie the callback now requires,
    which is exactly the check under test elsewhere in this file: a state on
    its own no longer completes an install. Going through authorize leaves the
    cookie in the TestClient's jar, the way a browser would carry it.
    """
    response = client.post(f"/api/v1/integrations/{provider}/authorize")
    assert response.status_code == 200
    return response.json()["authorizeUrl"].split("state=", 1)[1]


def pending_nonce_cookies(client, provider: str = "fake") -> list[str]:
    """Every install-nonce cookie this jar currently holds for `provider`.

    There is one cookie per pending install (see the router's own comment on
    _install_nonce_cookie_name), not one shared cookie, so "what is pending"
    is read as the set of cookie *names* matching the prefix.
    """
    prefix = f"cognee_oauth_nonce_{provider}_"
    return [name for name in client.cookies if name.startswith(prefix)]


def test_authorize_returns_the_integrations_own_url(client):
    response = client.post("/api/v1/integrations/fake/authorize")
    assert response.status_code == 200
    assert response.json()["authorizeUrl"].startswith("https://fake.example/authorize?state=")


def test_authorize_binds_the_install_to_the_browser_that_started_it(client):
    response = client.post("/api/v1/integrations/fake/authorize")
    names = [name for name in response.cookies if name.startswith("cognee_oauth_nonce_fake_")]
    assert len(names) == 1
    # httponly and the path restriction keep it off every unrelated request
    # and out of reach of page scripts.
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie
    assert "path=/api/v1/integrations" in set_cookie


def test_callback_without_the_install_cookie_is_refused(client):
    # The attack this closes: A calls authorize, hands the consent URL to B,
    # B completes it in B's own browser. The state is authentic, but it is not
    # B's install, and without this check B's provider account would land on
    # A's cognee user.
    state = start_install(client)
    client.cookies.clear()

    response = client.get(
        f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
    )
    assert "fake=error_invalid_state" in response.headers["location"]


def test_callback_with_a_foreign_install_cookie_is_refused(client):
    # A cookie for this provider that the real authorize call never set —
    # planted, or belonging to some other install — must not itself satisfy
    # the check. The real cookie is cleared first: unlike the old
    # single-cookie design, a foreign cookie under a different name does not
    # overwrite the real one, so this has to remove it explicitly to test
    # "some cookie is present but it is the wrong one" rather than "no
    # cookie" (already covered above) or "both are present" (which would
    # legitimately match).
    state = start_install(client)
    client.cookies.clear()
    client.cookies.set("cognee_oauth_nonce_fake_someone-elses-nonce", "1")

    response = client.get(
        f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
    )
    assert "fake=error_invalid_state" in response.headers["location"]


def test_authorize_surfaces_missing_config_as_503(client):
    with patch.object(
        _router_module,
        "make_state",
        side_effect=RuntimeError("FAKE_SIGNING_SECRET is not configured"),
    ):
        response = client.post("/api/v1/integrations/fake/authorize", follow_redirects=False)
    assert response.status_code == 503


def test_callback_with_error_param_redirects_cancelled(client):
    response = client.get(
        "/api/v1/integrations/fake/callback?error=access_denied", follow_redirects=False
    )
    assert response.status_code in (302, 307)
    assert "fake=cancelled" in response.headers["location"]


def test_callback_with_invalid_state_redirects_error(client):
    response = client.get(
        "/api/v1/integrations/fake/callback?code=abc&state=garbage", follow_redirects=False
    )
    assert "fake=error_invalid_state" in response.headers["location"]


def test_callback_surfaces_missing_frontend_url_as_503_not_a_raw_crash(client):
    # error="" so we hit the "cancelled" branch, the earliest _frontend_redirect
    # call in callback() — proves the guard applies before any real work runs,
    # not just on the success path.
    with patch.object(
        supported_integrations["fake"],
        "frontend_base_url",
        side_effect=RuntimeError("FAKE_FRONTEND_BASE_URL is not configured"),
    ):
        response = client.get(
            "/api/v1/integrations/fake/callback?error=access_denied", follow_redirects=False
        )
    assert response.status_code == 503


def test_callback_success_redirects_connected(client):
    state = start_install(client)
    with patch.object(
        _router_module,
        "complete_installation",
        new=AsyncMock(return_value=type("C", (), {"provider_account_id": "ACC1"})()),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
        )
    assert "fake=connected" in response.headers["location"]


def test_callback_cross_user_conflict_redirects_already_connected(client):
    state = start_install(client)
    with patch.object(
        _router_module,
        "complete_installation",
        new=AsyncMock(side_effect=CrossUserConflictError("ACC1")),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
        )
    assert "fake=error_already_connected" in response.headers["location"]


def test_callback_unexpected_error_redirects_exchange_failed(client):
    state = start_install(client)
    with patch.object(
        _router_module,
        "complete_installation",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
        )
    assert "fake=error_exchange_failed" in response.headers["location"]


def test_connection_status_reports_disconnected_by_default(client):
    with patch.object(
        _router_module,
        "get_active_credential_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.get("/api/v1/integrations/fake/connection")
    assert response.json() == {"connected": False}


def test_connection_status_reports_connected_with_generic_fields(client):
    fake_credential = type(
        "Cred",
        (),
        {
            "account_label": "Acme",
            "provider_account_id": "ACC1",
            "created_at": __import__("datetime").datetime(2026, 1, 1),
            "sync_status": None,
            "last_synced_at": None,
        },
    )()
    with patch.object(
        _router_module,
        "get_active_credential_for_user",
        new=AsyncMock(return_value=fake_credential),
    ):
        response = client.get("/api/v1/integrations/fake/connection")
    body = response.json()
    assert body["connected"] is True
    assert body["accountLabel"] == "Acme"
    assert body["providerAccountId"] == "ACC1"


def test_disconnect_calls_revoke_remote_and_revokes_locally(client):
    fake_credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    integration = supported_integrations["fake"]
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=fake_credential),
        ),
        patch.object(integration, "revoke_remote", new=AsyncMock()) as revoke_remote,
        patch.object(
            _router_module,
            "revoke_credential_by_account",
            new=AsyncMock(return_value=True),
        ) as revoke_local,
    ):
        response = client.delete("/api/v1/integrations/fake/connection")

    assert response.json() == {"disconnected": True}
    revoke_remote.assert_awaited_once_with(fake_credential)
    revoke_local.assert_awaited_once_with("fake", "ACC1")


def test_disconnect_with_no_active_credential_reports_false(client):
    with patch.object(
        _router_module,
        "get_active_credential_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.delete("/api/v1/integrations/fake/connection")
    assert response.json() == {"disconnected": False}


def _installed():
    """Patch the exchange out, the way every other callback test here does.

    These tests are about the nonce cookie, not about storage. Letting them
    reach the real credential store makes them depend on what earlier tests
    left in the local database, which shows up as an already-connected
    conflict on the second run rather than on the first.
    """
    return patch.object(
        _router_module,
        "complete_installation",
        new=AsyncMock(return_value=type("C", (), {"provider_account_id": "ACC1"})()),
    )


def test_two_tabs_on_one_provider_both_complete(client):
    # Two installs of the same provider open at once is ordinary. A cookie
    # shared between them, whether a single value or a list in one cookie,
    # has a second authorize call race the first's write; one cookie per
    # install removes the shared value entirely.
    first_state = start_install(client)
    second_state = start_install(client)

    with _installed():
        # The older tab finishes first, out of order.
        first = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={first_state}",
            follow_redirects=False,
        )
        second = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={second_state}",
            follow_redirects=False,
        )

    assert "fake=connected" in first.headers["location"]
    assert "fake=connected" in second.headers["location"]


def test_a_finished_install_does_not_cancel_the_tabs_still_waiting(client):
    first_state = start_install(client)
    start_install(client)
    before = pending_nonce_cookies(client)
    assert len(before) == 2

    with _installed():
        client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={first_state}",
            follow_redirects=False,
        )

    # The completed install retired its own cookie and left the other alone.
    after = pending_nonce_cookies(client)
    assert len(after) == 1
    first_nonce_cookie = f"cognee_oauth_nonce_fake_{first_state.split(':')[1]}"
    assert first_nonce_cookie not in after
    assert first_nonce_cookie in before


def test_a_callback_whose_nonce_is_unknown_leaves_the_other_tabs_alone(client):
    # Clearing the whole cookie when the nonce check fails would hand anyone a
    # way to cancel every install a user has open by replaying one stale or
    # relayed redirect. The state below is authentic and well-formed, which is
    # what makes it reach the nonce check at all rather than being thrown out
    # earlier as unparseable.
    stale_state = start_install(client)
    client.cookies.clear()

    open_state = start_install(client)
    start_install(client)
    before = pending_nonce_cookies(client)
    assert len(before) == 2

    refused = client.get(
        f"/api/v1/integrations/fake/callback?code=abc&state={stale_state}", follow_redirects=False
    )
    assert "fake=error_invalid_state" in refused.headers["location"]
    assert pending_nonce_cookies(client) == before

    # Both tabs that were genuinely waiting still complete.
    with _installed():
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={open_state}",
            follow_redirects=False,
        )
    assert "fake=connected" in response.headers["location"]


def test_an_unparseable_state_leaves_the_other_tabs_alone(client):
    # The other branch that must not clear: a state that never even parses
    # says nothing about which install, if any, it belongs to.
    start_install(client)
    before = pending_nonce_cookies(client)
    assert len(before) == 1

    refused = client.get(
        "/api/v1/integrations/fake/callback?code=abc&state=forged:1:deadbeef",
        follow_redirects=False,
    )
    assert "fake=error_invalid_state" in refused.headers["location"]
    assert pending_nonce_cookies(client) == before


def test_nothing_bounds_how_many_installs_can_be_pending_at_once(client):
    # One cookie per install rather than a shared value with a cap: nothing
    # here needs to evict an older pending install to make room for a newer
    # one, because there is no shared value for one to crowd out of.
    states = [start_install(client) for _ in range(8)]
    assert len(pending_nonce_cookies(client)) == 8

    with _installed():
        for state in states:
            response = client.get(
                f"/api/v1/integrations/fake/callback?code=abc&state={state}",
                follow_redirects=False,
            )
            assert "fake=connected" in response.headers["location"]

    assert pending_nonce_cookies(client) == []


def test_concurrent_authorize_calls_do_not_lose_a_nonce_to_each_other(client):
    # The race the single-shared-cookie design had: two authorize calls that
    # both read the jar before either writes it back would have the second
    # response's Set-Cookie overwrite the first's. Each call here sets its
    # own cookie in one write, so simulating the race — running both,
    # inspecting the jar as it stands after each response independently —
    # never has one call's write depend on having seen the other's.
    response_a = client.post("/api/v1/integrations/fake/authorize")
    response_b = client.post("/api/v1/integrations/fake/authorize")
    nonce_a = response_a.json()["authorizeUrl"].split("state=", 1)[1].split(":")[1]
    nonce_b = response_b.json()["authorizeUrl"].split("state=", 1)[1].split(":")[1]
    assert nonce_a != nonce_b

    names = {name for name in response_a.cookies} | {name for name in response_b.cookies}
    assert f"cognee_oauth_nonce_fake_{nonce_a}" in names
    assert f"cognee_oauth_nonce_fake_{nonce_b}" in names


def test_a_non_ascii_nonce_cookie_is_filtered_out_not_a_crash():
    # hmac.compare_digest raises TypeError on a non-ASCII str rather than
    # returning False, and a cookie name is attacker-influenceable (a sibling
    # subdomain, a network position on plain http). Exercised at the helper
    # level rather than over the TestClient: httpx's own header encoder
    # refuses to put a raw non-ASCII character on the wire at all (a real
    # browser would percent-encode it, which decodes back to plain ASCII
    # '%'/hex digits), so getting the literal byte into a Cookie header needs
    # tooling outside a normal HTTP client — the guard is for that case, and
    # this is how to reach it in a test.
    from cognee.api.v1.integrations.routers.get_integrations_router import (
        _install_nonce_matches,
        _pending_install_nonces,
    )

    class _FakeRequest:
        cookies = {"cognee_oauth_nonce_fake_é": "1", "cognee_oauth_nonce_fake_realnonce": "1"}

    pending = _pending_install_nonces(_FakeRequest(), "fake")
    assert pending == ["realnonce"]
    # The nonce argument here is what a real, validated state actually
    # carries — always ASCII, since it's secrets.token_urlsafe output the
    # server itself minted. Without the isascii() filter on the cookie side,
    # comparing an ASCII nonce against the unfiltered "é" candidate would
    # itself raise TypeError, before ever reaching the real one.
    assert _install_nonce_matches(_FakeRequest(), "fake", "someone-elses-nonce") is False
    assert _install_nonce_matches(_FakeRequest(), "fake", "realnonce") is True
