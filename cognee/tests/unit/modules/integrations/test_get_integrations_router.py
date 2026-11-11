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


# The forget package re-exports the function under the submodule's name, so a
# dotted-string patch target resolves to the function on Python 3.10.
_forget_module = importlib.import_module("cognee.api.v1.forget.forget")
_ingest_dlt_module = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


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
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=None),
        ),
        patch.object(_router_module, "get_ended_connection", new=AsyncMock(return_value=None)),
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


def _connected_credential():
    return type(
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


def test_connection_status_reports_the_dataset_of_any_provider_that_names_one(client):
    integration = supported_integrations["fake"]
    summary = AsyncMock(return_value=("dataset-1", 7))
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=_connected_credential()),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch("cognee.modules.integrations.ingestion.dataset_summary", summary),
    ):
        body = client.get("/api/v1/integrations/fake/connection").json()

    assert body["datasetId"] == "dataset-1"
    assert body["storedItems"] == 7
    assert summary.await_args.args[1] == "fake_acme"


def test_connection_status_has_no_dataset_for_a_provider_that_names_none(client):
    summary = AsyncMock()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=_connected_credential()),
        ),
        patch("cognee.modules.integrations.ingestion.dataset_summary", summary),
    ):
        body = client.get("/api/v1/integrations/fake/connection").json()

    summary.assert_not_awaited()
    assert body.get("datasetId") is None


def _ended_credential(revoked_by):
    return type(
        "Cred",
        (),
        {
            "account_label": "Acme",
            "provider_account_id": "ACC1",
            "revoked_by": revoked_by,
            "revoked_at": __import__("datetime").datetime(2026, 11, 1, 12, 0),
        },
    )()


@pytest.mark.parametrize("account_id", [None, "ACC1"])
def test_connection_status_tells_the_owner_who_ended_the_connection(client, account_id):
    """An uninstall or a rejected token ends a connection without its owner; the page
    says who ended it, when, and where the data stays."""
    integration = supported_integrations["fake"]
    ended = AsyncMock(return_value=_ended_credential("provider"))
    with (
        patch.object(
            _router_module, "get_active_credential_for_user", new=AsyncMock(return_value=None)
        ),
        patch.object(_router_module, "get_ended_connection", new=ended),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch(
            "cognee.modules.integrations.ingestion.dataset_summary",
            AsyncMock(return_value=("dataset-1", 7)),
        ),
    ):
        query = f"?account_id={account_id}" if account_id else ""
        body = client.get(f"/api/v1/integrations/fake/connection{query}").json()

    ended.assert_awaited_once_with(USER_ID, "fake", account_id)
    assert body == {
        "connected": False,
        "datasetId": "dataset-1",
        "storedItems": 7,
        "accountLabel": "Acme",
        "providerAccountId": "ACC1",
        "disconnectedBy": "provider",
        "disconnectedAt": "2026-11-01T12:00:00Z",
    }


def test_connection_status_of_a_connected_user_reads_no_ended_connection(client):
    ended = AsyncMock()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=_connected_credential()),
        ),
        patch.object(_router_module, "get_ended_connection", new=ended),
    ):
        assert client.get("/api/v1/integrations/fake/connection").json()["connected"] is True

    ended.assert_not_awaited()


def test_connection_status_of_an_unknown_named_account_is_still_a_404(client):
    with (
        patch.object(
            _router_module, "get_active_credential_for_user", new=AsyncMock(return_value=None)
        ),
        patch.object(_router_module, "get_ended_connection", new=AsyncMock(return_value=None)),
    ):
        response = client.get("/api/v1/integrations/fake/connection?account_id=NOPE")

    assert response.status_code == 404


def test_disconnect_with_delete_data_forgets_the_providers_dataset(client):
    integration = supported_integrations["fake"]
    forget = AsyncMock()
    related = AsyncMock()
    credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=credential),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(integration, "forget_related_data", related),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
        patch.object(_forget_module, "forget", forget),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    assert response.json() == {"disconnected": True, "remoteRevokeSkipped": False}
    assert forget.await_args.kwargs["dataset"] == "fake_acme"
    # What the provider's app stored for the caller outside the dataset goes with it.
    related.assert_awaited_once()
    assert related.await_args.args[0] is credential
    assert related.await_args.args[1] is forget.await_args.kwargs["user"]


def test_disconnect_with_delete_data_is_refused_without_a_dataset(client):
    with patch.object(
        _router_module,
        "get_active_credential_for_user",
        new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    assert response.status_code == 400


# --- a provider that cannot read null as "all resources" ---
#
# The picker stores a null selection as "everything", and listing shows every
# resource selected for it. A provider whose sync has no such meaning (Google
# Calendar) must refuse the null at the boundary instead of storing it.


def put_resources(client, integration, body, *, allows_all, monkeypatch):
    update = AsyncMock(return_value=type("Cred", (), {"provider_metadata": {"picked": body}})())
    monkeypatch.setattr(integration, "resource_selection_key", "picked")
    monkeypatch.setattr(integration, "resource_selection_allows_all", allows_all)
    monkeypatch.setattr(
        _router_module,
        "get_active_credential_for_user",
        AsyncMock(
            return_value=type(
                "Cred", (), {"provider_account_id": "ACC1", "provider_metadata": {}}
            )()
        ),
    )

    async def change_provider_metadata(provider, account_id, change):
        return await update(provider, account_id, change({}))

    monkeypatch.setattr(_router_module, "change_provider_metadata", change_provider_metadata)
    response = client.put("/api/v1/integrations/fake/resources", json={"resourceIds": body})
    return response, update


def test_a_null_selection_is_refused_for_a_provider_that_cannot_select_all(client, monkeypatch):
    response, update = put_resources(
        client, supported_integrations["fake"], None, allows_all=False, monkeypatch=monkeypatch
    )

    assert response.status_code == 422
    assert "explicit list" in response.json()["detail"]
    update.assert_not_awaited()


def test_a_null_selection_is_still_stored_for_a_provider_that_allows_all(client, monkeypatch):
    response, update = put_resources(
        client, supported_integrations["fake"], None, allows_all=True, monkeypatch=monkeypatch
    )

    assert response.status_code == 200
    update.assert_awaited_once_with("fake", "ACC1", {"picked": None})


def test_a_list_selection_is_stored_for_a_provider_that_cannot_select_all(client, monkeypatch):
    response, update = put_resources(
        client,
        supported_integrations["fake"],
        ["a", "a", "b"],
        allows_all=False,
        monkeypatch=monkeypatch,
    )

    assert response.status_code == 200
    update.assert_awaited_once_with("fake", "ACC1", {"picked": ["a", "b"]})


# A blank id is never a real resource. Stored, it makes the sync of every provider
# that checks the selection shape fail until the user saves again.
@pytest.mark.parametrize("allows_all", [True, False])
@pytest.mark.parametrize("body", [[""], [" "], ["a", "\t"], ["a", "  \n"]])
def test_a_blank_resource_id_is_refused_and_nothing_is_stored(
    client, monkeypatch, allows_all, body
):
    response, update = put_resources(
        client, supported_integrations["fake"], body, allows_all=allows_all, monkeypatch=monkeypatch
    )

    assert response.status_code == 422
    assert "blank" in response.json()["detail"]
    update.assert_not_awaited()


def test_ids_are_stored_as_given_and_deduplicated(client, monkeypatch):
    response, update = put_resources(
        client,
        supported_integrations["fake"],
        ["b", " a ", "b"],
        allows_all=True,
        monkeypatch=monkeypatch,
    )

    assert response.status_code == 200
    update.assert_awaited_once_with("fake", "ACC1", {"picked": ["b", " a "]})


def _sync_credential(account_id):
    return type(
        "Cred",
        (),
        {
            "id": uuid4(),
            "user_id": USER_ID,
            "workspace_id": None,
            "provider_account_id": account_id,
        },
    )()


def test_a_manual_sync_is_not_accepted_while_one_is_running(client):
    from cognee.modules.integrations import ingestion

    credential = _sync_credential("ACC1")
    integration = supported_integrations["fake"]
    sync_now = AsyncMock()
    with (
        patch.object(
            _router_module, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
        ),
        patch.object(_router_module, "require_sync_not_paused", new=AsyncMock()),
        patch.object(integration, "sync_now", sync_now),
    ):
        ingestion._running_syncs.add(("fake", "ACC1"))
        try:
            running = client.post("/api/v1/integrations/fake/sync").json()
        finally:
            ingestion._running_syncs.discard(("fake", "ACC1"))
        idle = client.post("/api/v1/integrations/fake/sync").json()

    assert running == {"accepted": False}
    assert idle == {"accepted": True}
    sync_now.assert_called_once_with(credential)


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

    assert response.json() == {"disconnected": True, "remoteRevokeSkipped": False}
    revoke_remote.assert_awaited_once_with(fake_credential)
    revoke_local.assert_awaited_once_with("fake", "ACC1", revoked_by="owner")


def test_disconnect_records_the_owner_before_asking_the_provider_to_revoke(client):
    """The provider's echo of our own revoke (Slack ``tokens_revoked``, a refresh that
    hits the revoked token) must find the connection already ended by its owner."""
    fake_credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    integration = supported_integrations["fake"]
    calls = []
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=fake_credential),
        ),
        patch.object(
            integration,
            "revoke_remote",
            new=AsyncMock(side_effect=lambda *_: calls.append("remote")),
        ),
        patch.object(
            _router_module,
            "revoke_credential_by_account",
            new=AsyncMock(side_effect=lambda *_, **__: calls.append("local") or True),
        ),
    ):
        client.delete("/api/v1/integrations/fake/connection")

    assert calls == ["local", "remote"]


def test_disconnect_reports_a_skipped_remote_revoke_while_outbound_http_is_disabled(
    client, monkeypatch
):
    """SDK-832: the local connection goes, but the token stays live at the provider
    and the caller is told so instead of reading ``disconnected: true`` as revoked."""
    fake_credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    integration = supported_integrations["fake"]
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")
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

    assert response.json() == {"disconnected": True, "remoteRevokeSkipped": True}
    revoke_remote.assert_not_awaited()
    revoke_local.assert_awaited_once_with("fake", "ACC1", revoked_by="owner")


def test_disconnect_with_no_active_credential_reports_false(client):
    with patch.object(
        _router_module,
        "get_active_credential_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.delete("/api/v1/integrations/fake/connection")
    assert response.json() == {"disconnected": False, "remoteRevokeSkipped": False}


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


@pytest.mark.asyncio
async def test_spawned_integration_work_is_registered_for_the_shutdown_drain():
    import asyncio

    from cognee.infrastructure import background_tasks
    from cognee.modules.integrations.background import spawn_background

    gate, finished = asyncio.Event(), []

    async def work():
        await gate.wait()
        finished.append(True)

    before = set(background_tasks._BACKGROUND_TASKS)
    spawn_background(work(), description="test work")
    assert len(background_tasks._BACKGROUND_TASKS - before) == 1
    gate.set()
    assert await background_tasks.wait_for_background_tasks(timeout=2)
    assert finished == [True]


class _Connection:
    def __init__(self, account_id, label="Workspace"):
        self.provider_account_id = account_id
        self.account_label = label
        self.created_at = None


def test_the_connections_of_a_provider_are_listed_with_their_account_ids(client):
    connections = [_Connection("ACC2", "Second"), _Connection("ACC1", "First")]
    with patch.object(
        _router_module,
        "list_active_credentials_for_provider",
        new=AsyncMock(return_value=connections),
    ) as listing:
        response = client.get("/api/v1/integrations/fake/connections")

    assert response.status_code == 200
    assert [item["providerAccountId"] for item in response.json()["connections"]] == [
        "ACC2",
        "ACC1",
    ]
    listing.assert_awaited_once_with(USER_ID, "fake")


_ACCOUNT_ROUTES = [
    ("get", "/api/v1/integrations/fake/connection"),
    ("get", "/api/v1/integrations/fake/resources"),
    ("put", "/api/v1/integrations/fake/resources"),
    ("post", "/api/v1/integrations/fake/sync"),
    ("delete", "/api/v1/integrations/fake/connection"),
    ("get", "/api/v1/integrations/fake/sync/policy"),
    ("put", "/api/v1/integrations/fake/sync/policy"),
    ("delete", "/api/v1/integrations/fake/sync/policy"),
    ("post", "/api/v1/integrations/fake/sync/policy/pause"),
    ("post", "/api/v1/integrations/fake/sync/policy/resume"),
]


def _call(client, method, url, **params):
    body = {"interval_minutes": 60} if url.endswith("/sync/policy") else {"resource_ids": ["a"]}
    kwargs = {"json": body} if method == "put" else {}
    return getattr(client, method)(url, params=params, **kwargs)


@pytest.mark.parametrize("method, url", _ACCOUNT_ROUTES)
def test_a_named_account_that_is_not_the_users_is_a_404_not_another_account(client, method, url):
    integration = supported_integrations["fake"]
    integration.resource_selection_key = "picked"
    lookup = AsyncMock(return_value=None)
    try:
        with patch.object(_router_module, "get_active_credential_for_user", new=lookup):
            response = _call(client, method, url, account_id="SOMEONE-ELSES")
    finally:
        integration.resource_selection_key = None

    # The body tells "that account" from "no connection at all": the other 404s of these
    # routes ("fake is not connected") would pass for the second.
    assert response.status_code == 404
    assert response.json()["detail"] == "fake account is not connected"
    lookup.assert_awaited_once_with(USER_ID, "fake", "SOMEONE-ELSES")


@pytest.mark.parametrize("method, url", _ACCOUNT_ROUTES)
def test_without_an_account_id_the_lookup_is_the_newest_connection_as_before(client, method, url):
    integration = supported_integrations["fake"]
    integration.resource_selection_key = "picked"
    lookup = AsyncMock(return_value=None)
    try:
        with patch.object(_router_module, "get_active_credential_for_user", new=lookup):
            response = _call(client, method, url)
    finally:
        integration.resource_selection_key = None

    # No connection at all keeps each route's own answer: a status, a 404 or a false.
    assert response.status_code in (200, 404)
    lookup.assert_awaited_once_with(USER_ID, "fake", None)


def test_a_manual_sync_runs_for_the_named_account(client):
    credential = _sync_credential("ACC2")
    integration = supported_integrations["fake"]
    sync_now = AsyncMock()
    lookup = AsyncMock(return_value=credential)
    with (
        patch.object(_router_module, "get_active_credential_for_user", new=lookup),
        patch.object(_router_module, "require_sync_not_paused", new=AsyncMock()),
        patch.object(integration, "sync_now", sync_now),
    ):
        response = client.post("/api/v1/integrations/fake/sync", params={"account_id": "ACC2"})

    assert response.json() == {"accepted": True}
    lookup.assert_awaited_once_with(USER_ID, "fake", "ACC2")
    sync_now.assert_called_once_with(credential)


def test_disconnect_revokes_the_named_account_only(client):
    credential = type("Cred", (), {"provider_account_id": "ACC2"})()
    integration = supported_integrations["fake"]
    with (
        patch.object(
            _router_module, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
        ),
        patch.object(integration, "revoke_remote", AsyncMock()) as revoke_remote,
        patch.object(_router_module, "revoke_credential_by_account", AsyncMock()) as revoke_local,
    ):
        response = client.delete(
            "/api/v1/integrations/fake/connection", params={"account_id": "ACC2"}
        )

    assert response.json() == {"disconnected": True, "remoteRevokeSkipped": False}
    revoke_remote.assert_awaited_once_with(credential)
    revoke_local.assert_awaited_once_with("fake", "ACC2", revoked_by="owner")


@pytest.mark.parametrize("method, url", _ACCOUNT_ROUTES)
@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_account_id_names_nothing_and_is_refused(client, method, url, blank):
    integration = supported_integrations["fake"]
    integration.resource_selection_key = "picked"
    lookup = AsyncMock(return_value=None)
    try:
        with patch.object(_router_module, "get_active_credential_for_user", new=lookup):
            response = _call(client, method, url, account_id=blank)
    finally:
        integration.resource_selection_key = None

    assert response.status_code == 422
    lookup.assert_not_awaited()


def test_a_connection_without_a_label_leaves_the_field_out(client):
    connections = [_Connection("ACC1", None)]
    with patch.object(
        _router_module,
        "list_active_credentials_for_provider",
        new=AsyncMock(return_value=connections),
    ):
        body = client.get("/api/v1/integrations/fake/connections").json()

    assert body == {"connections": [{"providerAccountId": "ACC1"}]}


def test_connect_and_sync_are_refused_while_outbound_http_is_disabled(client, monkeypatch):
    """SDK-832: the flag that disables cognee's outbound HTTP disables the connectors too."""
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")
    credential = type("Cred", (), {"provider_account_id": "ACC1"})()

    authorize = client.post("/api/v1/integrations/fake/authorize")
    with patch.object(
        _router_module, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
    ):
        sync = client.post("/api/v1/integrations/fake/sync")

        resources = client.get("/api/v1/integrations/fake/resources")

    assert authorize.status_code == 503
    assert "ALLOW_HTTP_REQUESTS" in authorize.json()["detail"]
    assert sync.status_code == 503
    assert resources.status_code == 503
    # No install was started: nothing for a callback to match.
    assert not any(name.startswith("cognee_oauth_nonce_") for name in authorize.cookies)


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    """No connection is paused unless a test says so (the pause lives in the database)."""
    monkeypatch.setattr(
        "cognee.modules.integrations.ingestion.sync_is_paused", AsyncMock(return_value=False)
    )


@pytest.fixture(autouse=True)
def no_staging_drop(monkeypatch):
    """A real drop takes the process-wide staging lock and deletes files under dlt's
    pipelines directory; a drop that outlives the test client's loop would keep that lock
    for every later dlt test. Tests that are about the drop pass their own."""
    monkeypatch.setattr(_ingest_dlt_module, "drop_dlt_dataset", AsyncMock())


def test_disconnect_with_delete_data_drops_the_staging_copy_too(client):
    """``forget`` leaves dlt's staging database, state and load packages, which hold the
    text of every synced row, so the same owner's next connect and sync built every
    document again (measured on Slack, 2026-11-10). They are deleted after the dataset."""
    integration = supported_integrations["fake"]
    calls = []
    forget = AsyncMock(side_effect=lambda **_: calls.append("forget"))
    drop = AsyncMock(side_effect=lambda name: calls.append(("drop", name)))
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(integration, "forget_related_data", new=AsyncMock()),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
        patch.object(_forget_module, "forget", forget),
        patch.object(_ingest_dlt_module, "drop_dlt_dataset", drop),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    assert response.status_code == 200, response.text
    assert calls == ["forget", ("drop", "fake_acme")]


def test_a_dataset_that_could_not_be_deleted_keeps_its_staging_copy(client):
    integration = supported_integrations["fake"]
    drop = AsyncMock()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(integration, "forget_related_data", new=AsyncMock()),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
        patch.object(_forget_module, "forget", AsyncMock(side_effect=RuntimeError("store down"))),
        patch.object(_ingest_dlt_module, "drop_dlt_dataset", drop),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    assert response.status_code == 502
    drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_with_delete_data_finishes_when_the_dataset_was_never_created():
    """A connection that never synced has no dataset, which is nothing to delete.

    The real ``forget`` runs here, with a real user (the other tests mock it): the dataset
    a Slack connection names does not exist until its first sync, since the selection
    starts empty. The credential is revoked before the deletion, so a failure here is
    final: a retry finds no active connection, and the provider's related data (Slack's
    remembered messages) has no other way out.
    """
    from httpx import ASGITransport, AsyncClient

    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: user

    before = dict(supported_integrations)
    supported_integrations.clear()
    integration = _FakeIntegration()
    use_integration(integration)
    credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    related = AsyncMock()
    try:
        with (
            patch.object(
                _router_module,
                "get_active_credential_for_user",
                new=AsyncMock(return_value=credential),
            ),
            patch.object(
                integration, "dataset_name", return_value=f"fake_never_synced_{uuid4().hex}"
            ),
            patch.object(integration, "revoke_remote", new=AsyncMock()),
            patch.object(integration, "forget_related_data", related),
            patch.object(
                _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as http:
                response = await http.delete(
                    "/api/v1/integrations/fake/connection?delete_data=true"
                )
    finally:
        supported_integrations.clear()
        supported_integrations.update(before)

    assert response.status_code == 200, response.text
    assert response.json() == {"disconnected": True, "remoteRevokeSkipped": False}
    related.assert_awaited_once()
    assert related.await_args.args[0] is credential


def test_disconnect_forgets_the_related_data_even_when_the_dataset_cannot_be_deleted(client):
    integration = supported_integrations["fake"]
    credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    related = AsyncMock()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=credential),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(integration, "forget_related_data", related),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
        patch.object(_forget_module, "forget", AsyncMock(side_effect=RuntimeError("store down"))),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    # The dataset's failure is still reported, but it did not keep the app's data.
    assert response.status_code == 502
    related.assert_awaited_once()


def test_disconnect_without_delete_data_leaves_the_providers_related_data(client):
    integration = supported_integrations["fake"]
    related = AsyncMock()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
        ),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(integration, "forget_related_data", related),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
    ):
        response = client.delete("/api/v1/integrations/fake/connection")

    assert response.json() == {"disconnected": True, "remoteRevokeSkipped": False}
    related.assert_not_awaited()


def test_disconnect_reports_related_data_that_could_not_be_deleted(client):
    integration = supported_integrations["fake"]
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(
            integration, "forget_related_data", new=AsyncMock(side_effect=RuntimeError("boom"))
        ),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
        patch.object(_forget_module, "forget", AsyncMock()),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    assert response.status_code == 502


def save_selection(client, monkeypatch, *, stored, body):
    """PUT ``body`` over a connection whose stored selection is ``stored``."""
    integration = supported_integrations["fake"]
    credential = type(
        "Cred", (), {"provider_account_id": "ACC1", "provider_metadata": {"picked": stored}}
    )()
    updated = type(
        "Cred", (), {"provider_account_id": "ACC1", "provider_metadata": {"picked": body}}
    )()
    sync_now = AsyncMock()
    monkeypatch.setattr(integration, "resource_selection_key", "picked")
    monkeypatch.setattr(integration, "resource_selection_allows_all", False)
    monkeypatch.setattr(integration, "sync_now", sync_now)
    monkeypatch.setattr(
        _router_module, "get_active_credential_for_user", AsyncMock(return_value=credential)
    )

    async def change_provider_metadata(provider, account_id, change):
        change({"picked": stored})
        return updated

    monkeypatch.setattr(_router_module, "change_provider_metadata", change_provider_metadata)
    response = client.put("/api/v1/integrations/fake/resources", json={"resourceIds": body})
    return response, sync_now, updated


@pytest.mark.parametrize(
    ("stored", "body"),
    [(["a"], ["a", "b"]), (["a", "b"], ["a"]), (["a", "b"], [])],
    ids=["a_scope_added", "a_scope_removed", "everything_deselected"],
)
def test_a_changed_selection_starts_a_sync_at_once(client, monkeypatch, stored, body):
    """What the user picked shows, and what they removed leaves, without waiting for the
    next scheduled sync; deselecting everything included, which no schedule would run."""
    response, sync_now, updated = save_selection(client, monkeypatch, stored=stored, body=body)

    assert response.status_code == 200
    sync_now.assert_called_once_with(updated)


def test_a_changed_selection_of_a_paused_connection_is_saved_and_not_synced(client, monkeypatch):
    """The pause (SDK-1434) holds a selection save too, for every provider."""
    monkeypatch.setattr(
        "cognee.modules.integrations.ingestion.sync_is_paused", AsyncMock(return_value=True)
    )
    response, sync_now, _ = save_selection(client, monkeypatch, stored=["a"], body=["a", "b"])

    assert response.status_code == 200
    sync_now.assert_not_called()


def test_an_unchanged_selection_starts_no_sync(client, monkeypatch):
    response, sync_now, _ = save_selection(client, monkeypatch, stored=["a"], body=["a"])

    assert response.status_code == 200
    sync_now.assert_not_called()


def test_a_selection_saved_during_a_sync_runs_once_more_after_it(client, monkeypatch):
    from cognee.modules.integrations import ingestion

    ingestion._running_syncs.add(("fake", "ACC1"))
    try:
        response, sync_now, _ = save_selection(client, monkeypatch, stored=["a"], body=[])
        # The running sync read the old selection; a second one would be dropped.
        sync_now.assert_not_called()
        assert ingestion.request_rerun_if_running("fake", "ACC1")
    finally:
        ingestion._running_syncs.discard(("fake", "ACC1"))
        ingestion._rerun_requested.discard(("fake", "ACC1"))
    assert response.status_code == 200


@pytest.mark.parametrize("revoked_by", ["provider", "owner"])
def test_connection_status_says_when_the_data_of_an_ended_connection_is_deleted(client, revoked_by):
    """A provider can schedule the deletion of an ended connection's data (Slack:
    14 days after an uninstall, also after the owner disconnected keeping it); the page must
    be able to warn the owner before it happens and say when it ran."""
    from datetime import datetime, timezone

    integration = supported_integrations["fake"]
    ended = type(
        "Cred",
        (),
        {
            "account_label": "Acme",
            "provider_account_id": "ACC1",
            "revoked_by": revoked_by,
            "revoked_at": datetime(2026, 11, 7, tzinfo=timezone.utc),
        },
    )()
    with (
        patch.object(
            _router_module, "get_active_credential_for_user", new=AsyncMock(return_value=None)
        ),
        patch.object(_router_module, "get_ended_connection", new=AsyncMock(return_value=ended)),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(
            integration,
            "data_deleted_after",
            return_value=datetime(2026, 11, 21, tzinfo=timezone.utc),
        ),
        patch.object(
            integration,
            "data_deleted_at",
            return_value=datetime(2026, 11, 21, 1, tzinfo=timezone.utc),
        ),
        patch(
            "cognee.modules.integrations.ingestion.dataset_summary",
            new=AsyncMock(return_value=("dataset-1", 12)),
        ),
    ):
        body = client.get("/api/v1/integrations/fake/connection").json()

    assert body["disconnectedBy"] == revoked_by
    assert body["dataDeletedAfter"] == "2026-11-21T00:00:00Z"
    assert body["dataDeletedAt"] == "2026-11-21T01:00:00Z"


def test_start_sends_the_browser_to_the_providers_authorize_page(client):
    """A link opened from the provider (Slack's ``@cognee link``) starts the install in the
    browser: the same nonce cookie as ``authorize``, then a redirect instead of JSON."""
    response = client.get("/api/v1/integrations/fake/start", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].startswith("https://fake.example/authorize?state=")
    names = [name for name in response.cookies if name.startswith("cognee_oauth_nonce_fake_")]
    assert len(names) == 1


def test_start_needs_a_logged_in_user():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")

    response = TestClient(app).get("/api/v1/integrations/fake/start", follow_redirects=False)

    assert response.status_code == 401


def test_a_selection_is_compared_with_what_is_stored_when_it_is_written(client, monkeypatch):
    """Two saves at once (X to Y, then Y back to X): the second read X before the first wrote
    Y, so against its own copy nothing changed and no sync started, while the sync the first
    started may already have read Y. The comparison is made under the connection's lock."""
    integration = supported_integrations["fake"]
    stale = type(
        "Cred", (), {"provider_account_id": "ACC1", "provider_metadata": {"picked": ["x"]}}
    )()
    updated = type(
        "Cred", (), {"provider_account_id": "ACC1", "provider_metadata": {"picked": ["x"]}}
    )()

    async def change_provider_metadata(provider, account_id, change):
        patch_ = change({"picked": ["y"]})  # what the concurrent save left
        assert patch_ == {"picked": ["x"]}
        return updated

    sync_now = AsyncMock()
    monkeypatch.setattr(integration, "resource_selection_key", "picked")
    monkeypatch.setattr(integration, "resource_selection_allows_all", False)
    monkeypatch.setattr(integration, "sync_now", sync_now)
    monkeypatch.setattr(
        _router_module, "get_active_credential_for_user", AsyncMock(return_value=stale)
    )
    monkeypatch.setattr(_router_module, "change_provider_metadata", change_provider_metadata)

    response = client.put("/api/v1/integrations/fake/resources", json={"resourceIds": ["x"]})

    assert response.status_code == 200
    sync_now.assert_called_once_with(updated)


def test_deleting_the_data_waits_for_no_running_sync(client):
    """A sync running while its data is deleted could store a scope it already started into
    the dataset just deleted; the owner is told to retry once it has finished."""
    integration = supported_integrations["fake"]
    revoke = AsyncMock(return_value=True)
    forget = AsyncMock()
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(_router_module, "revoke_credential_by_account", new=revoke),
        patch.object(_forget_module, "forget", forget),
        patch("cognee.modules.integrations.ingestion.sync_is_running", return_value=True),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")

    assert response.status_code == 409, response.text
    revoke.assert_not_awaited()
    forget.assert_not_awaited()


def _disconnect_deleting_data(client, monkeypatch, drop):
    """DELETE the connection with ``delete_data``; the calls the deletion made, in order."""
    integration = supported_integrations["fake"]
    calls = []
    monkeypatch.setattr(_ingest_dlt_module, "drop_dlt_dataset", drop)
    with (
        patch.object(
            _router_module,
            "get_active_credential_for_user",
            new=AsyncMock(return_value=type("Cred", (), {"provider_account_id": "ACC1"})()),
        ),
        patch.object(integration, "dataset_name", return_value="fake_acme"),
        patch.object(integration, "revoke_remote", new=AsyncMock()),
        patch.object(
            integration,
            "forget_related_data",
            new=AsyncMock(side_effect=lambda *_: calls.append("related")),
        ),
        patch.object(
            _router_module, "revoke_credential_by_account", new=AsyncMock(return_value=True)
        ),
        patch.object(
            _forget_module, "forget", AsyncMock(side_effect=lambda **_: calls.append("forget"))
        ),
    ):
        response = client.delete("/api/v1/integrations/fake/connection?delete_data=true")
    return response, calls


def test_a_staging_copy_that_cannot_be_dropped_is_reported(client, monkeypatch):
    """The staging copy holds every synced row's text and would bring the documents back on
    the next connect, so the owner is told when it was not deleted."""

    async def drop(name):
        raise OSError("disk full")

    response, calls = _disconnect_deleting_data(client, monkeypatch, drop)

    assert response.status_code == 502, response.text
    assert calls == ["forget", "related"]


def test_a_staging_drop_still_waiting_does_not_hold_the_disconnect(client, monkeypatch):
    """The drop waits for the lock every dlt run in the process holds, which another
    connection's backfill can keep for minutes; past a short wait it finishes in the
    background and the disconnect answers."""
    import asyncio
    import time

    monkeypatch.setattr(_router_module, "STAGING_DROP_WAIT_SECONDS", 0.05)

    async def drop(name):
        await asyncio.sleep(5)

    started = time.monotonic()
    response, calls = _disconnect_deleting_data(client, monkeypatch, drop)

    assert response.status_code == 200, response.text
    assert time.monotonic() - started < 4
    assert calls == ["forget", "related"]


@pytest.mark.asyncio
async def test_a_disconnect_cancelled_while_its_drop_waits_still_finishes_the_drop(monkeypatch):
    """A request cancelled during the wait (a shutdown past its grace period) leaves the drop
    to the background work the shutdown drains, where a failure is logged."""
    import asyncio

    spawned = []
    monkeypatch.setattr(
        _router_module,
        "spawn_background",
        lambda coroutine, *, description: spawned.append(asyncio.ensure_future(coroutine)),
    )
    released = asyncio.Event()
    dropped = []

    async def drop(name):
        await released.wait()
        dropped.append(name)

    waiting = asyncio.ensure_future(_router_module._drop_staging_copy("fake", "fake_acme", drop))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    released.set()
    await asyncio.gather(*spawned)
    assert dropped == ["fake_acme"]
    assert len(spawned) == 1
