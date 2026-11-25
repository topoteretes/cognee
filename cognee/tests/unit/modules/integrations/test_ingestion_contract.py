"""The provider-neutral ingestion contract and the persisted Google resource names."""

import re
from types import SimpleNamespace
from uuid import UUID

import pytest

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.google import ingestion as google_ingestion

USER_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CREDENTIAL = SimpleNamespace(user_id=USER_ID, provider_account_id="acct-1")


@pytest.mark.parametrize(
    ("provider", "scope", "expected"),
    [
        ("gmail", "", "gmail_messages_e8ba1b716982492b069485f8"),
        ("google_drive", "folder-1", "google_drive_files_c0d03acdd392958b38a3b6c4"),
        ("google_drive", "folder-é", "google_drive_files_4a926936081ee8448fba9596"),
        ("google_calendar", "", "google_calendar_events_e8ba1b716982492b069485f8"),
        (
            "google_calendar",
            "team@group.calendar.google.com",
            "google_calendar_events_834ee7babdf83d4d429c35f3",
        ),
    ],
)
def test_google_resource_names_are_pinned(provider, scope, expected):
    # These names key the DLT tables and cursors tenants already have. A changed
    # name orphans the table, resets the cursor and re-cognifies the corpus.
    # The literals are deliberate: never compute them with resource_name itself.
    assert ingestion.resource_name(provider, CREDENTIAL, scope) == expected
    assert google_ingestion.resource_name(provider, CREDENTIAL, scope) == expected


def test_linear_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Google pins: the literal is deliberate. The Cloud control
    # plane only tracks tables that start with the provider prefix, and the name
    # must survive dlt's table-name normalization unchanged.
    name = ingestion.resource_name("linear", CREDENTIAL, "team-uuid")
    assert name == "linear_7b1da76d68615244608f91ed"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("linear").__module__ == (
        "cognee.tasks.ingestion.connectors.linear"
    )


def test_apollo_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Linear pin: the literal is deliberate. One Apollo team is one
    # connection, so there is no scope, and the Cloud control plane only tracks tables that
    # start with the provider prefix.
    name = ingestion.resource_name("apollo", CREDENTIAL)
    assert name == "apollo_e8ba1b716982492b069485f8"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("apollo").__module__ == (
        "cognee.tasks.ingestion.connectors.apollo"
    )


def test_slack_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Linear pin: the literal is deliberate. One resource is one channel,
    # so the channel id is the scope, and the Cloud control plane only tracks tables that
    # start with the provider prefix.
    name = ingestion.resource_name("slack", CREDENTIAL, "C0123ABCD")
    assert name == "slack_channel_02eaa4a14233ea3cb69a0f49"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("slack").__module__ == (
        "cognee.tasks.ingestion.connectors.slack"
    )


def test_slack_service_is_built_from_the_connections_access_token():
    from cognee.tasks.ingestion.connectors.slack import SlackClient

    assert isinstance(ingestion.build_service("slack", "xoxb-a-bot-token"), SlackClient)


def test_apollo_service_is_built_from_the_connections_api_key():
    from cognee.tasks.ingestion.connectors.apollo import ApolloClient

    assert isinstance(ingestion.build_service("apollo", "an-apollo-api-key"), ApolloClient)


def test_asana_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Apollo pin: the literal is deliberate. One connection is one access token,
    # so there is no scope, and the Cloud control plane only tracks tables that start with the
    # provider prefix.
    name = ingestion.resource_name("asana", CREDENTIAL)
    assert name == "asana_e8ba1b716982492b069485f8"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("asana").__module__ == (
        "cognee.tasks.ingestion.connectors.asana"
    )


def test_asana_service_is_built_from_the_connections_access_token():
    from cognee.tasks.ingestion.connectors.asana import AsanaClient

    assert isinstance(ingestion.build_service("asana", "an-asana-access-token"), AsanaClient)


def test_evernote_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Asana pin: the literal is deliberate. One connection is one account, so
    # there is no scope, and the Cloud control plane only tracks tables that start with the
    # provider prefix.
    name = ingestion.resource_name("evernote", CREDENTIAL)
    assert name == "evernote_e8ba1b716982492b069485f8"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("evernote").__module__ == (
        "cognee.tasks.ingestion.connectors.evernote"
    )


def test_evernote_service_is_built_from_a_token_without_a_network_call():
    """Evernote is SDK-only (OAuth 1.0a): the builder takes the token a caller already holds."""
    from cognee.tasks.ingestion.connectors.evernote import EvernoteApi

    service = ingestion.build_service("evernote", "S=s1:U=1:E=1:C=1:P=1:A=app:V=2:H=secret")
    assert isinstance(service, EvernoteApi)
    assert "secret" not in repr(service)


def test_vercel_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Linear pin: the literal is deliberate. One project is one scope (one
    # source, one table), and the Cloud control plane only tracks tables that start with the
    # provider prefix.
    name = ingestion.resource_name("vercel", CREDENTIAL, "prj_web")
    assert name == "vercel_873f73768f80dfb52a91ccf7"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    another_owner = SimpleNamespace(
        user_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"), provider_account_id="acct-1"
    )
    assert ingestion.resource_name("vercel", another_owner, "prj_web") != name
    assert ingestion.source_factory("vercel").__module__ == (
        "cognee.tasks.ingestion.connectors.vercel"
    )


def test_vercel_service_built_from_an_oauth_access_token_sends_it_as_a_bearer_header(
    monkeypatch,
):
    # Cloud hands the service builder the Vercel Integration's OAuth access token. The service
    # alone must be enough for the source, and the token goes out as ``Authorization: Bearer``
    # on the request requests really sends (after its own auth handling).
    import requests

    from cognee.tasks.ingestion.connectors.vercel import VercelClient

    access_token = "vercel-oauth-access-token"
    sent = []

    class Answer:
        status_code = 200
        content = b'{"id": "prj_web", "name": "web", "accountId": "team_acme"}'

    def send(self, request, **kwargs):
        sent.append(request.headers.get("Authorization"))
        return Answer()

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    service = ingestion.build_service("vercel", access_token)
    assert isinstance(service, VercelClient)
    service.request("/v9/projects/{project}", ids={"project": "prj_web"})

    assert sent == [f"Bearer {access_token}"]
    assert access_token not in repr(service)
    source = ingestion.source_factory("vercel")(
        resource_name="vercel_x", project_id="prj_web", team_id="team_acme", service=service
    )
    assert source.name == "vercel_x"


def test_telegram_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Asana pin: the literal is deliberate. One connection is one bot, so there
    # is no scope. The name is the table, the dlt state key and the pipeline scope, and it must
    # pass the connector's own resource_name check.
    name = ingestion.resource_name("telegram", CREDENTIAL)
    assert name == "telegram_e8ba1b716982492b069485f8"
    assert re.fullmatch(r"[a-z][a-z0-9_]{0,62}", name)
    assert ingestion.source_factory("telegram").__module__ == (
        "cognee.tasks.ingestion.connectors.telegram"
    )


def test_telegram_service_is_built_from_a_bot_token_sent_only_in_the_path():
    # Telegram has no OAuth for bots, so a host connects it by API key: the builder takes
    # the bot token a user pasted. It goes into the request path, never into a header.
    from cognee.tasks.ingestion.connectors.telegram import TelegramBotClient

    token = "123456789:" + "A" * 35
    client = ingestion.build_service("telegram", token)
    assert isinstance(client, TelegramBotClient)
    sent = []

    class Answer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true, "result": {"id": 123456789}}'

    def capture(request):
        sent.append(request)
        return Answer()

    client._open = capture
    assert client.call("getMe") == {"id": 123456789}
    assert sent[0].full_url == f"https://api.telegram.org/bot{token}/getMe"
    assert not any(token in value for value in sent[0].headers.values())
    with pytest.raises(ValueError) as info:
        ingestion.build_service("telegram", "not-a-bot-token")
    assert "not-a-bot-token" not in str(info.value)


def test_telegram_privacy_mode_and_skip_reasons_reach_the_host_as_counts():
    """The host keeps only int counts, so a bool ``privacy_mode`` would never reach a card."""
    from cognee.tasks.ingestion.connectors import telegram_source
    from cognee.tests.unit.tasks.connectors.fake_telegram import GROUP, OTHER, FakeBot, msg

    bot = FakeBot(can_read_all=False)
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(message=msg(OTHER, 1, "b"))
    resource = telegram_source(resource_name="telegram_111", service=bot, chats=[GROUP["id"]])
    list(resource)
    counts = ingestion.source_counts(resource)
    assert counts["privacy_mode"] == 1
    assert counts["skipped"] == counts["skipped_unselected"] == 1


@pytest.mark.asyncio
async def test_a_telegram_failure_the_source_counted_is_counted_once(monkeypatch):
    """F15: the source counts a raised failure under its reason key, and the host used to add
    its own ``failed`` on top, so one failed run read as failed=2."""
    import importlib
    from unittest.mock import AsyncMock

    from cognee.tasks.ingestion.connectors.telegram import TelegramBotMismatchError

    async def failing_remember(*args, **kwargs):
        try:
            raise TelegramBotMismatchError("another bot")
        except TelegramBotMismatchError as error:
            raise RuntimeError("pipeline failed") from error

    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.remember"), "remember", failing_remember
    )
    monkeypatch.setattr("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    credential = SimpleNamespace(
        id=USER_ID, user_id=USER_ID, provider="telegram", provider_account_id="acct-1"
    )

    def make_source(scope, name, check_active):
        return SimpleNamespace(cognee_sync_stats={"failed": 1, "failed_bot_mismatch": 1})

    await ingestion.sync_scopes(
        "telegram",
        credential,
        counts,
        scopes=[""],
        dataset_name="telegram",
        make_source=make_source,
        classify_error=ingestion.error_classifier("telegram"),
    )
    assert counts["failed"] == 1
    assert counts["failed_bot_mismatch"] == 1
    assert "failed_ingestion" not in counts


@pytest.mark.asyncio
async def test_an_unexpected_telegram_failure_is_counted_once(monkeypatch):
    """F15: a KeyError while the page is planned was counted failed_provider by the source and
    failed_ingestion by the host, so the run read failed=2."""
    import importlib
    from unittest.mock import AsyncMock

    from cognee.tasks.ingestion.connectors import telegram as module
    from cognee.tests.unit.tasks.connectors.fake_telegram import GROUP, FakeBot, msg

    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    stats = module.new_stats()

    def broken(*args, **kwargs):
        raise KeyError("id")

    monkeypatch.setattr(module, "message_to_row", broken)

    async def remember(*args, **kwargs):
        rows = module.sync_rows(
            bot,
            {},
            stats,
            selection=module._selection(None, False),
            resource_name="telegram_x",
            staged_ids=list,
        )
        try:
            list(rows)
        except Exception as error:
            raise RuntimeError("pipeline failed") from error

    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.remember"), "remember", remember
    )
    monkeypatch.setattr("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    credential = SimpleNamespace(
        id=USER_ID, user_id=USER_ID, provider="telegram", provider_account_id="acct-1"
    )
    await ingestion.sync_scopes(
        "telegram",
        credential,
        counts,
        scopes=[""],
        dataset_name="telegram",
        make_source=lambda scope, name, check_active: SimpleNamespace(cognee_sync_stats=stats),
        classify_error=ingestion.error_classifier("telegram"),
    )
    assert counts["failed"] == 1
    assert counts["failed_unexpected"] == 1
    assert "failed_ingestion" not in counts


def test_asana_service_sends_the_token_as_a_bearer_header_and_shows_it_nowhere(monkeypatch):
    from cognee.tasks.ingestion.connectors.asana import AsanaAuthError

    token = "SECRET-asana-token-123"
    sent: list[dict] = []

    class Rejected:
        status_code = 401
        headers: dict = {}

        @staticmethod
        def json():
            return {"errors": [{"message": f"Not Authorized: {token}"}]}

    def request(method, url, **kwargs):
        sent.append(kwargs["headers"])
        return Rejected()

    monkeypatch.setattr("requests.request", request)
    service = ingestion.build_service("asana", token)

    with pytest.raises(AsanaAuthError) as raised:
        service.request("GET", "/projects/1201234567890123")

    assert [headers["Authorization"] for headers in sent] == [f"Bearer {token}"]
    assert token not in repr(service)
    assert token not in str(raised.value) and token not in repr(raised.value)
    assert token not in raised.value.detail
    assert str(raised.value) == "Asana /projects/{gid}: HTTP 401", "no id, no body"


def test_onenote_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Linear pin: the literal is deliberate. One notebook is one scope, and
    # the Cloud control plane only tracks tables that start with the provider prefix.
    name = ingestion.resource_name("onenote", CREDENTIAL, "notebook-1")
    assert name == "onenote_2dcaae86aa376af1f107f951"
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("onenote").__module__ == (
        "cognee.tasks.ingestion.connectors.onenote"
    )
    with pytest.raises(KeyError, match="has no service to build"):
        ingestion.build_service("onenote", "a-token")


def test_unknown_provider_is_refused_by_every_provider_keyed_function():
    for call in (
        lambda: ingestion.resource_name("nope", CREDENTIAL),
        lambda: ingestion.source_factory("nope"),
        lambda: ingestion.build_service("nope", "token"),
    ):
        with pytest.raises(KeyError, match="Unsupported document provider"):
            call()


def test_google_module_delegates_to_the_neutral_one():
    for name in (
        "resource_name",
        "source_factory",
        "build_service",
        "empty_resource",
        "source_counts",
        "add_source_counts",
        "extraction_checkpoint",
        "retire_resources",
        "run_sync",
        "sync_is_running",
        "dataset_summary",
        "require_active_credential",
    ):
        assert getattr(google_ingestion, name) is getattr(ingestion, name)


def test_source_counts_keeps_only_non_negative_ints():
    source = SimpleNamespace(cognee_sync_stats={"scanned": 2, "bad": -1, "flag": True, 3: 1})
    assert ingestion.source_counts(source) == {"scanned": 2}


def test_importing_the_neutral_module_loads_no_provider_library():
    import subprocess
    import sys

    code = (
        "import sys, cognee.modules.integrations.ingestion as m;"
        "bad = [n for n in sys.modules if n.startswith(('googleapiclient', 'google.oauth2',"
        " 'cognee.modules.integrations.linear', 'cognee.tasks.ingestion.connectors'))];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dropbox_names_are_pinned_and_a_valid_resource_name():
    # Same rule as the Linear pin: the literal is deliberate. One selected folder is one scope,
    # named by its Dropbox folder id so a rename keeps the table.
    name = ingestion.resource_name("dropbox", CREDENTIAL, "id:folder-1")
    assert name == "dropbox_85beb5eb5e658f00e364d3ab"
    assert len(name) <= 63
    assert ingestion.source_factory("dropbox").__module__ == (
        "cognee.tasks.ingestion.connectors.dropbox"
    )
    # The pinned name is one dropbox_source accepts.
    ingestion.source_factory("dropbox")(
        folder_id="id:folder-1", account_id="dbid:A", resource_name=name, token="t"
    )


def test_dropbox_service_from_providers_sends_the_oauth_token_as_bearer():
    import httpx

    from cognee.tasks.ingestion.connectors.dropbox import DropboxAuthError, DropboxClient

    token = "sl.cloud-oauth-access-token"
    service = ingestion.build_service("dropbox", token)
    assert isinstance(service, DropboxClient) and token not in repr(service)
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(401, json={"error": {".tag": "expired_access_token"}})

    service._http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(DropboxAuthError) as error:
        service.rpc("users/get_current_account", None)
    assert seen == [f"Bearer {token}"]
    assert token not in f"{error.value} {error.value!r}"


def test_granola_names_are_pinned_and_a_dlt_table_name():
    # Same rule as the Linear pin: the literal is deliberate. One scope (``all``, a folder or a
    # space) is one source and one table, and the Cloud control plane only tracks tables that
    # start with the provider prefix.
    name = ingestion.resource_name("granola", CREDENTIAL, "fol_123456789ABCDE")
    assert name == "granola_1eea05a80c498c37d607bf3e"
    assert ingestion.resource_name("granola", CREDENTIAL, "all") == (
        "granola_911eeeb211c574888c636ffe"
    )
    assert name == name.lower() and set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert len(name) <= 63
    assert ingestion.source_factory("granola").__module__ == (
        "cognee.tasks.ingestion.connectors.granola"
    )


def test_granola_service_from_providers_sends_the_api_key_as_bearer_and_shows_it_nowhere(
    monkeypatch,
):
    # Granola has no OAuth for its REST API: the credential a host hands the builder is the
    # pasted API key (Cloud: its API-key connector), and the client sends it as
    # ``Authorization: Bearer`` on the request requests really sends.
    import requests

    from cognee.tasks.ingestion.connectors.granola import GranolaClient, GranolaError

    api_key = "grn_SECRET-granola-key-123"
    sent = []

    def send(self, request, **kwargs):
        sent.append(request.headers.get("Authorization"))
        response = requests.Response()
        response.status_code = 401
        response._content = f'{{"error": "bad key {api_key}"}}'.encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    service = ingestion.build_service("granola", api_key)
    assert isinstance(service, GranolaClient)

    with pytest.raises(GranolaError) as raised:
        service.get("/v1/notes", {"page_size": 1})

    assert sent == [f"Bearer {api_key}"]
    assert raised.value.status == 401
    assert api_key not in str(raised.value) and api_key not in repr(service)
    source = ingestion.source_factory("granola")(
        scope="fol_123456789ABCDE", resource_name="granola_x", service=service
    )
    assert source.name == "granola_x"
