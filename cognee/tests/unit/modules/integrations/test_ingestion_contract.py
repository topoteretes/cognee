"""The provider-neutral ingestion contract and the persisted Google resource names."""

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
