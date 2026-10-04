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
