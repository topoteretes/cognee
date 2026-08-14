"""Per-URL credential profiles: local and cloud connections must coexist.

The store previously held a single flat record, so connecting to a local
instance overwrote the cloud profile — including the Auth0 refresh token,
forcing a browser re-auth on the next cloud connect. Profiles are keyed
by service URL; legacy single-record files load transparently.
"""

import json

import pytest

from cognee.api.v1.serve import credentials
from cognee.api.v1.serve.credentials import (
    CloudCredentials,
    clear_credentials,
    load_credentials,
    save_credentials,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    path = tmp_path / "cloud_credentials.json"
    monkeypatch.setattr(credentials, "_CREDENTIALS_FILE", path)
    return path


CLOUD = CloudCredentials(
    access_token="auth0-token",
    refresh_token="auth0-refresh",
    expires_at=9e9,
    service_url="https://tenant-abc.cloud.cognee.ai",
    api_key="ck_cloud",
    management_url="https://api.cloud.example",
    email="owner@corp.dev",
)

LOCAL = CloudCredentials(
    access_token="",
    service_url="http://localhost:8011",
    api_key="ck_local",
    email="local",
)


def test_local_save_does_not_clobber_cloud_profile():
    save_credentials(CLOUD)
    save_credentials(LOCAL)

    cloud = load_credentials("https://tenant-abc.cloud.cognee.ai")
    assert cloud is not None
    assert cloud.api_key == "ck_cloud"
    # The Auth0 session survives the local connect — the original bug.
    assert cloud.refresh_token == "auth0-refresh"

    local = load_credentials("http://localhost:8011")
    assert local is not None
    assert local.api_key == "ck_local"


def test_no_arg_load_returns_most_recently_used_profile():
    save_credentials(CLOUD)
    save_credentials(LOCAL)
    assert load_credentials().service_url == "http://localhost:8011"

    save_credentials(CLOUD)
    assert load_credentials().service_url == "https://tenant-abc.cloud.cognee.ai"


def test_cloud_filter_skips_direct_profiles():
    save_credentials(CLOUD)
    save_credentials(LOCAL)  # most recent, but not a cloud profile
    creds = load_credentials(cloud=True)
    assert creds is not None
    assert creds.service_url == "https://tenant-abc.cloud.cognee.ai"


def test_cloud_filter_returns_none_when_only_local_profiles_exist():
    save_credentials(LOCAL)
    assert load_credentials(cloud=True) is None


def test_legacy_flat_file_loads_and_upgrades(isolated_store):
    # A file written by the pre-profile schema: one flat record.
    isolated_store.write_text(
        json.dumps(
            {
                "access_token": "auth0-token",
                "refresh_token": "auth0-refresh",
                "expires_at": 9e9,
                "service_url": "https://tenant-abc.cloud.cognee.ai",
                "api_key": "ck_cloud",
                "email": "owner@corp.dev",
            }
        )
    )

    creds = load_credentials("https://tenant-abc.cloud.cognee.ai")
    assert creds is not None
    assert creds.api_key == "ck_cloud"

    # The next save upgrades the file to the profile schema without
    # losing the legacy record.
    save_credentials(LOCAL)
    upgraded = json.loads(isolated_store.read_text())
    assert set(upgraded["profiles"]) == {
        "https://tenant-abc.cloud.cognee.ai",
        "http://localhost:8011",
    }


def test_clear_single_profile_keeps_others():
    save_credentials(CLOUD)
    save_credentials(LOCAL)
    clear_credentials("http://localhost:8011")
    assert load_credentials("http://localhost:8011") is None
    assert load_credentials("https://tenant-abc.cloud.cognee.ai") is not None
    # last_used pointed at the removed profile; a no-arg load must still work.
    assert load_credentials() is not None


def test_clear_all_removes_the_store(isolated_store):
    save_credentials(CLOUD)
    clear_credentials()
    assert not isolated_store.exists()
    assert load_credentials() is None


def test_urls_are_normalized_on_save_and_load():
    save_credentials(
        CloudCredentials(access_token="", service_url="http://localhost:8011/", api_key="ck_x")
    )
    assert load_credentials("http://localhost:8011").api_key == "ck_x"
    assert load_credentials("http://localhost:8011/").api_key == "ck_x"
