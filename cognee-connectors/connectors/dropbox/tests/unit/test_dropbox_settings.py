"""``DROPBOX_*`` settings: the names cognee reads, and the file-size cap's default and refusals.

Copied from the settings tests of cognee.tests.unit.modules.integrations.test_dropbox_integration
(cognee dev, 2026-11-12).
"""

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from cognee_connector_dropbox.settings import DropboxSettings, require


def test_the_oauth_settings_come_from_the_dropbox_variables(monkeypatch):
    for name, value in {
        "DROPBOX_CLIENT_ID": "key",
        "DROPBOX_CLIENT_SECRET": "secret",
        "DROPBOX_REDIRECT_URI": "https://cognee.test/api/v1/integrations/dropbox/callback",
        "DROPBOX_FRONTEND_BASE_URL": "https://cognee.test",
        "DROPBOX_STATE_SECRET": "state",
    }.items():
        monkeypatch.setenv(name, value)

    settings = DropboxSettings()

    assert (settings.client_id, settings.client_secret, settings.state_secret) == (
        "key",
        "secret",
        "state",
    )
    assert settings.redirect_uri.endswith("/dropbox/callback")
    assert settings.frontend_base_url == "https://cognee.test"


def test_an_unset_setting_is_refused_by_its_variable_name(monkeypatch):
    from cognee_connector_dropbox import settings as settings_module

    monkeypatch.setattr(settings_module.dropbox_settings, "state_secret", "")
    with pytest.raises(RuntimeError, match="DROPBOX_STATE_SECRET is not configured"):
        require("state_secret")


def test_the_file_size_setting_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", "2.5")
    assert DropboxSettings().max_file_size_mb == 2.5
    monkeypatch.delenv("DROPBOX_MAX_FILE_SIZE_MB")
    assert DropboxSettings().max_file_size_mb == 50, "the integration's default cap"


def test_the_integration_caps_file_size_at_50_mb_by_default(monkeypatch):
    monkeypatch.delenv("DROPBOX_MAX_FILE_SIZE_MB", raising=False)
    assert DropboxSettings().max_file_size_mb == 50
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", "")
    assert DropboxSettings().max_file_size_mb == 50, "blank means the default, not no limit"
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", "200")
    assert DropboxSettings().max_file_size_mb == 200


@pytest.mark.parametrize("value", ["", " "])
def test_an_empty_file_size_setting_means_the_default(monkeypatch, value):
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", value)
    assert DropboxSettings().max_file_size_mb == 50


@pytest.mark.parametrize("value", ["inf", "nan", "0", "-1", "lots"])
def test_a_file_size_setting_that_would_skip_everything_or_nothing_is_refused(monkeypatch, value):
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", value)
    with pytest.raises(ValidationError):
        DropboxSettings()


def test_the_package_imports_with_an_empty_file_size_setting():
    env = {**os.environ, "DROPBOX_MAX_FILE_SIZE_MB": ""}
    subprocess.run([sys.executable, "-c", "import cognee_connector_dropbox"], env=env, check=True)
