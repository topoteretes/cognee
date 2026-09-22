"""The one ``.env`` cognee loads: found from the project, once, winning for every reader.

A bare ``load_dotenv()`` searched upward from the installed package, so a project
``.env`` was found only when the environment happened to live inside the project,
and a stranger's ``.env`` above the package could be loaded instead. Meanwhile the
settings classes resolved their own ``.env`` from the working directory, so the
same file was half-honoured: provider settings worked, feature flags read with
``os.getenv`` did not (SDK-781).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.shared import env_file

KEY = "COGNEE_ENVFILE_TEST_KEY"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Fresh resolution per test, and the real environment put back afterwards."""
    if env_file._find_upward(tmp_path) is not None:  # pragma: no cover - host has a stray .env
        pytest.skip("a .env above the pytest tmp dir would confound these tests")
    snapshot = dict(os.environ)
    monkeypatch.setattr(env_file, "_loaded", False)
    monkeypatch.setattr(env_file, "_resolved", None)
    monkeypatch.setattr(env_file, "_package_directory", lambda: tmp_path / "no-package-env")
    (tmp_path / "no-package-env").mkdir()
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _write_env(directory: Path, value: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".env"
    path.write_text(f"{KEY}={value}\n")
    return path


def test_the_working_directory_file_is_found_first(monkeypatch, tmp_path):
    project = _write_env(tmp_path / "project", "from-project")
    _write_env(tmp_path / "no-package-env", "from-package-side")
    monkeypatch.chdir(tmp_path / "project")

    assert env_file.load_env_file() == str(project)
    assert os.environ[KEY] == "from-project"


def test_a_parent_of_the_working_directory_counts_as_the_project(monkeypatch, tmp_path):
    root = _write_env(tmp_path / "repo", "from-repo-root")
    deep = tmp_path / "repo" / "services" / "api"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert env_file.resolve_env_file() == str(root)


def test_the_package_side_file_is_the_fallback(monkeypatch, tmp_path):
    """The old behaviour survives for environments that live inside the project."""
    beside_package = _write_env(tmp_path / "no-package-env", "from-package-side")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert env_file.load_env_file() == str(beside_package)
    assert os.environ[KEY] == "from-package-side"


def test_nothing_found_means_the_process_environment_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(KEY, "preset")

    assert env_file.load_env_file() is None
    assert os.environ[KEY] == "preset"
    assert env_file.describe_resolution(None).startswith("No .env file found")


def test_the_file_wins_over_a_preset_variable(monkeypatch, tmp_path):
    _write_env(tmp_path / "project", "from-file")
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv(KEY, "preset-in-shell")

    env_file.load_env_file()

    assert os.environ[KEY] == "from-file"


def test_settings_classes_read_the_same_file_with_the_same_precedence(monkeypatch, tmp_path):
    """Every reader agrees: a BaseSettings field sees the file's value, not the preset."""
    _write_env(tmp_path / "project", "from-file")
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv(KEY, "preset-in-shell")

    class Probe(BaseSettings):
        cognee_envfile_test_key: str = "default"
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env_file.load_env_file()

    assert Probe().cognee_envfile_test_key == "from-file"
    assert os.getenv(KEY) == "from-file"


def test_the_load_happens_once_per_process(monkeypatch, tmp_path):
    path = _write_env(tmp_path / "project", "from-file")
    monkeypatch.chdir(tmp_path / "project")

    with patch.object(env_file.dotenv, "load_dotenv", wraps=env_file.dotenv.load_dotenv) as load:
        first = env_file.load_env_file()
        monkeypatch.chdir(tmp_path)  # a later chdir must not change the answer
        second = env_file.load_env_file()

    assert first == second == str(path)
    load.assert_called_once_with(str(path), override=True)


def test_the_log_line_names_the_file_and_the_precedence(tmp_path):
    message = env_file.describe_resolution(str(tmp_path / ".env"))

    assert str(tmp_path / ".env") in message
    assert "take precedence over preset environment variables" in message
    assert "working directory" in message and "package directory" in message


def test_a_deleted_working_directory_does_not_break_import(monkeypatch, tmp_path):
    monkeypatch.setattr(env_file, "_working_directory", lambda: None)
    beside_package = _write_env(tmp_path / "no-package-env", "from-package-side")

    assert env_file.resolve_env_file() == str(beside_package)
