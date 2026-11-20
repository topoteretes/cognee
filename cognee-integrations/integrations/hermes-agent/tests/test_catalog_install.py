"""Catalog copies retain their reviewed code even alongside a newer pip package."""

import json
from types import SimpleNamespace

import pytest
from cognee_integration_hermes import catalog, cli, installer


@pytest.fixture
def installed_root(tmp_path, monkeypatch):
    root = tmp_path / "plugins" / "cognee"
    package = root / "cognee_integration_hermes"
    package.mkdir(parents=True)
    (root / "plugin.yaml").write_text("name: cognee\nversion: 1.2.2\n")
    monkeypatch.setattr(catalog, "__file__", str(package / "catalog.py"))
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    monkeypatch.setattr(cli, "_pip_package_version", lambda: "9.0.0")
    monkeypatch.setattr(cli, "load_config", lambda: {})
    monkeypatch.setattr(cli, "config_path", lambda: None)
    monkeypatch.setattr(cli, "_provider_active", lambda: True)
    monkeypatch.setattr(cli.runtime_env, "has_cognee", lambda: False)
    return root


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("command", ["status", "version", "install"])
@pytest.mark.parametrize(
    "marker",
    [
        '{"catalog_name": "cognee"}',
        '{"catalog_name": "cognee-memory"}',
        "broken json",
        "[]",
        '{"catalog_name": "bad;command"}',
    ],
)
def test_catalog_cli_never_checks_pypi_or_suggests_overwrite(
    installed_root,
    monkeypatch,
    capsys,
    force,
    command,
    marker,
):
    (installed_root / ".hermes-catalog.json").write_text(marker)

    def unexpected_pypi(**kwargs):
        pytest.fail("catalog installs must not query PyPI")

    monkeypatch.setattr(cli.update_check, "latest_published_version", unexpected_pypi)
    cli.cognee_command(SimpleNamespace(cognee_command=command, check_updates=force))
    output = capsys.readouterr().out
    name = "cognee-memory" if "cognee-memory" in marker else "cognee"
    assert f"hermes plugins update {name}" in output
    assert "cognee-hermes-install" not in output
    assert "pip install" not in output
    assert "9.0.0" not in output
    if command != "install":
        assert "1.2.2" in output


def test_pip_status_still_reports_copy_drift(installed_root, capsys):
    cli.cognee_command(SimpleNamespace(cognee_command="status", check_updates=False))
    output = capsys.readouterr().out
    assert "9.0.0" in output
    assert "cognee-hermes-install" in output


def test_pip_update_check_still_works(installed_root, monkeypatch):
    calls = []

    def latest(**kwargs):
        calls.append(kwargs)
        return "10.0.0"

    monkeypatch.setattr(cli.update_check, "latest_published_version", latest)
    assert "pip install -U" in cli._update_hint({}, force=True)
    assert calls == [{"interval": 3600.0, "force": True}]
    assert cli._update_hint({"update_check": False}, force=True) == ""
    assert len(calls) == 1


@pytest.mark.parametrize("marker", ['{"catalog_name": "cognee"}', "{broken", "null"])
def test_installer_preserves_all_catalog_files(installed_root, marker):
    (installed_root / ".hermes-catalog.json").write_text(marker)
    (installed_root / "cognee_integration_hermes" / "sentinel.py").write_text("reviewed code")
    before = {
        p.relative_to(installed_root): p.read_bytes()
        for p in installed_root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(RuntimeError, match="hermes plugins update cognee"):
        installer.install(installed_root.parents[1])
    after = {
        p.relative_to(installed_root): p.read_bytes()
        for p in installed_root.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_installer_cli_reports_catalog_refusal(installed_root, capsys):
    (installed_root / ".hermes-catalog.json").write_text(json.dumps({"catalog_name": "cognee"}))
    with pytest.raises(SystemExit) as exc:
        installer.main(["--home", str(installed_root.parents[1])])
    assert exc.value.code == 1
    assert "hermes plugins update cognee" in capsys.readouterr().err
