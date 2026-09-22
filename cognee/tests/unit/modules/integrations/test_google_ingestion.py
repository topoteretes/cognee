"""Google integration factories use SDK sources without community packages."""

import builtins

import pytest

from cognee.modules.integrations.google.ingestion import source_factory


@pytest.mark.parametrize("provider", ["gmail", "google_drive"])
def test_source_factory_uses_bundled_connector(provider, monkeypatch):
    original_import = builtins.__import__

    def sdk_only_import(name, *args, **kwargs):
        if name.startswith("cognee_community_connector_"):
            raise AssertionError("Google ingestion must not require a community package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", sdk_only_import)
    factory = source_factory(provider)
    assert factory.__module__ == f"cognee.tasks.ingestion.connectors.{provider}"
    assert factory.__name__ == f"{provider}_source"


@pytest.mark.parametrize("provider,extra", [("gmail", "gmail"), ("google_drive", "google-drive")])
def test_missing_dlt_points_to_sdk_extra(provider, extra, monkeypatch):
    factory = source_factory(provider)
    original_import = builtins.__import__

    def no_dlt(name, *args, **kwargs):
        if name == "dlt":
            raise ImportError("DLT is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_dlt)
    with pytest.raises(ImportError) as error:
        factory(service=object())
    assert f"cognee[{extra}]" in str(error.value)


def test_public_sources_are_the_integration_sources():
    from cognee.tasks.ingestion.connectors import gmail_source, google_drive_source

    assert source_factory("gmail") is gmail_source
    assert source_factory("google_drive") is google_drive_source
