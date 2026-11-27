"""Azure settings become adlfs arguments and LanceDB storage options, in the documented
order: connection string, then account name with key, SAS token or credential chain."""

import pytest

from cognee.infrastructure.files.storage.azure_config import (
    AzureConfig,
    AzureStorageConfigurationError,
)

AZURITE = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=a2V5==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


def _config(monkeypatch, **values) -> AzureConfig:
    """A config built only from ``values``, whatever the test machine's environment holds."""
    for name in AzureConfig.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return AzureConfig(**values)


def test_connection_string_is_passed_whole_to_adlfs_and_split_for_lancedb(monkeypatch):
    config = _config(monkeypatch, azure_storage_connection_string=AZURITE)

    assert config.adlfs_kwargs() == {"connection_string": AZURITE}
    assert config.lance_storage_options() == {
        "azure_storage_account_name": "devstoreaccount1",
        "azure_storage_account_key": "a2V5==",
        "azure_storage_endpoint": "http://127.0.0.1:10000/devstoreaccount1",
        "allow_http": "true",
    }


@pytest.mark.parametrize(
    "credential, adlfs_extra, lance_extra",
    [
        (
            {"azure_storage_account_key": "k"},
            {"account_key": "k"},
            {"azure_storage_account_key": "k"},
        ),
        (
            {"azure_storage_sas_token": "sv=1"},
            {"sas_token": "sv=1"},
            {"azure_storage_sas_token": "sv=1"},
        ),
        ({}, {"anon": False}, {}),  # credential chain
    ],
)
def test_account_name_with_each_credential(monkeypatch, credential, adlfs_extra, lance_extra):
    config = _config(monkeypatch, azure_storage_account_name="acct", **credential)

    assert config.adlfs_kwargs() == {"account_name": "acct", **adlfs_extra}
    assert config.lance_storage_options() == {"azure_storage_account_name": "acct", **lance_extra}


def test_no_account_is_a_configuration_error(monkeypatch):
    config = _config(monkeypatch)

    for build in (config.adlfs_kwargs, config.lance_storage_options):
        with pytest.raises(AzureStorageConfigurationError, match="AZURE_STORAGE_ACCOUNT_NAME"):
            build()


@pytest.mark.parametrize("connection_string", ["garbage", "AccountKey=a2V5==;BlobEndpoint=x"])
def test_unusable_connection_string_is_a_configuration_error(monkeypatch, connection_string):
    config = _config(monkeypatch, azure_storage_connection_string=connection_string)

    with pytest.raises(AzureStorageConfigurationError, match="AZURE_STORAGE_CONNECTION_STRING"):
        config.lance_storage_options()
