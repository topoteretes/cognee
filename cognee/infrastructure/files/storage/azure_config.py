from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.exceptions import CogneeConfigurationError

CREDENTIAL_SETTINGS = (
    "AZURE_STORAGE_CONNECTION_STRING, or AZURE_STORAGE_ACCOUNT_NAME with "
    "AZURE_STORAGE_ACCOUNT_KEY, AZURE_STORAGE_SAS_TOKEN or Azure's credential chain"
)


class AzureStorageConfigurationError(CogneeConfigurationError):
    """Azure Blob Storage cannot be reached with the configured settings."""

    def __init__(self, message: str, remediation: str = f"Set {CREDENTIAL_SETTINGS}."):
        super().__init__(message, "AzureStorageConfigurationError", remediation=remediation)


class AzureConfig(BaseSettings):
    """Azure Blob Storage credentials, under the names Azure's own tools read.

    Used in this order: a connection string; else the account name with an account
    key or a SAS token; else the account name with Azure's credential chain
    (managed identity, workload identity, service principal, Azure CLI).
    """

    azure_storage_connection_string: str | None = None
    azure_storage_account_name: str | None = None
    azure_storage_account_key: str | None = None
    azure_storage_sas_token: str | None = None
    model_config = SettingsConfigDict(extra="allow")

    def adlfs_kwargs(self) -> dict:
        """Arguments for ``adlfs.AzureBlobFileSystem``."""
        if self.azure_storage_connection_string:
            return {"connection_string": self.azure_storage_connection_string}
        kwargs = {"account_name": self._account_name()}
        if self.azure_storage_account_key:
            return {**kwargs, "account_key": self.azure_storage_account_key}
        if self.azure_storage_sas_token:
            return {**kwargs, "sas_token": self.azure_storage_sas_token}
        # anon=False: adlfs authenticates through azure-identity's DefaultAzureCredential.
        return {**kwargs, "anon": False}

    def lance_storage_options(self) -> dict[str, str]:
        """LanceDB storage options. LanceDB can't read a connection string, so it is split."""
        if self.azure_storage_connection_string:
            return _lance_options_from_connection_string(self.azure_storage_connection_string)
        options = {"azure_storage_account_name": self._account_name()}
        if self.azure_storage_account_key:
            options["azure_storage_account_key"] = self.azure_storage_account_key
        elif self.azure_storage_sas_token:
            options["azure_storage_sas_token"] = self.azure_storage_sas_token
        return options

    def _account_name(self) -> str:
        if not self.azure_storage_account_name:
            raise AzureStorageConfigurationError(
                "Azure Blob Storage has no account: neither AZURE_STORAGE_CONNECTION_STRING "
                "nor AZURE_STORAGE_ACCOUNT_NAME is set."
            )
        return self.azure_storage_account_name


def _lance_options_from_connection_string(connection_string: str) -> dict[str, str]:
    from azure.core.utils import parse_connection_string  # installed with the azure extra

    try:
        parts = parse_connection_string(connection_string)  # keys lower-cased
    except ValueError:
        raise AzureStorageConfigurationError(
            "AZURE_STORAGE_CONNECTION_STRING is malformed."
        ) from None
    if "accountname" not in parts:
        raise AzureStorageConfigurationError("AZURE_STORAGE_CONNECTION_STRING has no AccountName.")

    options = {"azure_storage_account_name": parts["accountname"]}
    if "accountkey" in parts:
        options["azure_storage_account_key"] = parts["accountkey"]
    if "sharedaccesssignature" in parts:
        options["azure_storage_sas_token"] = parts["sharedaccesssignature"]
    if "blobendpoint" in parts:
        options["azure_storage_endpoint"] = parts["blobendpoint"]
        if parts["blobendpoint"].startswith("http://"):
            options["allow_http"] = "true"
    return options


@lru_cache
def get_azure_config() -> AzureConfig:
    return AzureConfig()
