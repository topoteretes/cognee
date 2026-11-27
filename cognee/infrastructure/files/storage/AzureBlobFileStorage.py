from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from .azure_config import AzureStorageConfigurationError, get_azure_config
from .RemoteFileStorage import RemoteFileStorage

T = TypeVar("T")

# (account, container) pairs this process has seen exist and accept its credentials.
_CHECKED_CONTAINERS: set[tuple[str | None, str]] = set()

if TYPE_CHECKING:
    import adlfs
    from azure.core.exceptions import HttpResponseError

# Azure Storage error codes for credentials the account refuses: a wrong key or SAS
# token, or an identity without a data role on the container.
CREDENTIAL_ERROR_CODES = {
    "AuthenticationFailed",
    "AuthorizationFailure",
    "AuthorizationPermissionMismatch",
}


class AzureBlobFileStorage(RemoteFileStorage):
    """File storage on Azure Blob Storage, through adlfs. A root is ``az://<container>/<prefix>``."""

    scheme = "az"
    fs: "adlfs.AzureBlobFileSystem"

    def _create_filesystem(self) -> "adlfs.AzureBlobFileSystem":
        try:
            import adlfs
        except ImportError:
            raise ImportError(
                "adlfs is required for AzureBlobFileStorage. "
                'Install it with: pip install "cognee[azure]"'
            )
        return adlfs.AzureBlobFileSystem(**get_azure_config().adlfs_kwargs())

    async def _call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        await self._check_container()
        return await super()._call(func, *args, **kwargs)

    async def _check_container(self) -> None:
        """Once per process and container: fail fast on a missing container or refused access.

        adlfs reads a missing container as empty, so without this check it would surface
        only on the first write, possibly after a whole pipeline run.
        """
        container = (self.fs.service_client.account_name, self._container())
        if container in _CHECKED_CONTAINERS:
            return
        if not await super()._call(self.fs.exists, self._container()):
            raise self._missing_container_error()
        _CHECKED_CONTAINERS.add(container)

    def _translate_error(self, error: Exception) -> Exception:
        """A missing container or refused credentials become a configuration error."""
        azure_error = _azure_error_in(error)
        if azure_error is None:
            return error
        if getattr(azure_error, "error_code", None) == "ContainerNotFound":
            return self._missing_container_error()
        if _is_credential_error(azure_error):
            return AzureStorageConfigurationError(
                f"Azure Blob Storage refused access to {self._where()}."
            )
        return error

    def _missing_container_error(self) -> AzureStorageConfigurationError:
        return AzureStorageConfigurationError(
            f"Azure Blob Storage has no {self._where()}.",
            remediation="Create the container, or point the storage roots at an existing one.",
        )

    def _where(self) -> str:
        account = self.fs.service_client.account_name
        return f"container {self._container()!r} of account {account!r}"

    def _container(self) -> str:
        return self._root().split("/", 1)[0]


def _azure_error_in(error: Exception) -> "HttpResponseError | None":
    """The Azure service error behind ``error``: itself, or the cause adlfs wraps on writes."""
    from azure.core.exceptions import HttpResponseError

    for candidate in (error, error.__cause__):
        if isinstance(candidate, HttpResponseError):
            return candidate
    return None


def _is_credential_error(error: "HttpResponseError") -> bool:
    from azure.core.exceptions import ClientAuthenticationError

    # error_code is not an HttpResponseError attribute: azure-storage-blob adds it only
    # to the errors it processes, so other Azure errors lack it.
    return (
        isinstance(error, ClientAuthenticationError)
        or getattr(error, "error_code", None) in CREDENTIAL_ERROR_CODES
    )
