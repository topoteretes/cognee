"""Which storage paths live on a remote object store, and which backend serves each.

The one place that knows the remote URL schemes: code that must treat a remote root
differently from a local one asks here instead of testing for ``"s3://"`` itself.
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .RemoteFileStorage import RemoteFileStorage

# URL scheme -> the STORAGE_BACKEND value of the object store that serves it.
REMOTE_STORAGE_BACKENDS = {"s3": "s3", "az": "azure"}


def get_remote_scheme(path: str) -> str | None:
    """The object-store scheme (``s3``, ``az``) of ``path``; None for a local path or file URI."""
    scheme, separator, _ = path.partition("://")
    return scheme if separator and scheme in REMOTE_STORAGE_BACKENDS else None


def is_remote_path(path: str) -> bool:
    return get_remote_scheme(path) is not None


def get_backend_scheme(storage_backend: str) -> str | None:
    """The URL scheme of a remote STORAGE_BACKEND value (``s3`` -> ``s3``, ``azure`` -> ``az``)."""
    for scheme, backend in REMOTE_STORAGE_BACKENDS.items():
        if backend == storage_backend.lower():
            return scheme
    return None


def is_configured_backend(scheme: str) -> bool:
    """Whether STORAGE_BACKEND names the object store that serves ``scheme``."""
    return os.getenv("STORAGE_BACKEND", "").lower() == REMOTE_STORAGE_BACKENDS[scheme]


def get_remote_storage_class(scheme: str) -> type["RemoteFileStorage"]:
    """The backend class for a remote scheme, imported only when used."""
    if scheme == "s3":
        from .S3FileStorage import S3FileStorage

        return S3FileStorage
    if scheme == "az":
        from .AzureBlobFileStorage import AzureBlobFileStorage

        return AzureBlobFileStorage
    raise ValueError(f"Not a remote storage scheme: {scheme!r}")


def get_remote_file_storage(path: str) -> "RemoteFileStorage":
    """The backend serving the remote ``path``, rooted at it."""
    scheme = get_remote_scheme(path)
    if scheme is None:
        raise ValueError(f"Not a remote storage path: {path!r}")
    return get_remote_storage_class(scheme)(path)
