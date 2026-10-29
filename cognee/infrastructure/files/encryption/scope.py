"""Which storage roots the encryption layer applies to, and how it is attached.

Only objects under ``DATA_ROOT_DIRECTORY`` are sealed: the ingested files and
the text derived from them. ``SYSTEM_ROOT_DIRECTORY`` holds embedded database
files that their engines open directly by path, and the cache and log roots
hold nothing a storage call reads back through this layer — wrapping those
would corrupt the databases, not protect them.
"""

import os

from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage.get_storage_config import get_storage_config
from cognee.infrastructure.files.storage.LocalFileStorage import get_parsed_path
from cognee.infrastructure.files.storage.storage import Storage

from .config import get_file_encryption_config
from .envelope import MAGIC, is_envelope


def _same_or_within(path: str, root: str) -> bool:
    if path.startswith("s3://") or root.startswith("s3://"):
        if not (path.startswith("s3://") and root.startswith("s3://")):
            return False
        path_key = path[len("s3://") :].rstrip("/")
        root_key = root[len("s3://") :].rstrip("/")
        return path_key == root_key or path_key.startswith(root_key + "/")

    resolved_path = os.path.realpath(os.path.expanduser(get_parsed_path(path)))
    resolved_root = os.path.realpath(os.path.expanduser(get_parsed_path(root)))
    return resolved_path == resolved_root or resolved_path.startswith(
        resolved_root.rstrip(os.sep) + os.sep
    )


def is_under_data_root(storage_path: str) -> bool:
    """Whether ``storage_path`` is the data root or lies beneath it.

    Both the process-wide root and the dataset-scoped one a pipeline context
    binds count: a per-dataset data directory is still cognee's data.
    """
    roots = {get_base_config().data_root_directory}
    context_root = get_storage_config().get("data_root_directory")
    if context_root:
        roots.add(context_root)
    return any(_same_or_within(storage_path, root) for root in roots if root)


def wrap_storage_if_encrypted(storage: Storage) -> Storage:
    """``storage`` behind the encryption layer when it is on and this root is in scope."""
    config = get_file_encryption_config()
    if not config.enabled:
        return storage
    if not is_under_data_root(storage.storage_path):
        return storage

    from .EncryptedFileStorage import EncryptedFileStorage
    from .key_providers import get_key_provider

    return EncryptedFileStorage(storage, get_key_provider(config), config.chunk_size)


def is_encrypted_local_file(path: str) -> bool:
    """Whether the local file at ``path`` starts with the envelope magic.

    For callers that must hand a *path* to code that opens files itself
    (loaders, ``FileResponse``): a sealed file has to be unsealed to a
    temporary copy first. Cheap — one small read — and false for a missing
    file, so it can gate the fallback without a separate existence check.
    """
    try:
        with open(path, "rb") as file:
            return is_envelope(file.read(len(MAGIC)))
    except OSError:
        return False
