from typing import Protocol
from .storage import Storage


class CloudStorageInterface(Storage, Protocol):
    """
    Abstract interface for cloud storage operations.
    """

    pass
