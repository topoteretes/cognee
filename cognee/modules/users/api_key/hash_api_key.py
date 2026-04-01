import hashlib
import os

HASH_API_KEY = os.getenv("HASH_API_KEY", "false").lower() == "true"


def hash_api_key(api_key: str) -> str:
    """Return SHA-256 hex digest of the key. Used for both storage and lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def prepare_api_key_for_storage(api_key: str) -> str:
    """Return the value that should be persisted to the database."""
    if HASH_API_KEY:
        return hash_api_key(api_key)
    return api_key


def prepare_api_key_for_lookup(api_key: str) -> str:
    """Return the value to use when querying the database."""
    if HASH_API_KEY:
        return hash_api_key(api_key)
    return api_key
