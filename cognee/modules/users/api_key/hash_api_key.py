import hashlib
import os

HASH_API_KEY = os.getenv("HASH_API_KEY", "false").lower() == "true"


def hash_api_key(api_key: str) -> str:
    """Return SHA-256 hex digest of the key. Used for both storage and lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def prepare_api_key(api_key: str) -> str:
    """Return the value to use when storing or querying the database."""
    if HASH_API_KEY:
        return hash_api_key(api_key)
    return api_key
