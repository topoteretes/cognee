import hashlib
import os

HASH_API_KEY = os.getenv("HASH_API_KEY", "false").lower() == "true"
API_KEY_HASH_ITERATIONS = int(os.getenv("API_KEY_HASH_ITERATIONS", "600000"))
API_KEY_HASH_SALT = os.getenv("API_KEY_HASH_SALT", "cognee-api-key-v1").encode("utf-8")


def hash_api_key(api_key: str) -> str:
    """Return a deterministic slow hash of the key for storage and lookup."""
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        api_key.encode("utf-8"),
        API_KEY_HASH_SALT,
        API_KEY_HASH_ITERATIONS,
    )
    return f"pbkdf2_sha256${API_KEY_HASH_ITERATIONS}${digest.hex()}"


def prepare_api_key(api_key: str) -> str:
    """Return the value to use when storing or querying the database."""
    if HASH_API_KEY:
        return hash_api_key(api_key)
    return api_key
