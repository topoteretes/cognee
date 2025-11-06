"""Shared constants for multi-user database configuration."""

DEFAULT_VECTOR_DB_NAME = "lance.db"
DEFAULT_VECTOR_DB_PROVIDER = "lancedb"
DEFAULT_GRAPH_DB_PROVIDER = "kuzu"
DEFAULT_VECTOR_DB_URL = None
DEFAULT_GRAPH_DB_URL = None
DEFAULT_VECTOR_DB_KEY = None
DEFAULT_GRAPH_DB_KEY = None

VECTOR_DBS_WITH_MULTI_USER_SUPPORT = {"lancedb", "falkor"}
GRAPH_DBS_WITH_MULTI_USER_SUPPORT = {"kuzu", "falkor", "ladybug"}

GRAPH_DB_EXTENSIONS = {
    "kuzu": ".pkl",
    "ladybug": ".db",
}

