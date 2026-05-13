"""Legacy import path for Ladybug graph database migration helpers."""

from cognee.infrastructure.databases.graph.ladybug.ladybug_migrate import (
    ladybug_migration,
    read_ladybug_storage_version,
)


kuzu_migration = ladybug_migration
read_kuzu_storage_version = read_ladybug_storage_version
