"""Legacy import path for the Ladybug dataset database handler."""

from cognee.infrastructure.databases.graph.ladybug.LadybugDatasetDatabaseHandler import (
    LadybugDatasetDatabaseHandler,
)


KuzuDatasetDatabaseHandler = LadybugDatasetDatabaseHandler
