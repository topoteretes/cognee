from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jAuraDatasetDatabaseHandler import (
    Neo4jAuraDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.vector.lancedb.LanceDBDatasetDatabaseHandler import (
    LanceDBDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.kuzu.KuzuDatasetDatabaseHandler import (
    KuzuDatasetDatabaseHandler,
)

supported_dataset_database_handlers = {
    "neo4j_aura": {
        "handler_instance": Neo4jAuraDatasetDatabaseHandler,
        "handler_provider": "neo4j",
    },
    "lancedb": {"handler_instance": LanceDBDatasetDatabaseHandler, "handler_provider": "lancedb"},
    "kuzu": {"handler_instance": KuzuDatasetDatabaseHandler, "handler_provider": "kuzu"},
}
