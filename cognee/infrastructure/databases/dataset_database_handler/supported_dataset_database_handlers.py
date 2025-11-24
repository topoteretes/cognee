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
    "neo4j_aura": Neo4jAuraDatasetDatabaseHandler,
    "lancedb": LanceDBDatasetDatabaseHandler,
    "kuzu": KuzuDatasetDatabaseHandler,
}
