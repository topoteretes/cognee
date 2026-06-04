from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jAuraDevDatasetDatabaseHandler import (
    Neo4jAuraDevDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jDatasetDatabaseHandler import (
    Neo4jDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.vector.lancedb.LanceDBDatasetDatabaseHandler import (
    LanceDBDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.ladybug.LadybugDatasetDatabaseHandler import (
    LadybugDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.vector.pgvector.PGVectorDatasetDatabaseHandler import (
    PGVectorDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.postgres.PostgresGraphDatasetDatabaseHandler import (
    PostgresGraphDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.hybrid.helix.HelixGraphDatasetDatabaseHandler import (
    HelixGraphDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.hybrid.helix.HelixVectorDatasetDatabaseHandler import (
    HelixVectorDatasetDatabaseHandler,
)

supported_dataset_database_handlers = {
    "neo4j_aura_dev": {
        "handler_instance": Neo4jAuraDevDatasetDatabaseHandler,
        "handler_provider": "neo4j",
    },
    "neo4j": {
        "handler_instance": Neo4jDatasetDatabaseHandler,
        "handler_provider": "neo4j",
    },
    "lancedb": {"handler_instance": LanceDBDatasetDatabaseHandler, "handler_provider": "lancedb"},
    "pgvector": {
        "handler_instance": PGVectorDatasetDatabaseHandler,
        "handler_provider": "pgvector",
    },
    "postgres_graph": {
        "handler_instance": PostgresGraphDatasetDatabaseHandler,
        "handler_provider": "postgres",
    },
    "ladybug": {
        "handler_instance": LadybugDatasetDatabaseHandler,
        "handler_provider": "ladybug",
    },
    "kuzu": {"handler_instance": LadybugDatasetDatabaseHandler, "handler_provider": "kuzu"},
    "helix_graph": {
        "handler_instance": HelixGraphDatasetDatabaseHandler,
        "handler_provider": "helix",
    },
    "helix_vector": {
        "handler_instance": HelixVectorDatasetDatabaseHandler,
        "handler_provider": "helix",
    },
}
