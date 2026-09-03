from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jAuraDevDatasetDatabaseHandler import (
    Neo4jAuraDevDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jDatasetDatabaseHandler import (
    Neo4jDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.neo4j_driver.Neo4jCommunityDatasetDatabaseHandler import (
    Neo4jCommunityDatasetDatabaseHandler,
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
from cognee.infrastructure.databases.vector.pgvector.PGVectorSharedDatasetDatabaseHandler import (
    PGVectorSharedDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.vector.turso.TursoVectorDatasetDatabaseHandler import (
    TursoVectorDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.postgres_demo.PostgresGraphDatasetDatabaseHandler import (
    PostgresGraphDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.postgres_demo.PostgresGraphSharedDatasetDatabaseHandler import (
    PostgresGraphSharedDatasetDatabaseHandler,
)
from cognee.infrastructure.databases.graph.turso.TursoGraphDatasetDatabaseHandler import (
    TursoGraphDatasetDatabaseHandler,
)

# handler_provider is the database provider a handler works with: a plain string,
# or a tuple only when the handler is compatible with multiple providers.
supported_dataset_database_handlers = {
    "neo4j_aura_dev": {
        "handler_instance": Neo4jAuraDevDatasetDatabaseHandler,
        "handler_provider": "neo4j",
    },
    "neo4j": {
        "handler_instance": Neo4jDatasetDatabaseHandler,
        "handler_provider": "neo4j",
    },
    "neo4j_community": {
        "handler_instance": Neo4jCommunityDatasetDatabaseHandler,
        "handler_provider": "neo4j",
    },
    "lancedb": {"handler_instance": LanceDBDatasetDatabaseHandler, "handler_provider": "lancedb"},
    "pgvector": {
        "handler_instance": PGVectorDatasetDatabaseHandler,
        "handler_provider": "pgvector",
    },
    "pgvector_shared": {
        "handler_instance": PGVectorSharedDatasetDatabaseHandler,
        "handler_provider": "pgvector",
    },
    "turso": {
        "handler_instance": TursoVectorDatasetDatabaseHandler,
        "handler_provider": "turso",
    },
    # postgres_demo is the canonical provider name; postgres remains accepted.
    "postgres_graph": {
        "handler_instance": PostgresGraphDatasetDatabaseHandler,
        "handler_provider": ("postgres", "postgres_demo"),
    },
    "postgres_graph_shared": {
        "handler_instance": PostgresGraphSharedDatasetDatabaseHandler,
        "handler_provider": ("postgres", "postgres_demo"),
    },
    # Ladybug is the renamed Kuzu engine — either provider name works with either handler.
    "ladybug": {
        "handler_instance": LadybugDatasetDatabaseHandler,
        "handler_provider": ("ladybug", "kuzu"),
    },
    "kuzu": {
        "handler_instance": LadybugDatasetDatabaseHandler,
        "handler_provider": ("ladybug", "kuzu"),
    },
    "turso_graph": {
        "handler_instance": TursoGraphDatasetDatabaseHandler,
        "handler_provider": "turso",
    },
}
