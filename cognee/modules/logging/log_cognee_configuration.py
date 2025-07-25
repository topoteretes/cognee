def log_cognee_configuration(logger):
    """Log the current database configuration for all database types"""
    # NOTE: Has to be imporated at runtime to avoid circular import
    from cognee.infrastructure.databases.relational.config import get_relational_config
    from cognee.infrastructure.databases.vector.config import get_vectordb_config
    from cognee.infrastructure.databases.graph.config import get_graph_config

    try:
        # Log relational database configuration
        relational_config = get_relational_config()
        logger.info(f"Relational database: {relational_config.db_provider}")
        if relational_config.db_provider == "postgres":
            logger.info(f"Postgres host: {relational_config.db_host}:{relational_config.db_port}")
            logger.info(f"Postgres database: {relational_config.db_name}")
        elif relational_config.db_provider == "sqlite":
            logger.info(f"SQLite path: {relational_config.db_path}/{relational_config.db_name}")
            logger.info(f"SQLite database: {relational_config.db_name}")

        # Log vector database configuration
        vector_config = get_vectordb_config()
        logger.info(f"Vector database: {vector_config.vector_db_provider}")
        if vector_config.vector_db_provider == "lancedb":
            logger.info(f"Vector database path: {vector_config.vector_db_url}")
        else:
            logger.info(f"Vector database URL: {vector_config.vector_db_url}")

        # Log graph database configuration
        graph_config = get_graph_config()
        logger.info(f"Graph database: {graph_config.graph_database_provider}")
        if graph_config.graph_database_provider == "kuzu":
            logger.info(f"Graph database path: {graph_config.graph_file_path}")
        else:
            logger.info(f"Graph database URL: {graph_config.graph_database_url}")

    except Exception as e:
        logger.warning(f"Could not retrieve database configuration: {str(e)}")
