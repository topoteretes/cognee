from sqlalchemy import URL

from .supported_databases import supported_databases
from .embeddings import get_embedding_engine
from cognee.infrastructure.databases.graph.config import get_graph_context_config

from functools import lru_cache


def create_vector_engine(
    vector_db_provider: str,
    vector_db_url: str,
    vector_db_name: str,
    vector_db_port: str = "",
    vector_db_key: str = "",
    vector_dataset_database_handler: str = "",
    vector_db_username: str = "",
    vector_db_password: str = "",
    vector_db_host: str = "",
):
    """
    Wrapper function to call create vector engine with caching.
    For a detailed description, see _create_vector_engine.
    """
    return _create_vector_engine(
        vector_db_provider,
        vector_db_url,
        vector_db_name,
        vector_db_port,
        vector_db_key,
        vector_dataset_database_handler,
        vector_db_username,
        vector_db_password,
        vector_db_host,
    )


@lru_cache
def _create_vector_engine(
    vector_db_provider: str,
    vector_db_url: str,
    vector_db_name: str,
    vector_db_port: str,
    vector_db_key: str,
    vector_dataset_database_handler: str,
    vector_db_username: str,
    vector_db_password: str,
    vector_db_host: str,
):
    """
    Create a vector database engine based on the specified provider.

    This function initializes and returns a database adapter for vector storage, depending
    on the provided vector database provider. The function checks for required credentials
    for each provider, raising an EnvironmentError if any are missing, or ImportError if the
    ChromaDB package is not installed.

    Supported providers include: pgvector, ChromaDB, and LanceDB.

    Parameters:
    -----------

        - vector_db_url (str): The URL for the vector database instance.
        - vector_db_port (str): The port for the vector database instance. Required for some
          providers.
        - vector_db_name (str): The name of the vector database instance.
        - vector_db_key (str): The API key or access token for the vector database instance.
        - vector_db_provider (str): The name of the vector database provider to use (e.g.,
          'pgvector').

    Returns:
    --------

        An instance of the corresponding database adapter class for the specified provider.
    """
    embedding_engine = get_embedding_engine()

    if vector_db_provider in supported_databases:
        adapter = supported_databases[vector_db_provider]

        return adapter(
            url=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
            database_name=vector_db_name,
        )

    if vector_db_provider.lower() == "pgvector":
        from cognee.context_global_variables import backend_access_control_enabled

        if backend_access_control_enabled():
            connection_string: str = (
                f"postgresql+asyncpg://{vector_db_username}:{vector_db_password}"
                f"@{vector_db_host}:{vector_db_port}/{vector_db_name}"
            )
        else:
            if (
                vector_db_port
                and vector_db_username
                and vector_db_password
                and vector_db_host
                and vector_db_name
            ):
                connection_string: str = (
                    f"postgresql+asyncpg://{vector_db_username}:{vector_db_password}"
                    f"@{vector_db_host}:{vector_db_port}/{vector_db_name}"
                )
            else:
                from cognee.infrastructure.databases.relational import get_relational_config

                # Get configuration for postgres database
                relational_config = get_relational_config()
                db_username = relational_config.db_username
                db_password = relational_config.db_password
                db_host = relational_config.db_host
                db_port = relational_config.db_port
                db_name = relational_config.db_name

                if not (db_host and db_port and db_name and db_username and db_password):
                    raise EnvironmentError("Missing required pgvector credentials!")

                connection_string: str = (
                    f"postgresql+asyncpg://{db_username}:{db_password}"
                    f"@{db_host}:{db_port}/{db_name}"
                )

        try:
            from .pgvector.PGVectorAdapter import PGVectorAdapter
        except ImportError:
            raise ImportError(
                "PostgreSQL dependencies are not installed. Please install with 'pip install cognee\"[postgres]\"' or 'pip install cognee\"[postgres-binary]\"' to use PGVector functionality."
            )

        return PGVectorAdapter(
            connection_string,
            vector_db_key,
            embedding_engine,
        )

    elif vector_db_provider.lower() == "chromadb":
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Please install it with 'pip install chromadb'"
            )

        from .chromadb.ChromaDBAdapter import ChromaDBAdapter

        return ChromaDBAdapter(
            url=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
        )

    elif vector_db_provider.lower() == "neptune_analytics":
        try:
            from langchain_aws import NeptuneAnalyticsGraph
        except ImportError:
            raise ImportError(
                "langchain_aws is not installed. Please install it with 'pip install langchain_aws'"
            )

        if not vector_db_url:
            raise EnvironmentError("Missing Neptune endpoint.")

        from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
            NeptuneAnalyticsAdapter,
            NEPTUNE_ANALYTICS_ENDPOINT_URL,
        )

        if not vector_db_url.startswith(NEPTUNE_ANALYTICS_ENDPOINT_URL):
            raise ValueError(
                f"Neptune endpoint must have the format '{NEPTUNE_ANALYTICS_ENDPOINT_URL}<GRAPH_ID>'"
            )

        graph_identifier = vector_db_url.replace(NEPTUNE_ANALYTICS_ENDPOINT_URL, "")

        return NeptuneAnalyticsAdapter(
            graph_id=graph_identifier,
            embedding_engine=embedding_engine,
        )

    elif vector_db_provider.lower() == "lancedb":
        from .lancedb.LanceDBAdapter import LanceDBAdapter

        return LanceDBAdapter(
            url=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
        )

    else:
        raise EnvironmentError(
            f"Unsupported vector database provider: {vector_db_provider}. "
            f"Supported providers are: {', '.join(list(supported_databases.keys()) + ['LanceDB', 'PGVector', 'neptune_analytics', 'ChromaDB'])}"
        )
