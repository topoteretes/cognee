from .supported_databases import supported_databases
from .embeddings import get_embedding_engine

from functools import lru_cache
import base64
import json

@lru_cache
def create_vector_engine(
    vector_db_url: str,
    vector_db_port: str,
    vector_db_key: str,
    vector_db_provider: str,
):
    """
    Create a vector database engine based on the specified provider.

    This function initializes and returns a database adapter for vector storage, depending
    on the provided vector database provider. The function checks for required credentials
    for each provider, raising an EnvironmentError if any are missing, or ImportError if the
    ChromaDB package is not installed.

    Supported providers include: Weaviate, Qdrant, Milvus, pgvector, FalkorDB, ChromaDB, and
    LanceDB.

    Parameters:
    -----------

        - vector_db_url (str): The URL for the vector database instance.
        - vector_db_port (str): The port for the vector database instance. Required for some
          providers.
        - vector_db_key (str): The API key or access token for the vector database instance.
        - vector_db_provider (str): The name of the vector database provider to use (e.g.,
          'weaviate', 'qdrant').

    Returns:
    --------

        An instance of the corresponding database adapter class for the specified provider.
    """
    embedding_engine = get_embedding_engine()

    if vector_db_provider in supported_databases:
        adapter = supported_databases[vector_db_provider]

        return adapter(
            utl=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
        )

    if vector_db_provider == "weaviate":
        from .weaviate_db import WeaviateAdapter

        if not (vector_db_url and vector_db_key):
            raise EnvironmentError("Missing requred Weaviate credentials!")

        return WeaviateAdapter(vector_db_url, vector_db_key, embedding_engine=embedding_engine)

    elif vector_db_provider == "qdrant":
        if not (vector_db_url and vector_db_key):
            raise EnvironmentError("Missing requred Qdrant credentials!")

        from .qdrant.QDrantAdapter import QDrantAdapter

        return QDrantAdapter(
            url=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
        )

    elif vector_db_provider == "milvus":
        from .milvus.MilvusAdapter import MilvusAdapter

        if not vector_db_url:
            raise EnvironmentError("Missing required Milvus credentials!")

        return MilvusAdapter(
            url=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
        )

    elif vector_db_provider == "pgvector":
        from cognee.infrastructure.databases.relational import get_relational_config

        # Get configuration for postgres database
        relational_config = get_relational_config()
        db_username = relational_config.db_username
        db_password = relational_config.db_password
        db_host = relational_config.db_host
        db_port = relational_config.db_port
        db_name = relational_config.db_name

        if not (db_host and db_port and db_name and db_username and db_password):
            raise EnvironmentError("Missing requred pgvector credentials!")

        connection_string: str = (
            f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        from .pgvector.PGVectorAdapter import PGVectorAdapter

        return PGVectorAdapter(
            connection_string,
            vector_db_key,
            embedding_engine,
        )

    elif vector_db_provider == "falkordb":
        if not (vector_db_url and vector_db_port):
            raise EnvironmentError("Missing requred FalkorDB credentials!")

        from ..hybrid.falkordb.FalkorDBAdapter import FalkorDBAdapter

        return FalkorDBAdapter(
            database_url=vector_db_url,
            database_port=vector_db_port,
            embedding_engine=embedding_engine,
        )

    elif vector_db_provider == "chromadb":
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

    elif vector_db_provider == "opensearch":
        from .opensearch.OpenSearchAdapter import OpenSearchAdapter

        if not vector_db_url:
            raise EnvironmentError("Missing required OpenSearch hosts!")

        # hosts pode ser passado como string separada por vírgula ou lista
        hosts = [h.strip() for h in vector_db_url.split(",")] if isinstance(vector_db_url, str) else vector_db_url
        http_auth = None
        if vector_db_key:
            vector_db_key_decoded = base64.b64decode(vector_db_key).decode("utf-8")
            vector_db_key_decoded_dict = json.loads(vector_db_key_decoded)
            username = vector_db_key_decoded_dict.get("username")
            password = vector_db_key_decoded_dict.get("password")
            if username and password:
                http_auth = (username, password)
            use_ssl = vector_db_key_decoded_dict.get("use_ssl", "False").lower() == "true"
            verify_certs = vector_db_key_decoded_dict.get("verify_certs", "True").lower() == "true"
            ssl_assert_hostname = vector_db_key_decoded_dict.get("ssl_assert_hostname", "True").lower() == "true"
            ssl_show_warn = vector_db_key_decoded_dict.get("ssl_show_warn", "True").lower() == "true"
            index_prefix = vector_db_key_decoded_dict.get("index_prefix", "")

        return OpenSearchAdapter(
            hosts=hosts,
            embedding_engine=embedding_engine,
            http_auth=http_auth,
            index_prefix=f"{index_prefix}cognee",
            **{
                "use_ssl": use_ssl,
                "verify_certs": verify_certs,
                "ssl_assert_hostname": ssl_assert_hostname,
                "ssl_show_warn": ssl_show_warn,
            }
        )

    else:
        from .lancedb.LanceDBAdapter import LanceDBAdapter

        return LanceDBAdapter(
            url=vector_db_url,
            api_key=vector_db_key,
            embedding_engine=embedding_engine,
        )
