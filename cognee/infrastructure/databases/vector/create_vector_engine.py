from typing import Dict


class VectorConfig(Dict):
    vector_db_url: str
    vector_db_port: str
    vector_db_key: str
    vector_db_provider: str


def create_vector_engine(config: VectorConfig, embedding_engine):
    if config["vector_db_provider"] == "weaviate":
        from .weaviate_db import WeaviateAdapter

        if not (config["vector_db_url"] and config["vector_db_key"]):
            raise EnvironmentError("Missing requred Weaviate credentials!")

        return WeaviateAdapter(
            config["vector_db_url"], config["vector_db_key"], embedding_engine=embedding_engine
        )

    elif config["vector_db_provider"] == "qdrant":
        if not (config["vector_db_url"] and config["vector_db_key"]):
            raise EnvironmentError("Missing requred Qdrant credentials!")

        from .qdrant.QDrantAdapter import QDrantAdapter

        return QDrantAdapter(
            url=config["vector_db_url"],
            api_key=config["vector_db_key"],
            embedding_engine=embedding_engine,
        )

    elif config["vector_db_provider"] == "milvus":
        from .milvus.MilvusAdapter import MilvusAdapter

        if not config["vector_db_url"]:
            raise EnvironmentError("Missing required Milvus credentials!")

        return MilvusAdapter(
            url=config["vector_db_url"],
            api_key=config["vector_db_key"],
            embedding_engine=embedding_engine,
        )

    elif config["vector_db_provider"] == "pgvector":
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
            config["vector_db_key"],
            embedding_engine,
        )

    elif config["vector_db_provider"] == "falkordb":
        if not (config["vector_db_url"] and config["vector_db_port"]):
            raise EnvironmentError("Missing requred FalkorDB credentials!")

        from ..hybrid.falkordb.FalkorDBAdapter import FalkorDBAdapter

        return FalkorDBAdapter(
            database_url=config["vector_db_url"],
            database_port=config["vector_db_port"],
            embedding_engine=embedding_engine,
        )

    else:
        from .lancedb.LanceDBAdapter import LanceDBAdapter

        return LanceDBAdapter(
            url=config["vector_db_url"],
            api_key=config["vector_db_key"],
            embedding_engine=embedding_engine,
        )
