from typing import Dict

from ..relational.config import get_relational_config

class VectorConfig(Dict):
    vector_db_url: str
    vector_db_key: str
    vector_db_provider: str
    vector_db_name: str

def create_vector_engine(config: VectorConfig, embedding_engine):
    if config["vector_db_provider"] == "weaviate":
        from .weaviate_db import WeaviateAdapter

        if config["vector_db_url"] is None and config["vector_db_key"] is None:
            raise EnvironmentError("Weaviate is not configured!")

        return WeaviateAdapter(
            config["vector_db_url"],
            config["vector_db_key"],
            embedding_engine = embedding_engine
        )
    elif config["vector_db_provider"] == "qdrant":
        if config["vector_db_url"] and config["vector_db_key"]:
            from .qdrant.QDrantAdapter import QDrantAdapter

            return QDrantAdapter(
                url = config["vector_db_url"],
                api_key = config["vector_db_key"],
                embedding_engine = embedding_engine
            )
    elif config["vector_db_provider"] == "pgvector":
        from .pgvector.PGVectorAdapter import PGVectorAdapter
        
        # Get configuration for postgres database
        relational_config = get_relational_config()
        db_username = relational_config.db_username
        db_password = relational_config.db_password
        db_host = relational_config.db_host
        db_port = relational_config.db_port

        # Get name of vector database
        db_name = config["vector_db_name"]

        connection_string: str = f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        return PGVectorAdapter(connection_string, 
                        config["vector_db_key"], 
                        embedding_engine
        )
    else:
        from .lancedb.LanceDBAdapter import LanceDBAdapter

        return LanceDBAdapter(
            url = config["vector_db_url"],
            api_key = config["vector_db_key"],
            embedding_engine = embedding_engine,
        )

    raise EnvironmentError(f"Vector provider not configured correctly: {config['vector_db_provider']}")
