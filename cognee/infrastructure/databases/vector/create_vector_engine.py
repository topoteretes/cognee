from typing import Dict

class VectorConfig(Dict):
    vector_db_url: str
    vector_db_key: str
    vector_db_provider: str

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
    else:
        from .lancedb.LanceDBAdapter import LanceDBAdapter
        # from cognee.infrastructure.files.storage import LocalStorage

        # LocalStorage.ensure_directory_exists(config["vector_db_url"])

        return LanceDBAdapter(
            url = config["vector_db_url"],
            api_key = config["vector_db_key"],
            embedding_engine = embedding_engine,
        )
