from cognee.config import Config
# from .qdrant import QDrantAdapter
from .weaviate_db import WeaviateAdapter

config = Config()
config.load()

def get_vector_database():
    # return QDrantAdapter(config.qdrant_path, config.qdrant_url, config.qdrant_api_key)
    return WeaviateAdapter(config.weaviate_url, config.weaviate_api_key, config.openai_key)
