from cognitive_architecture.config import Config
from .qdrant import QDrantAdapter

config = Config()
config.load()

def get_vector_database():
    return QDrantAdapter(config.qdrant_path, config.qdrant_url, config.qdrant_api_key)
