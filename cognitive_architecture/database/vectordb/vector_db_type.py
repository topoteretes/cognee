from enum import Enum

class VectorDBType(Enum):
    PINECONE = 'pinecone'
    WEAVIATE = 'weaviate'
    MILVUS = 'milvus'
    QDRANT = 'qdrant'
    DEEPLAKE = 'deeplake'
    VESPA = 'vespa'
    PGVECTOR = 'pgvector'
    REDIS = 'redis'
    LANCEDB = 'lancedb'
    MONGODB = 'mongodb'