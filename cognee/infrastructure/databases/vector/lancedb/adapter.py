import typing
from qdrant_client import AsyncQdrantClient, models
from databases.vector.vector_db_interface import VectorDBInterface

class VectorConfig(extra='forbid'):
    size: int
    distance: str
    on_disk: bool

class CollectionConfig(extra='forbid'):
    vector_config: VectorConfig
    hnsw_config: models.HnswConfig
    optimizers_config: models.OptimizersConfig
    quantization_config: models.QuantizationConfig

class LanceDBAdapter(VectorDBInterface):
    def __init__(self, lancedb_url, lancedb_api_key):
        self.lancedb_url = lancedb_url
        self.lancedb_api_key = lancedb_api_key

    def get_lancedb_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url = self.lancedb_url,
            api_key = self.lancedb_api_key,
            location = ':memory:'
        )

    async def create_collection(
      self,
      collection_name: str,
      collection_config: CollectionConfig
    ):
        client = self.get_lancedb_client()

        return await client.create_collection(
            collection_name = collection_name,
            vectors_config = collection_config.vector_config,
            hnsw_config = collection_config.hnsw_config,
            optimizers_config = collection_config.optimizers_config,
            quantization_config = collection_config.quantization_config
        )

    async def create_data_points(self, collection_name: str, data_points: typing.List[any]):
        client = self.get_lancedb_client()

        async def create_data_point(data):
            return {
                'vector': {},
                'payload': data
            }

        return await client.upload_points(
            collection_name = collection_name,
            points = map(create_data_point, data_points)
        )


# class LanceDB(VectorDB):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.db = self.init_lancedb()

#     def init_lancedb(self):
#         # Initialize LanceDB connection
#         # Adjust the URI as needed for your LanceDB setup
#         uri = "s3://my-bucket/lancedb" if self.namespace else "~/.lancedb"
#         db = lancedb.connect(uri, api_key=os.getenv("LANCEDB_API_KEY"))
#         return db

#     def create_table(
#         self,
#         name: str,
#         schema: Optional[pa.Schema] = None,
#         data: Optional[pd.DataFrame] = None,
#     ):
#         # Create a table in LanceDB. If schema is not provided, it will be inferred from the data.
#         if data is not None and schema is None:
#             schema = pa.Schema.from_pandas(data)
#         table = self.db.create_table(name, schema=schema)
#         if data is not None:
#             table.add(data.to_dict("records"))
#         return table

#     def add_memories(self, table_name: str, data: pd.DataFrame):
#         # Add data to an existing table in LanceDB
#         table = self.db.open_table(table_name)
#         table.add(data.to_dict("records"))

#     def fetch_memories(
#         self, table_name: str, query_vector: List[float], top_k: int = 10
#     ):
#         # Perform a vector search in the specified table
#         table = self.db.open_table(table_name)
#         results = table.search(query_vector).limit(top_k).to_pandas()
#         return results
