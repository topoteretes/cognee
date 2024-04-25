import asyncio
import lancedb
from typing import List, Optional

import asyncio
import lancedb
from pathlib import Path
import tempfile

class LanceDBAdapter:
    def __init__(self, uri: Optional[str] = None, api_key: Optional[str] = None):
        if uri:
            self.uri = uri
        else:
            # Create a temporary directory for the LanceDB 'in-memory' simulation
            self.temp_dir = tempfile.mkdtemp(suffix='.lancedb')
            self.uri = f"file://{self.temp_dir}"
        self.api_key = api_key
        self.db = None

    async def connect(self):
        # Asynchronously connect to a LanceDB database, effectively in-memory if no URI is provided
        self.db = await lancedb.connect_async(self.uri, api_key=self.api_key)

    async def disconnect(self):
        # Disconnect and clean up the database if it was set up as temporary
        await self.db.close()
        if hasattr(self, 'temp_dir'):
            Path(self.temp_dir).unlink(missing_ok=True)  # Remove the temporary directory

    async def create_table(self, table_name: str, schema=None, data=None):
        if not await self.table_exists(table_name):
            return await self.db.create_table(name=table_name, schema=schema, data=data)
        else:
            raise ValueError(f"Table {table_name} already exists")

    async def table_exists(self, table_name: str) -> bool:
        table_names = await self.db.table_names()
        return table_name in table_names

    async def insert_data(self, table_name: str, data_points: List[dict]):
        table = await self.db.open_table(table_name)
        await table.add(data_points)

    async def query_data(self, table_name: str, query=None, limit=10):
        # Asynchronously query data from a table
        table = await self.db.open_table(table_name)
        if query:
            query_result = await table.query().where(query).limit(limit).to_pandas()
        else:
            query_result = await table.query().limit(limit).to_pandas()
        return query_result

    async def vector_search(self, table_name: str, query_vector: List[float], limit=10):
        # Perform an asynchronous vector search
        table = await self.db.open_table(table_name)
        query_result = await table.vector_search().nearest_to(query_vector).limit(limit).to_pandas()
        return query_result


async def main():
    # Example without providing a URI, simulates in-memory behavior
    adapter = LanceDBAdapter()
    await adapter.connect()

    try:
        await adapter.create_table("my_table")
        data_points = [{"id": 1, "text": "example", "vector": [0.1, 0.2, 0.3]}]
        await adapter.insert_data("my_table", data_points)
    finally:
        await adapter.disconnect()

if __name__ == "__main__":
    asyncio.run(main())