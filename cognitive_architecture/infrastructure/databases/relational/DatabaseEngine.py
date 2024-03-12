from typing import Protocol

class DatabaseEngine(Protocol):
    async def ensure_tables(self):
        pass

    def database_exists(self, db_name: str) -> bool:
        pass

    def create_database(self, db_name: str):
        pass

    def drop_database(self, db_name: str):
        pass

    async def table_exists(self, table_name: str) -> bool:
        pass

    async def create_tables(self):
        pass

    async def create(self, data):
        pass
