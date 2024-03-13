
from cognee.infrastructure.databases.relational import DuckDBAdapter

def list_datasets():
    db = DuckDBAdapter()
    return db.get_datasets()
