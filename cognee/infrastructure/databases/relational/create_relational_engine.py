from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational import DuckDBAdapter

def create_relational_engine(db_path: str, db_name: str):
    LocalStorage.ensure_directory_exists(db_path)

    return DuckDBAdapter(
        db_name = db_name,
        db_path = db_path,
    )
