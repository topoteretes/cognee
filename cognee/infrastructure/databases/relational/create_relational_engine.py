from enum import Enum

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational import DuckDBAdapter, get_relationaldb_config


class DBProvider(Enum):
    DUCKDB = "duckdb"
    POSTGRES = "postgres"



def create_relational_engine(db_path: str, db_name: str, db_provider:str):
    LocalStorage.ensure_directory_exists(db_path)

    llm_config = get_relationaldb_config()

    provider = DBProvider(llm_config.llm_provider)


    if provider == DBProvider.DUCKDB:

        return DuckDBAdapter(
            db_name = db_name,
            db_path = db_path,
        )
    elif provider == DBProvider.POSTGRES:
        return SQLAlchemyAdapter(
            db_name = db_name,
            db_path = db_path,
            db_type = db_provider,
        )