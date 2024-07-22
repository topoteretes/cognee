from enum import Enum

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational import DuckDBAdapter


class DBProvider(Enum):
    DUCKDB = "duckdb"
    POSTGRES = "postgresql+asyncpg"

def create_relational_engine(db_path: str, db_name: str, db_provider:str, db_host:str, db_port:str, db_user:str, db_password:str):
    LocalStorage.ensure_directory_exists(db_path)

    provider = DBProvider(db_provider)

    if provider == DBProvider.DUCKDB:
        # return DuckDBAdapter(
        #     db_name = db_name,
        #     db_path = db_path,
        # )
        return SQLAlchemyAdapter(
            db_name = db_name,
            db_path = db_path,
            db_type = db_provider,
            db_host=db_host,
            db_port=db_port,
            db_user=db_user,
            db_password=db_password
        )
    elif provider == DBProvider.POSTGRES:
        return SQLAlchemyAdapter(
            db_name = db_name,
            db_path = db_path,
            db_type = db_provider,
            db_host= db_host,
            db_port= db_port,
            db_user= db_user,
            db_password= db_password
        )