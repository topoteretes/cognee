from .sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter


def create_relational_engine(
    db_path: str,
    db_name: str,
    db_provider: str,
    db_host: str,
    db_port: str,
    db_user: str,
    db_password: str,
):
    # print(f"DB_NAME={db_name}")
    # print(f"DB_USER={db_user}")
    # print(f"DB_PASSWORD={db_password}")
    # print(f"DB_HOST={db_host}")
    # print(f"DB_PORT={db_port}")

    return SQLAlchemyAdapter(
        db_name = db_name,
        db_path = db_path,
        db_type = db_provider,
        db_host = db_host,
        db_port = db_port,
        db_user = db_user,
        db_password = db_password
    )
