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
    return SQLAlchemyAdapter(
        db_name = db_name,
        db_path = db_path,
        db_type = db_provider,
        db_host = db_host,
        db_port = db_port,
        db_user = db_user,
        db_password = db_password
    )
