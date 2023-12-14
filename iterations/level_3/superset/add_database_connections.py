from superset_config import DATABASE_CONNECTIONS
from superset import app
from superset.extensions import db
from superset.models.core import Database


def add_database_connections():
    with app.app_context():
        for db_conn in DATABASE_CONNECTIONS:
            database_name = db_conn['database_name']
            sqlalchemy_uri = db_conn['sqlalchemy_uri']

            # Check if database already exists
            existing_db = db.session.query(Database).filter_by(database_name=database_name).first()
            if existing_db:
                print(f"Database {database_name} already exists, skipping")
                continue

            # Add new database connection
            new_db = Database(
                database_name=database_name,
                sqlalchemy_uri=sqlalchemy_uri,
            )
            db.session.add(new_db)
            db.session.commit()
            print(f"Added database: {database_name}")


if __name__ == "__main__":
    add_database_connections()


