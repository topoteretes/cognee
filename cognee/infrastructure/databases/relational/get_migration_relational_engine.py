from .config import get_migration_config
from .create_relational_engine import create_relational_engine


def get_migration_relational_engine():
    """
    Create and return a relational database engine using the migration configuration.

    This function fetches the migration configuration details and uses them to create a
    relational database engine. It does not explicitly handle exceptions but may propagate
    any errors raised in the underlying functions.

    Returns:
    --------

        A relational database engine instance configured with the migration settings.
    """
    migration_config = get_migration_config()

    return create_relational_engine(
        db_path=migration_config.migration_db_path,
        db_name=migration_config.migration_db_name,
        db_host=migration_config.migration_db_host,
        db_port=migration_config.migration_db_port,
        db_username=migration_config.migration_db_username,
        db_password=migration_config.migration_db_password,
        db_provider=migration_config.migration_db_provider,
    )
