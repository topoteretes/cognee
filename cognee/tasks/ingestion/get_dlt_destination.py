import os
from functools import lru_cache

try:
    import dlt
except ImportError:
    dlt = None
from typing import Optional, Any

from cognee.infrastructure.databases.relational import get_relational_config


@lru_cache
def get_dlt_destination(dlt_db_name: str) -> Optional[Any]:
    """
    Handle the propagation of the cognee database configuration to the dlt library.

    This function determines the appropriate sqlalchemy destination based on the database
    provider specified in the relational configuration. It constructs the destination
    credentials for either sqlite or postgres databases accordingly. If the database
    provider is neither sqlite nor postgres, it returns None.

    Returns:
    --------

        - Union[type[dlt.destinations.sqlalchemy], None]: An instance of sqlalchemy
          destination used by the dlt library, or None if the database provider is
          unsupported.
    """
    relational_config = get_relational_config()

    if relational_config.db_provider == "sqlite":
        # When sqlite is the database provider hostname, port, username and password should not be forwarded.
        # The database is found  by combining the path location and the database name
        destination = dlt.destinations.sqlalchemy(
            credentials={
                "database": os.path.join(relational_config.db_path, dlt_db_name),
                "drivername": relational_config.db_provider,
            },
        )
    elif relational_config.db_provider == "postgres":
        # The dlt library doesn't accept postgres as the drivername, it only accepts postgresql
        destination = dlt.destinations.sqlalchemy(
            credentials={
                "host": relational_config.db_host,
                "port": relational_config.db_port,
                "username": relational_config.db_username,
                "password": relational_config.db_password,
                "database": dlt_db_name,
                "drivername": "postgresql",
            },
        )
    else:
        destination = None

    return destination
