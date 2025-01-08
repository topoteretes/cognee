import os
from functools import lru_cache

import dlt
from typing import Union

from cognee.infrastructure.databases.relational import get_relational_config


@lru_cache
def get_dlt_destination() -> Union[type[dlt.destinations.sqlalchemy], None]:
    """
    Handles propagation of the cognee database configuration to the dlt library

    Returns:
        sqlachemy: sqlachemy destination used by the dlt library

    """
    relational_config = get_relational_config()

    if relational_config.db_provider == "sqlite":
        # When sqlite is the database provider hostname, port, username and password should not be forwarded.
        # The database is found  by combining the path location and the database name
        destination = dlt.destinations.sqlalchemy(
            credentials={
                "database": os.path.join(relational_config.db_path, relational_config.db_name),
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
                "database": relational_config.db_name,
                "drivername": "postgresql",
            },
        )
    else:
        destination = None

    return destination
