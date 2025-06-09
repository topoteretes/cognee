from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)


async def setup():
    """
    Set up the necessary databases and tables.

    This function asynchronously creates a relational database and its corresponding tables,
    followed by creating a PGVector database and its tables.
    """
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()
