from sqlalchemy import select, text

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils.get_graph_dataset_database_handler import (
    get_graph_dataset_database_handler,
)
from cognee.infrastructure.databases.utils.get_vector_dataset_database_handler import (
    get_vector_dataset_database_handler,
)
from cognee.modules.data.models import Data, Dataset
from cognee.modules.users.models import DatasetDatabase


async def delete_dataset(dataset: Dataset):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        if db_engine.engine.dialect.name == "sqlite":
            # Foreign key constraints are disabled by default in SQLite (for backwards compatibility),
            # so must be enabled for each database connection/session separately.
            await session.execute(text("PRAGMA foreign_keys = ON;"))

        stmt = select(DatasetDatabase).where(
            DatasetDatabase.dataset_id == dataset.id,
        )
        dataset_database: DatasetDatabase = await session.scalar(stmt)

        # Rows are dataset-scoped: deleting the dataset deletes its documents.
        # (Stale pipeline_status can't survive because the rows themselves go.)
        data_ids = (
            (await session.execute(select(Data.id).where(Data.dataset_id == dataset.id)))
            .scalars()
            .all()
        )

        # No index build follows a dataset delete, so its routing rows go here,
        # while the data ids they derive from are still known, and before the
        # dataset's own stores are dropped: a routing store that cannot be
        # reached aborts the delete with everything still in place, so it can
        # be retried, instead of leaving rows and files whose stores are gone.
        # Imported here: the routing package reads this package, which imports
        # this module.
        from cognee.modules.search.dataset_routing import remove_dataset_routes

        await remove_dataset_routes(dataset.id, data_ids)

        if dataset_database:
            graph_dataset_database_handler = get_graph_dataset_database_handler(dataset_database)
            vector_dataset_database_handler = get_vector_dataset_database_handler(dataset_database)
            await graph_dataset_database_handler["handler_instance"].delete_dataset(
                dataset_database
            )
            await vector_dataset_database_handler["handler_instance"].delete_dataset(
                dataset_database
            )
        await session.commit()

    # Files are refcounted by raw_data_location inside delete_data_entity, so
    # content shared with other datasets' rows keeps its file.
    for data_id in data_ids:
        await db_engine.delete_data_entity(data_id, dataset.id)

    # Use ORM session.delete() instead of raw table reflection with hardcoded
    # schema.  The previous ``delete_entity_by_id`` call reflected the table
    # using ``schema_name="public"`` by default, which fails when PostgreSQL
    # tables live in a non-public schema (e.g. ``cognee``).  session.delete()
    # resolves the table through SQLAlchemy metadata and triggers ORM cascades
    # (e.g. deleting related ACL rows via cascade="all, delete-orphan").
    async with db_engine.get_async_session() as session:
        if db_engine.engine.dialect.name == "sqlite":
            # Foreign key constraints are disabled by default in SQLite (for backwards compatibility),
            # so must be enabled for each database connection/session separately.
            await session.execute(text("PRAGMA foreign_keys = ON;"))

        dataset = await session.get(Dataset, dataset.id)
        if dataset:
            await session.delete(dataset)
            await session.commit()
