import dlt
import cognee.modules.ingestion as ingestion

from uuid import UUID
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import create_dataset
from cognee.modules.users.permissions.methods import give_permission_on_document
from .get_dlt_destination import get_dlt_destination


async def ingest_data(file_paths: list[str], dataset_name: str, user: User):
    destination = get_dlt_destination()

    pipeline = dlt.pipeline(
        pipeline_name="file_load_from_filesystem",
        destination=destination,
    )

    @dlt.resource(standalone=True, merge_key="id")
    async def data_resources(file_paths: str):
        for file_path in file_paths:
            with open(file_path.replace("file://", ""), mode="rb") as file:
                classified_data = ingestion.classify(file)
                data_id = ingestion.identify(classified_data)
                file_metadata = classified_data.get_metadata()
                yield {
                    "id": data_id,
                    "name": file_metadata["name"],
                    "file_path": file_metadata["file_path"],
                    "extension": file_metadata["extension"],
                    "mime_type": file_metadata["mime_type"],
                }

    async def data_storing(table_name, dataset_name, user: User):
        db_engine = get_relational_engine()

        async with db_engine.get_async_session() as session:
            # Read metadata stored with dlt
            files_metadata = await db_engine.get_all_data_from_table(table_name, dataset_name)
            for file_metadata in files_metadata:
                from sqlalchemy import select
                from cognee.modules.data.models import Data

                dataset = await create_dataset(dataset_name, user.id, session)

                data = (
                    await session.execute(select(Data).filter(Data.id == UUID(file_metadata["id"])))
                ).scalar_one_or_none()

                if data is not None:
                    data.name = file_metadata["name"]
                    data.raw_data_location = file_metadata["file_path"]
                    data.extension = file_metadata["extension"]
                    data.mime_type = file_metadata["mime_type"]

                    await session.merge(data)
                    await session.commit()
                else:
                    data = Data(
                        id=UUID(file_metadata["id"]),
                        name=file_metadata["name"],
                        raw_data_location=file_metadata["file_path"],
                        extension=file_metadata["extension"],
                        mime_type=file_metadata["mime_type"],
                    )

                    dataset.data.append(data)
                    await session.commit()

                await give_permission_on_document(user, UUID(file_metadata["id"]), "read")
                await give_permission_on_document(user, UUID(file_metadata["id"]), "write")

    send_telemetry("cognee.add EXECUTION STARTED", user_id=user.id)

    db_engine = get_relational_engine()

    # Note: DLT pipeline has its own event loop, therefore objects created in another event loop
    # can't be used inside the pipeline
    if db_engine.engine.dialect.name == "sqlite":
        # To use sqlite with dlt dataset_name must be set to "main".
        # Sqlite doesn't support schemas
        run_info = pipeline.run(
            data_resources(file_paths),
            table_name="file_metadata",
            dataset_name="main",
            write_disposition="merge",
        )
    else:
        run_info = pipeline.run(
            data_resources(file_paths),
            table_name="file_metadata",
            dataset_name=dataset_name,
            write_disposition="merge",
        )

    await data_storing("file_metadata", dataset_name, user)
    send_telemetry("cognee.add EXECUTION COMPLETED", user_id=user.id)

    return run_info
