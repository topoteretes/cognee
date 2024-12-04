import dlt
import cognee.modules.ingestion as ingestion
from typing import Any
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import create_dataset
from cognee.modules.users.permissions.methods import give_permission_on_document
from .get_dlt_destination import get_dlt_destination
from .save_data_item_with_metadata_to_storage import save_data_item_with_metadata_to_storage

async def ingest_data_with_metadata(data: Any, dataset_name: str, user: User):
    destination = get_dlt_destination()

    pipeline = dlt.pipeline(
        pipeline_name = "file_load_from_filesystem",
        destination = destination,
    )

    @dlt.resource(standalone = True, merge_key = "id")
    async def data_resources(data: Any, user: User):
        if not isinstance(data, list):
            # Convert data to a list as we work with lists further down.
            data = [data]

        # Process data
        for data_item in data:

            file_path = save_data_item_with_metadata_to_storage(data_item, dataset_name)

            # Ingest data and add metadata
            with open(file_path.replace("file://", ""), mode = "rb") as file:
                classified_data = ingestion.classify(file)

                data_id = ingestion.identify(classified_data)

                file_metadata = classified_data.get_metadata()

                from sqlalchemy import select
                from cognee.modules.data.models import Data

                db_engine = get_relational_engine()

                async with db_engine.get_async_session() as session:
                    dataset = await create_dataset(dataset_name, user.id, session)

                    data_point = (await session.execute(
                        select(Data).filter(Data.id == data_id)
                    )).scalar_one_or_none()

                    if data_point is not None:
                        data_point.name = file_metadata["name"]
                        data_point.raw_data_location = file_metadata["file_path"]
                        data_point.extension = file_metadata["extension"]
                        data_point.mime_type = file_metadata["mime_type"]

                        await session.merge(data_point)
                        await session.commit()
                    else:
                        data_point = Data(
                            id = data_id,
                            name = file_metadata["name"],
                            raw_data_location = file_metadata["file_path"],
                            extension = file_metadata["extension"],
                            mime_type = file_metadata["mime_type"],
                        )

                        dataset.data.append(data_point)
                        await session.commit()

                yield {
                    "id": data_id,
                    "name": file_metadata["name"],
                    "file_path": file_metadata["file_path"],
                    "extension": file_metadata["extension"],
                    "mime_type": file_metadata["mime_type"],
                }

                await give_permission_on_document(user, data_id, "read")
                await give_permission_on_document(user, data_id, "write")


    send_telemetry("cognee.add EXECUTION STARTED", user_id = user.id)
    run_info = pipeline.run(
        data_resources(data, user),
        table_name = "file_metadata",
        dataset_name = dataset_name,
        write_disposition = "merge",
    )
    send_telemetry("cognee.add EXECUTION COMPLETED", user_id = user.id)

    return run_info
