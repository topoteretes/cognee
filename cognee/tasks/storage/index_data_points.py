import asyncio

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint

logger = get_logger("index_data_points")


async def index_data_points(data_points: list[DataPoint]):
    """Index data points in the vector engine by creating embeddings for specified fields.

    Process:
    1. Groups data points into a nested dict: {type_name: {field_name: [points]}}
    2. Creates vector indexes for each (type, field) combination on first encounter
    3. Batches points per (type, field) and creates async indexing tasks
    4. Executes all indexing tasks in parallel for efficient embedding generation

    Args:
        data_points: List of DataPoint objects to index. Each DataPoint's metadata must
                     contain an 'index_fields' list specifying which fields to embed.

    Returns:
        The original data_points list.
    """
    data_points_by_type = {}

    vector_engine = get_vector_engine()

    for data_point in data_points:
        data_point_type = type(data_point)
        type_name = data_point_type.__name__

        for field_name in data_point.metadata["index_fields"]:
            if getattr(data_point, field_name, None) is None:
                continue

            if type_name not in data_points_by_type:
                data_points_by_type[type_name] = {}

            if field_name not in data_points_by_type[type_name]:
                await vector_engine.create_vector_index(type_name, field_name)
                data_points_by_type[type_name][field_name] = []

            indexed_data_point = data_point.model_copy()
            indexed_data_point.metadata["index_fields"] = [field_name]
            data_points_by_type[type_name][field_name].append(indexed_data_point)

    batch_size = vector_engine.embedding_engine.get_batch_size()

    batches = (
        (type_name, field_name, points[i : i + batch_size])
        for type_name, fields in data_points_by_type.items()
        for field_name, points in fields.items()
        for i in range(0, len(points), batch_size)
    )

    tasks = [
        asyncio.create_task(vector_engine.index_data_points(type_name, field_name, batch_points))
        for type_name, field_name, batch_points in batches
    ]

    await asyncio.gather(*tasks)

    return data_points


async def get_data_points_from_model(
    data_point: DataPoint, added_data_points=None, visited_properties=None
) -> list[DataPoint]:
    data_points = []
    added_data_points = added_data_points or {}
    visited_properties = visited_properties or {}

    for field_name, field_value in data_point:
        if isinstance(field_value, DataPoint):
            property_key = f"{str(data_point.id)}{field_name}{str(field_value.id)}"

            if property_key in visited_properties:
                return []

            visited_properties[property_key] = True

            new_data_points = await get_data_points_from_model(
                field_value, added_data_points, visited_properties
            )

            for new_point in new_data_points:
                if str(new_point.id) not in added_data_points:
                    added_data_points[str(new_point.id)] = True
                    data_points.append(new_point)

        if (
            isinstance(field_value, list)
            and len(field_value) > 0
            and isinstance(field_value[0], DataPoint)
        ):
            for field_value_item in field_value:
                property_key = f"{str(data_point.id)}{field_name}{str(field_value_item.id)}"

                if property_key in visited_properties:
                    return []

                visited_properties[property_key] = True

                new_data_points = await get_data_points_from_model(
                    field_value_item, added_data_points, visited_properties
                )

                for new_point in new_data_points:
                    if str(new_point.id) not in added_data_points:
                        added_data_points[str(new_point.id)] = True
                        data_points.append(new_point)

    if str(data_point.id) not in added_data_points:
        data_points.append(data_point)

    return data_points


if __name__ == "__main__":

    class Car(DataPoint):
        model: str
        color: str
        metadata: dict = {"index_fields": ["name"]}

    class Person(DataPoint):
        name: str
        age: int
        owns_car: list[Car]
        metadata: dict = {"index_fields": ["name"]}

    car1 = Car(model="Tesla Model S", color="Blue")
    car2 = Car(model="Toyota Camry", color="Red")
    person = Person(name="John", age=30, owns_car=[car1, car2])

    data_points = get_data_points_from_model(person)

    print(data_points)
