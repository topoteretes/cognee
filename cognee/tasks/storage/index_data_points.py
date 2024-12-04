from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint

async def index_data_points(data_points: list[DataPoint]):
    created_indexes = {}
    index_points = {}

    vector_engine = get_vector_engine()

    for data_point in data_points:
        data_point_type = type(data_point)

        for field_name in data_point._metadata["index_fields"]:
            if getattr(data_point, field_name, None) is None:
                continue

            index_name = f"{data_point_type.__tablename__}.{field_name}"

            if index_name not in created_indexes:
                await vector_engine.create_vector_index(data_point_type.__tablename__, field_name)
                created_indexes[index_name] = True

            if index_name not in index_points:
                index_points[index_name] = []

            indexed_data_point = data_point.model_copy()
            indexed_data_point._metadata["index_fields"] = [field_name]
            index_points[index_name].append(indexed_data_point)

    for index_name, indexable_points in index_points.items():
        index_name, field_name = index_name.split(".")
        await vector_engine.index_data_points(index_name, field_name, indexable_points)

    return data_points

async def get_data_points_from_model(data_point: DataPoint, added_data_points = None, visited_properties = None) -> list[DataPoint]:
    data_points = []
    added_data_points = added_data_points or {}
    visited_properties = visited_properties or {}

    for field_name, field_value in data_point:
        if isinstance(field_value, DataPoint):
            property_key = f"{str(data_point.id)}{field_name}{str(field_value.id)}"

            if property_key in visited_properties:
                return []

            visited_properties[property_key] = True

            new_data_points = await get_data_points_from_model(field_value, added_data_points, visited_properties)

            for new_point in new_data_points:
                if str(new_point.id) not in added_data_points:
                    added_data_points[str(new_point.id)] = True
                    data_points.append(new_point)

        if isinstance(field_value, list) and len(field_value) > 0 and isinstance(field_value[0], DataPoint):
            for field_value_item in field_value:
                property_key = f"{str(data_point.id)}{field_name}{str(field_value_item.id)}"

                if property_key in visited_properties:
                    return []

                visited_properties[property_key] = True
              
                new_data_points = await get_data_points_from_model(field_value_item, added_data_points, visited_properties)

                for new_point in new_data_points:
                    if str(new_point.id) not in added_data_points:
                        added_data_points[str(new_point.id)] = True
                        data_points.append(new_point)

    if (str(data_point.id) not in added_data_points):
        data_points.append(data_point)

    return data_points


if __name__ == "__main__":
    class Car(DataPoint):
        model: str
        color: str
  
    class Person(DataPoint):
        name: str
        age: int
        owns_car: list[Car]

    car1 = Car(model = "Tesla Model S", color = "Blue")
    car2 = Car(model = "Toyota Camry", color = "Red")
    person = Person(name = "John", age = 30, owns_car = [car1, car2])

    data_points = get_data_points_from_model(person)

    print(data_points)
