from cognee.infrastructure.engine import DataPoint


def convert_node_to_data_point(node_data: dict) -> DataPoint:
    subclass = find_subclass_by_name(DataPoint, node_data["type"])

    return subclass(**node_data)


def get_all_subclasses(cls):
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(get_all_subclasses(subclass))  # Recursively get subclasses

    return subclasses


def find_subclass_by_name(cls, name):
    for subclass in get_all_subclasses(cls):
        if subclass.__name__ == name:
            return subclass

    return None
