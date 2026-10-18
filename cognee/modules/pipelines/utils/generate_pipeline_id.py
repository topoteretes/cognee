from uuid import NAMESPACE_OID, UUID, uuid5


def generate_pipeline_id(user_id: UUID, dataset_id: UUID, pipeline_name: str):
    return uuid5(NAMESPACE_OID, f"{user_id!s}{pipeline_name}{dataset_id!s}")
