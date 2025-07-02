from uuid import NAMESPACE_OID, UUID, uuid5


def generate_pipeline_run_id(pipeline_id: UUID, dataset_id: UUID):
    return uuid5(NAMESPACE_OID, f"{str(pipeline_id)}_{str(dataset_id)}")
