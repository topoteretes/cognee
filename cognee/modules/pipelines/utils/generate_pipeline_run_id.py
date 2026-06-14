from uuid import UUID, uuid4


def generate_pipeline_run_id(pipeline_id: UUID, dataset_id: UUID):
    # pipeline_run_id must be unique per execution; keep args for API compatibility.
    return uuid4()
