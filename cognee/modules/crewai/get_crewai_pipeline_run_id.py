from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.pipelines.utils import generate_pipeline_id, generate_pipeline_run_id


def get_crewai_pipeline_run_id(user_id: UUID):
    dataset_id = uuid5(NAMESPACE_OID, "Github")
    pipeline_id = generate_pipeline_id(user_id, "github_pipeline")
    pipeline_run_id = generate_pipeline_run_id(pipeline_id, dataset_id)

    return pipeline_run_id
