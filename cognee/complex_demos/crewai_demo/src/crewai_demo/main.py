import os
import time
import warnings
from uuid import uuid4

from cognee.complex_demos.crewai_demo.src.crewai_demo.events.crewai_listener import CrewAIListener
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_ingest_datapoints import cognify_github_data_from_username
from cognee.modules.crewai.get_crewai_pipeline_run_id import get_crewai_pipeline_run_id
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunActivity
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from .hiring_crew import HiringCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def print_environment():
    for key in sorted(os.environ):
        print(f"{key}={os.environ[key]}")


async def run_github_ingestion(user, applicant_1, applicant_2):
    token = os.getenv("GITHUB_TOKEN")

    pipeline_run_id = get_crewai_pipeline_run_id(user.id)

    push_to_queue(pipeline_run_id, PipelineRunActivity(
        pipeline_run_id=pipeline_run_id,
        payload={
            "id": str(uuid4()),
            "timestamp": time.time() * 1000,
            "activity": "GitHub applicant data ingestion started",
        }
    ))

    await cognify_github_data_from_username(applicant_1, token)

    push_to_queue(pipeline_run_id, PipelineRunActivity(
        pipeline_run_id=pipeline_run_id,
        payload={
            "id": str(uuid4()),
            "timestamp": time.time() * 1000,
            "activity": f"Applicant's ({applicant_1}) data ingestion finished",
        }
    ))

    await cognify_github_data_from_username(applicant_2, token)

    push_to_queue(pipeline_run_id, PipelineRunActivity(
        pipeline_run_id=pipeline_run_id,
        payload={
            "id": str(uuid4()),
            "timestamp": time.time() * 1000,
            "activity": f"Applicant's ({applicant_2}) data ingestion finished",
        }
    ))

    push_to_queue(pipeline_run_id, PipelineRunActivity(
        pipeline_run_id=pipeline_run_id,
        payload={
            "id": str(uuid4()),
            "timestamp": time.time() * 1000,
            "activity": "GitHub applicant data ingestion finished",
        }
    ))


def run_hiring_crew(user, applicants: dict, number_of_rounds: int = 1):
    pipeline_run_id = get_crewai_pipeline_run_id(user.id)

    # Instantiate CrewAI listener to capture pipeline run events
    crewai_listener = CrewAIListener(pipeline_run_id=pipeline_run_id)

    push_to_queue(pipeline_run_id, PipelineRunActivity(
        pipeline_run_id=pipeline_run_id,
        payload={
            "id": str(uuid4()),
            "timestamp": time.time() * 1000,
            "activity": "Hiring crew research started",
        }
    ))

    for hiring_round in range(number_of_rounds):
        print(f"\nStarting hiring round {hiring_round + 1}...\n")

        push_to_queue(pipeline_run_id, PipelineRunActivity(
            pipeline_run_id=pipeline_run_id,
            payload={
                "id": str(uuid4()),
                "timestamp": time.time() * 1000,
                "activity": f"Research round {hiring_round + 1} started",
            }
        ))

        crew = HiringCrew(inputs=applicants)
        if hiring_round > 0:
            print("Refining agent prompts for this round...")

            push_to_queue(pipeline_run_id, PipelineRunActivity(
                pipeline_run_id=pipeline_run_id,
                payload={
                    "id": str(uuid4()),
                    "timestamp": time.time() * 1000,
                    "activity": "Refining agent prompts for the next round",
                }
            ))

            crew.refine_agent_configs(agent_name="soft_skills_expert_agent")
            crew.refine_agent_configs(agent_name="technical_expert_agent")
            crew.refine_agent_configs(agent_name="decision_maker_agent")

        crew.crew().kickoff()

    push_to_queue(pipeline_run_id, PipelineRunActivity(
        pipeline_run_id=pipeline_run_id,
        payload={
            "id": str(uuid4()),
            "timestamp": time.time() * 1000,
            "activity": "Hiring crew research finished",
        }
    ))


def run(enable_ingestion=True, enable_crew=True):
    try:
        print_environment()

        applicants = {"applicant_1": "hajdul88", "applicant_2": "lxobr"}

        if enable_ingestion:
            run_github_ingestion(applicants["applicant_1"], applicants["applicant_2"])

        if enable_crew:
            run_hiring_crew(applicants=applicants, number_of_rounds=5)

    except Exception as e:
        raise Exception(f"An error occurred while running the process: {e}")


if __name__ == "__main__":
    enable_ingestion = True
    enable_crew = True

    run(enable_ingestion=enable_ingestion, enable_crew=enable_crew)
