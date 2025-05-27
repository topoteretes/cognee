import warnings
import os
from .hiring_crew import HiringCrew
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.github_ingestion import (
    GithubIngestion,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def print_environment():
    for key in sorted(os.environ):
        print(f"{key}={os.environ[key]}")


def run_github_ingestion(applicant_1, applicant_2):
    return GithubIngestion().run(applicant_1=applicant_1, applicant_2=applicant_2)


def run_hiring_crew(applicants: dict, number_of_rounds: int = 1, llm_client=None):
    for hiring_round in range(number_of_rounds):
        print(f"\nStarting hiring round {hiring_round + 1}...\n")
        crew = HiringCrew(inputs=applicants)
        if hiring_round > 0:
            print("Refining agent prompts for this round...")
            crew.refine_agent_configs(agent_name="soft_skills_expert_agent")
            crew.refine_agent_configs(agent_name="technical_expert_agent")
            crew.refine_agent_configs(agent_name="decision_maker_agent")

        crew.crew().kickoff()


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
