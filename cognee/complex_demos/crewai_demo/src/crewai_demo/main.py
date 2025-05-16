import warnings
import os
from hiring_crew import HiringCrew
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.github_ingestion import (
    GithubIngestion,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def print_environment():
    for key in sorted(os.environ):
        print(f"{key}={os.environ[key]}")


def run_github_ingestion(applicant_1, applicant_2):
    GithubIngestion().run(applicant_1=applicant_1, applicant_2=applicant_2)


def run_hiring_crew(applicants):
    HiringCrew(inputs=applicants).crew().kickoff()


def run(enable_ingestion=True, enable_crew=True):
    try:
        print_environment()

        applicants = {"applicant_1": "hajdul88", "applicant_2": "lxobr"}

        if enable_ingestion:
            run_github_ingestion(applicants["applicant_1"], applicants["applicant_2"])

        if enable_crew:
            run_hiring_crew(applicants)

    except Exception as e:
        raise Exception(f"An error occurred while running the process: {e}")


if __name__ == "__main__":
    enable_ingestion = True
    enable_crew = False

    run(enable_ingestion=enable_ingestion, enable_crew=enable_crew)
