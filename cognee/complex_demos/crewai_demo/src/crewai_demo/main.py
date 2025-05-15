#!/usr/bin/env python
import os
import warnings
import cognee
from cognee.modules.engine.models import NodeSet
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from hiring_crew import HiringCrew

# from crewai_demo.cognify_crew import CognifyCrew
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_search import CogneeSearch

from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_build import CogneeBuild
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.github_ingestion import (
    GithubIngestion,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    try:
        for key in sorted(os.environ):
            print(f"{key}={os.environ[key]}")

        applicant_1 = "lxobr"
        applicant_2 = "hajdul88"

        GithubIngestion().run(applicant_1=applicant_1, applicant_2=applicant_2)

        HiringCrew().crew().kickoff()

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    # Run the async entry point
    run()
