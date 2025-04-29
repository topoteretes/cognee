#!/usr/bin/env python
import warnings
from ingestion_crew import IngestionCrew
# from crewai_demo.ingestion_crew import IngestionCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    try:
        IngestionCrew().crew().kickoff()
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()
