#!/usr/bin/env python
import warnings
import os
from ingestion_crew import IngestionCrew
import cognee
import asyncio
# from association_layer_demo.ingestion_crew import IngestionCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    try:
        IngestionCrew().crew().kickoff()
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()
