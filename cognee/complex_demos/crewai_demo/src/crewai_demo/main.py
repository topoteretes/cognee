#!/usr/bin/env python
import os
import warnings
import cognee
from hiring_crew import HiringCrew

# from crewai_demo.cognify_crew import CognifyCrew
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_search import CogneeSearch

from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_build import CogneeBuild

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    try:
        for key in sorted(os.environ):
            print(f"{key}={os.environ[key]}")

        data = {
            "comment_1": {
                "file_content": "Dean P: Hey I believe doing this as lambda expression would work better. What do you think?",
                "nodeset": ["soft"],
            },
            "comment_2": {
                "file_content": "Thomas M: Hey this feature is really not good. I dont care how just solve it.",
                "nodeset": ["soft"],
            },
            "code_1": {
                "file_content": """Author: Thomas M:
                                    user_code = input("Enter some Python code to run:")
                                    exec(user_code)""",
                "nodeset": ["technical"],
            },
            "code_2": {
                "file_content": """Author: Dean P:
                                with open('data.txt', 'r') as f:
                                    contents = f.read()
                                """,
                "nodeset": ["technical"],
            },
        }
        CogneeBuild().run(inputs=data)

        HiringCrew().crew().kickoff()

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    # Run the async entry point
    run()
