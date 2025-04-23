from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from .tools import CogneeAdd, CogneeSearch

from crewai_tools import (
    DirectoryReadTool
)
import os

docs_tool = DirectoryReadTool(directory='/Users/vasilije/cognee/examples/python/latest_ai_development/src/latest_ai_development/multimedia')

# Utility function to format paths with file:// prefix
def format_file_paths(paths):
    """
    Formats a list of file paths with 'file://' prefix
    
    Args:
        paths: A string representing the output of DirectoryReadTool containing file paths
        
    Returns:
        A formatted string where each path is prefixed with 'file://'
    """
    if isinstance(paths, str):
        # Split the paths by newline if it's a string output
        file_list = [line for line in paths.split('\n') if line.strip()]
        # Format each path with file:// prefix
        formatted_paths = [f"file://{os.path.abspath(path.strip())}" for path in file_list if "File paths:" not in path]
        return '\n'.join(formatted_paths)
    return paths


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class LatestAiDevelopment:
    """LatestAiDevelopment crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        # Initialize the tools with different nodesets
        cognee_search = CogneeSearch()
        
        # CogneeAdd for documents with a "documents" nodeset
        documents_cognee_add = CogneeAdd()
        documents_cognee_add.default_nodeset = ["documents"]
        documents_cognee_add.name = "Add Documents to Memory"
        documents_cognee_add.description = "Add document content to Cognee memory with documents nodeset"
        
        # CogneeAdd for reasoning/analysis with a "reasoning" nodeset
        reasoning_cognee_add = CogneeAdd()
        reasoning_cognee_add.default_nodeset = ["reasoning"]
        reasoning_cognee_add.name = "Add Reasoning to Memory"
        reasoning_cognee_add.description = "Add reasoning and analysis text to Cognee memory with reasoning nodeset"
        
        # Create a wrapper for the DirectoryReadTool that formats output
        class FormattedDirectoryReadTool(DirectoryReadTool):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                return format_file_paths(result)
        
        formatted_docs_tool = FormattedDirectoryReadTool(directory='/Users/vasilije/cognee/examples/python/latest_ai_development/src/latest_ai_development/multimedia')

        return Agent(
            config=self.agents_config["researcher"], 
            tools=[formatted_docs_tool, documents_cognee_add, reasoning_cognee_add, cognee_search],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        # Initialize the tools with default parameters
        cognee_search = CogneeSearch()
        
        # Reporting analyst can use a "reports" nodeset
        reports_cognee_add = CogneeAdd()
        reports_cognee_add.default_nodeset = ["reports"]
        reports_cognee_add.name = "Add Reports to Memory"
        reports_cognee_add.description = "Add report content to Cognee memory with reports nodeset"

        return Agent(
            config=self.agents_config["reporting_analyst"],
            tools=[cognee_search, reports_cognee_add],
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )