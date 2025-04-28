from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from custom_tools.cognee_add import CogneeAdd
from custom_tools.cognee_cognify import CogneeCognify
from custom_tools.cognee_search import CogneeSearch
from crewai_tools import SerperDevTool
import os


@CrewBase
class IngestionCrew:
    @before_kickoff
    def dump_env(self, *args, **kwargs):
        """Print environment variables at startup."""
        print("=== Environment Variables ===")
        for key in sorted(os.environ):
            print(f"{key}={os.environ[key]}")

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def ingestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["ingestion_agent"],
            tools=[CogneeAdd(), CogneeCognify(), SerperDevTool()],
            verbose=True,
            allow_delegation=True,
        )

    @agent
    def search_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["search_agent"], tools=[CogneeSearch()], verbose=True
        )

    @task
    def search_on_google(self) -> Task:
        return Task(config=self.tasks_config["google_task"], async_execution=False)

    @task
    def cognify(self) -> Task:
        return Task(
            config=self.tasks_config["cognify_task"],
            async_execution=False,
        )

    @task
    def search(self) -> Task:
        return Task(config=self.tasks_config["search_task"], async_execution=False)

    @crew
    def crew(self) -> Crew:
        print(self.tasks)
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            share_crew=True,
            output_log_file="logs.txt",
        )
