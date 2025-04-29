from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
import os
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_build import CogneeBuild
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_search import CogneeSearch


@CrewBase
class IngestionCrew:
    @before_kickoff
    def dump_env(self, *args, **kwargs):
        print("=== Environment Variables ===")
        for key in sorted(os.environ):
            print(f"{key}={os.environ[key]}")

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def test_agent(self) -> Agent:
        print(self.agents_config)
        return Agent(
            config=self.agents_config["test_agent"],
            tools=[CogneeBuild(), CogneeSearch()],
            verbose=True,
            allow_delegation=True,
        )

    @task
    def preliminary(self) -> Task:
        return Task(config=self.tasks_config["preliminary_task"], async_execution=False)

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
