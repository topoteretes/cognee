from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
import os


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
    def ingestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["ingestion_agent"],
            verbose=True,
            allow_delegation=True,
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
