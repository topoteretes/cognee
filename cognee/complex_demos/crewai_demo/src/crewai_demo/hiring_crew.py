import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_search import CogneeSearch


@CrewBase
class HiringCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, inputs):
        self.inputs = inputs
        self

    @agent
    def soft_skills_expert_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["soft_skills_expert_agent"],
            tools=[CogneeSearch(nodeset_name="soft")],
            verbose=True,
        )

    @agent
    def technical_expert_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_expert_agent"],
            tools=[CogneeSearch(nodeset_name="technical")],
            verbose=True,
        )

    @agent
    def decision_maker_agent(self) -> Agent:
        return Agent(config=self.agents_config["decision_maker_agent"], verbose=True)

    @task
    def soft_skills_assessment_applicant1_task(self) -> Task:
        self.tasks_config["soft_skills_assessment_applicant1_task"]["description"] = (
            self.tasks_config["soft_skills_assessment_applicant1_task"]["description"].format(
                **self.inputs
            )
        )
        return Task(
            config=self.tasks_config["soft_skills_assessment_applicant1_task"],
            async_execution=False,
        )

    @task
    def soft_skills_assessment_applicant2_task(self) -> Task:
        self.tasks_config["soft_skills_assessment_applicant2_task"]["description"] = (
            self.tasks_config["soft_skills_assessment_applicant2_task"]["description"].format(
                **self.inputs
            )
        )
        return Task(
            config=self.tasks_config["soft_skills_assessment_applicant2_task"],
            async_execution=False,
        )

    @task
    def technical_assessment_task(self) -> Task:
        self.tasks_config["technical_assessment_task"]["description"] = self.tasks_config[
            "technical_assessment_task"
        ]["description"].format(**self.inputs)
        return Task(config=self.tasks_config["technical_assessment_task"], async_execution=False)

    @task
    def hiring_decision_task(self) -> Task:
        self.tasks_config["hiring_decision_task"]["description"] = self.tasks_config[
            "hiring_decision_task"
        ]["description"].format(**self.inputs)
        return Task(config=self.tasks_config["hiring_decision_task"], async_execution=False)

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            share_crew=True,
            output_log_file="hiring_crew_log.txt",
        )
