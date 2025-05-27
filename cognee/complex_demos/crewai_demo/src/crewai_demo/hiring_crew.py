from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_ingestion import (
    CogneeIngestion,
)
from cognee.complex_demos.crewai_demo.src.crewai_demo.custom_tools.cognee_search import CogneeSearch
from cognee.infrastructure.llm.get_llm_client import get_llm_client


class AgentConfig(BaseModel):
    role: str
    goal: str
    backstory: str


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
        return Agent(
            config=self.agents_config["decision_maker_agent"],
            tools=[CogneeIngestion(nodeset_name="final_report")],
            verbose=True,
        )

    @task
    def soft_skills_assessment_applicant1_task(self) -> Task:
        self.tasks_config["soft_skills_assessment_applicant1_task"]["description"] = (
            self.tasks_config["soft_skills_assessment_applicant1_task"]["description"].format(
                **self.inputs
            )
        )
        self.tasks_config["soft_skills_assessment_applicant1_task"]["expected_output"] = (
            self.tasks_config["soft_skills_assessment_applicant1_task"]["expected_output"].format(
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
        self.tasks_config["soft_skills_assessment_applicant2_task"]["expected_output"] = (
            self.tasks_config["soft_skills_assessment_applicant2_task"]["expected_output"].format(
                **self.inputs
            )
        )
        return Task(
            config=self.tasks_config["soft_skills_assessment_applicant2_task"],
            async_execution=False,
        )

    @task
    def technical_assessment_applicant1_task(self) -> Task:
        self.tasks_config["technical_assessment_applicant1_task"]["description"] = (
            self.tasks_config["technical_assessment_applicant1_task"]["description"].format(
                **self.inputs
            )
        )
        self.tasks_config["technical_assessment_applicant1_task"]["expected_output"] = (
            self.tasks_config["technical_assessment_applicant1_task"]["expected_output"].format(
                **self.inputs
            )
        )
        return Task(
            config=self.tasks_config["technical_assessment_applicant1_task"], async_execution=False
        )

    @task
    def technical_assessment_applicant2_task(self) -> Task:
        self.tasks_config["technical_assessment_applicant2_task"]["description"] = (
            self.tasks_config["technical_assessment_applicant2_task"]["description"].format(
                **self.inputs
            )
        )
        self.tasks_config["technical_assessment_applicant2_task"]["expected_output"] = (
            self.tasks_config["technical_assessment_applicant2_task"]["expected_output"].format(
                **self.inputs
            )
        )
        return Task(
            config=self.tasks_config["technical_assessment_applicant2_task"], async_execution=False
        )

    @task
    def hiring_decision_task(self) -> Task:
        self.tasks_config["hiring_decision_task"]["description"] = self.tasks_config[
            "hiring_decision_task"
        ]["description"].format(**self.inputs)
        self.tasks_config["hiring_decision_task"]["expected_output"] = self.tasks_config[
            "hiring_decision_task"
        ]["expected_output"].format(**self.inputs)
        return Task(config=self.tasks_config["hiring_decision_task"], async_execution=False)

    @task
    def ingest_hiring_decision_task(self) -> Task:
        self.tasks_config["ingest_hiring_decision_task"]["description"] = self.tasks_config[
            "ingest_hiring_decision_task"
        ]["description"].format(**self.inputs)
        self.tasks_config["ingest_hiring_decision_task"]["expected_output"] = self.tasks_config[
            "ingest_hiring_decision_task"
        ]["expected_output"].format(**self.inputs)
        return Task(
            config=self.tasks_config["ingest_hiring_decision_task"],
            async_execution=False,
        )

    def refine_agent_configs(self, agent_name: str = None):
        system_prompt = (
            "You are an expert in improving agent definitions for autonomous AI systems. "
            "Given an agent's role, goal, and backstory, refine them to be:\n"
            "- Concise and well-written\n"
            "- Aligned with the agent’s function\n"
            "- Clear and professional\n"
            "- Consistent with multi-agent teamwork\n\n"
            "Return the updated definition as a JSON object with keys: role, goal, backstory."
        )

        agent_keys = [agent_name] if agent_name else self.agents_config.keys()

        for name in agent_keys:
            agent_def = self.agents_config[name]

            user_prompt = f"""Here is the current agent definition:
                                role: {agent_def["role"]}
                                goal: {agent_def["goal"]}
                                backstory: {agent_def["backstory"]}

                                Please improve it."""
            llm_client = get_llm_client()
            improved = llm_client.create_structured_output(
                text_input=user_prompt, system_prompt=system_prompt, response_model=AgentConfig
            )

            self.agents_config[name] = improved.dict()

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
