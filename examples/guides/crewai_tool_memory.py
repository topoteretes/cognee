"""CrewAI agent with Cognee memory via tool calling.

This example shows how to give a CrewAI agent persistent memory
using cognee.tools. The agent gets `remember` and `recall` as
CrewAI BaseTool objects.

Prerequisites:
    pip install cognee crewai
    export LLM_API_KEY="sk-..."
"""

from crewai import Agent, Task, Crew

from cognee.tools import for_crewai


def main():
    tools = for_crewai()

    agent = Agent(
        role="Personal Assistant",
        goal="Help the user and remember important information",
        backstory=(
            "You are a helpful assistant with persistent memory. "
            "Use the `remember` tool to save important facts. "
            "Use the `recall` tool to retrieve previously stored information."
        ),
        tools=tools,
    )

    # Store some facts
    remember_task = Task(
        description="Remember that the user prefers dark mode and uses vim keybindings.",
        expected_output="Confirmation that the preferences were saved.",
        agent=agent,
    )

    # Retrieve them later
    recall_task = Task(
        description="What are the user's editor preferences?",
        expected_output="The user's editor preferences based on stored memory.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[remember_task, recall_task],
    )

    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
