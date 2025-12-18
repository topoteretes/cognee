from agno.agent import Agent
from agno.models.openai import OpenAIChat

from tools import CogneeTools
from constants import INSTRUCTIONS, MY_PREFERENCE
from dotenv import load_dotenv
load_dotenv()

def main():
    cognee_tools = CogneeTools()
    llm = OpenAIChat(id="gpt-5-mini")

    agent = Agent(
        model=llm,
        tools=[cognee_tools],
        description="You are my executive assistant who plans my itinerary based on my preference",
        instructions=INSTRUCTIONS
    )

    agent.print_response(MY_PREFERENCE, stream=True)
    print("\n")
    agent.print_response("I am visiting Rome, give me restaurants list to stop by", stream=True)
    
if __name__ == "__main__":
    main()