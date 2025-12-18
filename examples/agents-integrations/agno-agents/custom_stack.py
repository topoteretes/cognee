import os, pathlib, cognee
from cognee_community_vector_adapter_qdrant import register

from agno.agent import Agent
from agno.models.google import Gemini

from tools import CogneeTools
from constants import INSTRUCTIONS, MY_PREFERENCE

from dotenv import load_dotenv
load_dotenv()

def get_db_config():
   system_path = pathlib.Path(__file__).parent
   cognee.config.system_root_directory(os.path.join(system_path, ".cognee_system"))
   cognee.config.data_root_directory(os.path.join(system_path, ".data_storage"))
   cognee.config.set_relational_db_config({"db_provider": "sqlite"})
   cognee.config.set_vector_db_config({
       "vector_db_provider": "qdrant",
       "vector_db_url": os.getenv("QDRANT_URL"),
       "vector_db_key": os.getenv("QDRANT_API_KEY")
   })
   cognee.config.set_graph_db_config({"graph_database_provider": "kuzu"})

def main():
    get_db_config()
    cognee_tools = CogneeTools()
    llm = Gemini(id="gemini-2.5-flash")

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
