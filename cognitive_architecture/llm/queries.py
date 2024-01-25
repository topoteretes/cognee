import os

from dotenv import load_dotenv

from ..shared.data_models import Node, Edge, KnowledgeGraph, GraphQLQuery, MemorySummary
from ..config import Config
import instructor
from openai import OpenAI
config = Config()
config.load()

print(config.model)
print(config.openai_key)

OPENAI_API_KEY = config.openai_key

aclient = instructor.patch(OpenAI())

load_dotenv()
import logging


# Function to read query prompts from files
def read_query_prompt(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        logging.info(f"Error: File not found. Attempted to read: {filename}")
        logging.info(f"Current working directory: {os.getcwd()}")
        return None
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return None


def generate_graph(input) -> KnowledgeGraph:
    model = "gpt-4-1106-preview"
    user_prompt = f"Use the given format to extract information from the following input: {input}."
    system_prompt = read_query_prompt('cognitive_architecture/llm/prompts/generate_graph_prompt.txt')

    out = aclient.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "system",
                "content": system_prompt,
            },
        ],
        response_model=KnowledgeGraph,
    )
    return out



async def generate_summary(input) -> MemorySummary:
    out =  aclient.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": f"""Use the given format summarize and reduce the following input: {input}. """,

            },
            {   "role":"system", "content": """You are a top-tier algorithm
                designed for summarizing existing knowledge graphs in structured formats based on a knowledge graph.
                ## 1. Strict Compliance
                Adhere to the rules strictly. Non-compliance will result in termination.
                ## 2. Don't forget your main goal is to reduce the number of nodes in the knowledge graph while preserving the information contained in it."""}
        ],
        response_model=MemorySummary,
    )
    return out


def user_query_to_edges_and_nodes( input: str) ->KnowledgeGraph:
    system_prompt = read_query_prompt('cognitive_architecture/llm/prompts/generate_graph_prompt.txt')
    return aclient.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": f"""Use the given format to extract information from the following input: {input}. """,

            },
            {"role": "system", "content":system_prompt}
        ],
        response_model=KnowledgeGraph,
    )