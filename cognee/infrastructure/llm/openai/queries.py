""" This module contains the functions that are used to query the language model. """
import os
import instructor
from openai import OpenAI
import logging
from cognitive_architecture.shared.data_models import  KnowledgeGraph,  MemorySummary
from cognitive_architecture.config import Config



config = Config()
config.load()

OPENAI_API_KEY = config.openai_key

aclient = instructor.patch(OpenAI())


# Function to read query prompts from files
def read_query_prompt(filename):
    """Read a query prompt from a file."""
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        logging.info(f"Error: File not found. Attempted to read: %s {filename}")
        logging.info(f"Current working directory: %s {os.getcwd()}")
        return None
    except Exception as e:
        logging.info(f"An error occurred: %s {e}")
        return None


def generate_graph(input) -> KnowledgeGraph:
    """Generate a knowledge graph from a user query."""
    model = "gpt-4-1106-preview"
    user_prompt = f"Use the given format to extract information from the following input: {input}."
    system_prompt = read_query_prompt(
        "cognitive_architecture/llm/prompts/generate_graph_prompt.txt"
    )

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
    """Generate a summary from a user query."""
    out = aclient.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": f"""Use the given format summarize 
                and reduce the following input: {input}. """,
            },
            {
                "role": "system",
                "content": """You are a top-tier algorithm
                designed for summarizing existing knowledge 
                graphs in structured formats based on a knowledge graph.
                ## 1. Strict Compliance
                Adhere to the rules strictly. 
                Non-compliance will result in termination.
                ## 2. Don't forget your main goal 
                is to reduce the number of nodes in the knowledge graph 
                while preserving the information contained in it.""",
            },
        ],
        response_model=MemorySummary,
    )
    return out


def user_query_to_edges_and_nodes(input: str) -> KnowledgeGraph:
    """Generate a knowledge graph from a user query."""
    system_prompt = read_query_prompt(
        "cognitive_architecture/llm/prompts/generate_graph_prompt.txt"
    )
    return aclient.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": f"""Use the given format to
                 extract information from the following input: {input}. """,
            },
            {"role": "system", "content": system_prompt},
        ],
        response_model=KnowledgeGraph,
    )
