from os import path
import logging
from cognee.root_dir import get_absolute_path

def read_query_prompt(prompt_file_name: str):
    """Read a query prompt from a file."""
    try:
        file_path = path.join(get_absolute_path("./infrastructure/llm/prompts"), prompt_file_name)

        with open(file_path, "r", encoding = "utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Error: Prompt file not found. Attempted to read: %s {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: %s {e}")
        return None
