from os import path
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.root_dir import get_absolute_path


def read_query_prompt(prompt_file_name: str, base_directory: str = None):
    """Read a query prompt from a file."""
    logger = get_logger(level=ERROR)
    try:
        if base_directory is None:
            base_directory = get_absolute_path("./infrastructure/llm/prompts")

        file_path = path.join(base_directory, prompt_file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: Prompt file not found. Attempted to read: %s {file_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: %s {e}")
        return None
