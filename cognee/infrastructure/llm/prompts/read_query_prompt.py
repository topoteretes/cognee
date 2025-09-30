from os import path
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.root_dir import get_absolute_path


def read_query_prompt(prompt_file_name: str, base_directory: str = None):
    """
    Read a query prompt from a file.

    Retrieve the contents of a specified prompt file, optionally using a provided base
    directory for the file path. If the base directory is not specified, a default path is
    used. Log errors if the file is not found or if another error occurs during file
    reading.

    Parameters:
    -----------

        - prompt_file_name (str): The name of the prompt file to be read.
        - base_directory (str): The base directory from which to read the prompt file. If
          None, a default path is used. (default None)

    Returns:
    --------

        Returns the contents of the prompt file as a string, or None if the file cannot be
        read due to an error.
    """
    logger = get_logger(level=ERROR)

    try:
        if base_directory is None:
            base_directory = get_absolute_path("./infrastructure/llm/prompts")

        file_path = path.join(base_directory, prompt_file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: Prompt file not found. Attempted to read: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None
