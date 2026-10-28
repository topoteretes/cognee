from os import path

from cognee.infrastructure.llm.exceptions import PromptPathOutsideBaseDirectoryError
from cognee.root_dir import get_absolute_path
from cognee.shared.logging_utils import ERROR, get_logger


def _resolve_prompt_path(base_directory: str, prompt_file_name: str) -> str:
    """Join ``prompt_file_name`` onto ``base_directory`` and refuse anything that leaves it.

    Prompt names can come from callers (``system_prompt_path`` on search/recall and
    the ``/responses`` tools), so an absolute name or a ``..`` segment must not reach
    an arbitrary file. realpath follows symlinks, so a link inside the directory
    cannot be used to escape it either.
    """
    base_path = path.realpath(base_directory)
    file_path = path.realpath(path.join(base_path, prompt_file_name))

    if not file_path.startswith(base_path.rstrip(path.sep) + path.sep):
        raise PromptPathOutsideBaseDirectoryError()

    return file_path


def read_query_prompt(prompt_file_name: str, base_directory: str | None = None) -> str | None:
    """
    Read a query prompt from a file.

    Retrieve the contents of a specified prompt file, optionally using a provided base
    directory for the file path. If the base directory is not specified, a default path is
    used. Log errors if the file is not found or if another error occurs during file
    reading.

    Parameters:
    -----------

        - prompt_file_name (str): The name of the prompt file to be read, relative to
          the base directory.
        - base_directory (str): The base directory from which to read the prompt file. If
          None, a default path is used. (default None)

    Returns:
    --------

        Returns the contents of the prompt file as a string, or None if the file cannot be
        read due to an error.

    Raises:
    -------

        - PromptPathOutsideBaseDirectoryError: The name resolves outside the base
          directory. Raised, not logged-and-swallowed, so the caller sees the refusal.
    """
    logger = get_logger(level=ERROR)

    if base_directory is None:
        base_directory = get_absolute_path("./infrastructure/llm/prompts")

    file_path = _resolve_prompt_path(base_directory, prompt_file_name)

    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: Prompt file not found. Attempted to read: {file_path}")
        return None
    except Exception:
        logger.exception("An error occurred")
        return None
