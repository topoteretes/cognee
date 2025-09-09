from cognee.infrastructure.llm.exceptions import MissingSystemPromptPathError
from cognee.infrastructure.llm.prompts import read_query_prompt


def show_prompt(text_input: str, system_prompt: str) -> str:
    """
    Format and display the prompt for a user query.

    This method formats the prompt using the provided user input and system prompt,
    returning a string representation. Raises MissingSystemPromptPathError if the system prompt is not
    provided.

    Parameters:
    -----------

        - text_input (str): The input text provided by the user.
        - system_prompt (str): The system's prompt to guide the model's response.

    Returns:
    --------

        - str: A formatted string representing the user input and system prompt.
    """
    if not text_input:
        text_input = "No user input provided."
    if not system_prompt:
        raise MissingSystemPromptPathError()
    system_prompt = read_query_prompt(system_prompt)

    formatted_prompt = (
        f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
        if system_prompt
        else None
    )
    return formatted_prompt
