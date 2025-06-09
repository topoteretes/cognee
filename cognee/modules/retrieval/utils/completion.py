import os
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt

# Define the directory where prompt templates are allowed to reside.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')
# Define the set of allowed prompt filenames. Extend as needed.
ALLOWED_PROMPT_FILENAMES = {
    "summarize_search_results.txt",
    # Add other allowed prompt template files here.
}

def validate_prompt_path(prompt_path: str) -> str:
    """
    Validates the prompt path to prevent path traversal and local file inclusion.
    Only allows files within PROMPTS_DIR and with an allowed filename.
    Returns the cleaned absolute path to the prompt file if valid, raises ValueError otherwise.
    """
    # Only allow filenames (no directory component)
    filename = os.path.basename(prompt_path)

    # Check for allowed filenames
    if filename not in ALLOWED_PROMPT_FILENAMES:
        raise ValueError(f"Invalid prompt filename: {filename}")

    # Construct absolute path to file in prompts directory
    abs_path = os.path.abspath(os.path.join(PROMPTS_DIR, filename))
    # Ensure the path is within the prompts directory
    if not abs_path.startswith(os.path.abspath(PROMPTS_DIR) + os.sep):
        raise ValueError("Attempted path traversal in prompt path.")
    return abs_path

async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}

    # Validate prompt paths
    user_prompt_file = validate_prompt_path(user_prompt_path)
    system_prompt_file = validate_prompt_path(system_prompt_path)

    user_prompt = render_prompt(user_prompt_file, args)
    system_prompt = read_query_prompt(system_prompt_file)

    llm_client = get_llm_client()
    return await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )

async def summarize_text(
    text: str,
    prompt_path: str = "summarize_search_results.txt",
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    # Validate prompt path
    prompt_file = validate_prompt_path(prompt_path)

    system_prompt = read_query_prompt(prompt_file)
    llm_client = get_llm_client()

    return await llm_client.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )