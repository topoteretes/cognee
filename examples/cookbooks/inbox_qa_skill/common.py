"""Settings every step shares, so each step also runs on its own."""

import os
from pathlib import Path

HERE = Path(__file__).parent
DATASET = "inbox_qa_skill"
GMAIL_CREDENTIALS = HERE / "credentials.json"
GMAIL_TOKEN = HERE / "token.json"
SAMPLE_FILE = HERE / "data" / "notes.txt"


def missing_setup(need_gmail: bool = True) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    import cognee  # loads .env, so an LLM_API_KEY set there is seen

    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if need_gmail and not GMAIL_CREDENTIALS.exists():
        missing.append(f"Gmail OAuth client not found at {GMAIL_CREDENTIALS}.")
    return missing
