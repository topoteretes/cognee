"""Run cognee on the LLM key you have, and embed on your machine with fastembed.

Every script that calls cognee calls configure_cognee() first, so each one also runs alone.
The key picks the LLM only: LLM_API_KEY (with what .env configures), else OPENAI_API_KEY, else
ANTHROPIC_API_KEY. Embeddings are always local fastembed unless .env sets EMBEDDING_PROVIDER,
so switching keys between runs never changes the vectors the dataset already holds.
"""

import os

import cognee  # loads .env, so keys set there are seen below

ANTHROPIC_MODEL = "claude-haiku-5-5"  # cheap and fast; cognify makes one call per chunk
NO_KEY = "No LLM key is set. Put LLM_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY in .env."


def has_key() -> bool:
    """Whether cognee has an LLM key to run on. Any one key imports every agent's history."""
    return any(
        os.environ.get(name) for name in ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )


def configure_cognee() -> str:
    """Point cognee's LLM at the first key set, pin the embedder, and say which they are."""
    if os.environ.get("LLM_API_KEY"):
        llm = f"{os.environ.get('LLM_PROVIDER', 'openai')} (LLM_API_KEY)"
    elif os.environ.get("OPENAI_API_KEY"):
        cognee.config.set_llm_api_key(os.environ["OPENAI_API_KEY"])
        llm = "openai (OPENAI_API_KEY)"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        cognee.config.set_llm_config(
            {
                "llm_provider": "anthropic",
                "llm_model": ANTHROPIC_MODEL,
                "llm_api_key": os.environ["ANTHROPIC_API_KEY"],
            }
        )
        llm = f"anthropic {ANTHROPIC_MODEL} (ANTHROPIC_API_KEY)"
    else:
        raise RuntimeError(NO_KEY)

    if os.environ.get("EMBEDDING_PROVIDER"):
        return f"{llm}, {os.environ['EMBEDDING_PROVIDER']} embeddings (EMBEDDING_PROVIDER)"
    # One embedder whatever the key: a dataset's vectors must all have one size, and cognee
    # refuses a dataset embedded with another model. fastembed is a core dependency; the model
    # downloads once, about 70 MB.
    cognee.config.set_embedding_config(
        {
            "embedding_provider": "fastembed",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dimensions": 384,
        }
    )
    return f"{llm}, local fastembed embeddings"
