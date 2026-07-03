"""Sub-second pre-flight: fail fast with a fix, before any real work runs.

Runs before add/cognify/search/remember/recall. No network calls — those
belong to ``cognee-cli doctor``. A failure raises PreflightError carrying the
calm-error fields; the central handler renders it and exits 1 within ~0.2s,
instead of letting the same problem surface minutes later as a retry loop.
"""

import os
import time
from typing import List, Optional, Tuple

from cognee.cli.diagnostics import CalmError, DOCS_LLM_PROVIDERS

_PLACEHOLDER_KEYS = {
    "",
    "sk-...",
    "your_api_key",
    "your-api-key",
    "your_openai_api_key",
    "changeme",
    "none",
    "null",
}

# Providers that cannot run without an API key (ollama and llama_cpp are local).
_KEY_REQUIRED = {"openai", "anthropic", "gemini", "mistral", "azure", "custom", "groq"}

# Providers with no embedding API of their own — without explicit EMBEDDING_*
# configuration the run is guaranteed to fail after the LLM stage.
_NO_EMBEDDING_API = {"anthropic", "llama_cpp", "azure", "bedrock"}


class PreflightError(Exception):
    def __init__(self, calm: CalmError) -> None:
        super().__init__(calm.title)
        self.calm = calm


def _embedding_env_configured() -> bool:
    # Must match the engine's _embeddings_explicitly_configured: a user who
    # sets only EMBEDDING_API_KEY is using the documented OpenAI defaults and
    # counts as configured.
    return any(
        os.environ.get(var)
        for var in (
            "EMBEDDING_PROVIDER",
            "EMBEDDING_MODEL",
            "EMBEDDING_API_KEY",
            "EMBEDDING_ENDPOINT",
        )
    )


def _check_llm(strict_key: bool) -> Optional[CalmError]:
    try:
        from cognee.infrastructure.llm.config import get_llm_config

        config = get_llm_config()
    except Exception as error:  # pydantic validation of a broken .env
        return CalmError(
            title="the LLM configuration could not be loaded",
            why=str(error).splitlines()[0][:200],
            fixes=[("Check", "cognee-cli doctor"), ("Docs", DOCS_LLM_PROVIDERS)],
        )

    provider = (config.llm_provider or "").lower()
    model = config.llm_model or ""

    if strict_key and provider in _KEY_REQUIRED:
        key = (config.llm_api_key or "").strip()
        if key.lower() in _PLACEHOLDER_KEYS:
            looks_placeholder = bool(key)
            return CalmError(
                title=(
                    "LLM_API_KEY looks like a placeholder"
                    if looks_placeholder
                    else "LLM_API_KEY is not set"
                ),
                why=(
                    f"cognee uses an LLM ({model or provider}) to extract entities\n"
                    "and relations — it needs an API key before it can start."
                ),
                fixes=[
                    ("Fix", "export LLM_API_KEY=sk-..."),
                    ("Other", DOCS_LLM_PROVIDERS),
                    ("Check", "cognee-cli doctor"),
                ],
            )
    return None


def _check_embeddings() -> Optional[CalmError]:
    if _embedding_env_configured():
        return None
    try:
        from cognee.infrastructure.llm.config import get_llm_config

        provider = (get_llm_config().llm_provider or "").lower()
    except Exception:
        return None
    if provider in _NO_EMBEDDING_API:
        return CalmError(
            title=f"{provider} has no embedding API — embeddings need their own provider",
            why=(
                "Building and searching memory requires embeddings, and your LLM\n"
                "provider doesn't offer them, so cognee can't derive a default."
            ),
            fixes=[
                ("Fix", "export EMBEDDING_PROVIDER=openai  (plus EMBEDDING_API_KEY=...)"),
                ("Local", "export EMBEDDING_PROVIDER=fastembed  (no key needed)"),
                ("Docs", DOCS_LLM_PROVIDERS),
            ],
        )
    return None


def _check_storage() -> Optional[CalmError]:
    try:
        from cognee.base_config import get_base_config

        base = get_base_config()
        for root in (base.data_root_directory, base.system_root_directory):
            if not root or str(root).startswith("s3://"):
                continue
            os.makedirs(root, exist_ok=True)
            probe = os.path.join(root, ".cognee_write_probe")
            with open(probe, "w") as handle:
                handle.write("ok")
            os.remove(probe)
    except OSError as error:
        return CalmError(
            title="cognee's storage directory is not writable",
            why=str(error),
            fixes=[
                ("Fix", "check permissions on DATA_ROOT_DIRECTORY / SYSTEM_ROOT_DIRECTORY"),
                ("Check", "cognee-cli doctor"),
            ],
        )
    except Exception:
        return None  # config import problems surface via _check_llm or the command
    return None


def run_preflight(need_llm: bool = True, need_embeddings: bool = True) -> float:
    """Raise PreflightError on the first blocking problem; return elapsed seconds."""
    started = time.monotonic()
    checks = [lambda: _check_storage()]
    if need_llm or need_embeddings:
        checks.append(lambda: _check_llm(strict_key=need_llm))
    if need_embeddings:
        checks.append(lambda: _check_embeddings())

    for check in checks:
        problem = check()
        if problem is not None:
            elapsed = time.monotonic() - started
            if elapsed >= 0.5:
                problem.footer = f"(failed in {elapsed:.1f}s — before any work ran)"
            else:
                problem.footer = "(stopped before any work ran)"
            raise PreflightError(problem)
    return time.monotonic() - started


def needs_for_search_type(search_type_name: str) -> Tuple[bool, bool]:
    """(need_llm, need_embeddings) for a given SearchType name."""
    name = (search_type_name or "").upper()
    if name == "CYPHER":
        return False, False
    if name in ("CHUNKS", "CHUNKS_LEXICAL", "SUMMARIES"):
        return False, True
    return True, True
