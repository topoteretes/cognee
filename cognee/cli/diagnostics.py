"""Translate the failures a first-time user actually hits into calm next steps.

Every entry answers three questions in order: what happened, why, and the
exact command that fixes it — ending with one deep docs link, never a bare
"see the docs". Anything unrecognized falls back to an honest generic block
that names the log file and the --debug escape hatch.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DOCS_LLM_PROVIDERS = "https://docs.cognee.ai/setup-configuration/llm-providers"
DOCS_INSTALL = "https://docs.cognee.ai/getting-started/installation"

# Provider extras a missing import usually maps to.
_EXTRA_FOR_MODULE = {
    "anthropic": "anthropic",
    "google.generativeai": "gemini",
    "google": "gemini",
    "mistralai": "mistral",
    "groq": "groq",
    "boto3": "aws",
    "chromadb": "chromadb",
    "neo4j": "neo4j",
    "llama_cpp": "llama-cpp",
    "transformers": "huggingface",
}


@dataclass
class CalmError:
    title: str
    why: Optional[str] = None
    fixes: List[Tuple[str, str]] = field(default_factory=list)
    footer: Optional[str] = None
    exit_code: int = 1


def _exception_chain(exc: BaseException) -> list:
    """The full cause/context chain, outermost first — matching must try every
    link, because wrappers like litellm nest the interesting exception in the
    middle of the chain, not at either end."""
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _llm_context() -> Tuple[str, str]:
    """(provider, model) for error copy; never raises."""
    try:
        from cognee.infrastructure.llm.config import get_llm_config

        config = get_llm_config()
        return config.llm_provider, config.llm_model
    except Exception:
        return "openai", "openai/gpt-5-mini"


def describe_exception(exc: BaseException) -> Optional[CalmError]:
    """Return a calm rendering for a known first-run failure, else None."""
    for cause in _exception_chain(exc):
        calm = _describe_single(cause)
        if calm is not None:
            return calm
    return None


def _describe_single(cause: BaseException) -> Optional[CalmError]:
    cause_type = type(cause).__name__
    message = str(cause)

    # 1. No API key configured at all.
    if cause_type == "LLMAPIKeyNotSetError" or "LLM API key is not set" in message:
        provider, model = _llm_context()
        return CalmError(
            title="LLM_API_KEY is not set",
            why=(
                f"cognee uses an LLM ({model}) to extract entities\n"
                "and relations — it needs an API key before it can start."
            ),
            fixes=[
                ("Fix", "export LLM_API_KEY=sk-..."),
                ("Other", DOCS_LLM_PROVIDERS),
                ("Check", "cognee-cli doctor"),
            ],
        )

    # 2. A key is set but the provider rejected it.
    if cause_type == "AuthenticationError" or "Incorrect API key" in message:
        provider, _ = _llm_context()
        key_hint = ""
        match = re.search(r"(sk-[A-Za-z0-9*_-]{4,})", message)
        if match:
            key_hint = f" ({match.group(1)[:8]}…)"
        return CalmError(
            title=f"{provider} rejected your API key{key_hint}",
            why="The key in LLM_API_KEY exists but the provider refused it.",
            fixes=[
                ("Fix", "double-check the key, or issue a new one from your provider"),
                ("Check", "cognee-cli doctor"),
                ("Docs", DOCS_LLM_PROVIDERS),
            ],
        )

    # 3. Provider package not installed (extras).
    if isinstance(cause, ModuleNotFoundError):
        module = (getattr(cause, "name", "") or "").split(".")[0]
        extra = _EXTRA_FOR_MODULE.get(module) or _EXTRA_FOR_MODULE.get(
            getattr(cause, "name", "") or ""
        )
        if extra:
            return CalmError(
                title=f"the '{module}' package is not installed",
                why="Your configuration needs it, but it ships as an optional extra.",
                fixes=[
                    ("Fix", f'pip install "cognee[{extra}]"'),
                    ("Docs", DOCS_INSTALL),
                ],
            )

    # 4. Embedding/LLM provider mismatch (raised by embedding derivation).
    if cause_type == "EmbeddingProviderMismatchError":
        return CalmError(
            title="this LLM provider has no embedding API",
            why=getattr(cause, "message", None) or message,
            fixes=[
                ("Fix", "export EMBEDDING_PROVIDER=openai  (plus EMBEDDING_API_KEY=...)"),
                ("Local", "export EMBEDDING_PROVIDER=fastembed  (no key needed)"),
                ("Docs", DOCS_LLM_PROVIDERS),
            ],
        )

    # 5. Storage directory not usable.
    if "unable to open database file" in message:
        return CalmError(
            title="cognee's storage directory can't be opened",
            why=(
                "The system database directory is missing or not writable\n"
                "(often after setting SYSTEM_ROOT_DIRECTORY to a new path)."
            ),
            fixes=[
                ("Fix", "cognee-cli doctor   (creates and verifies storage directories)"),
                ("Or", "check permissions on DATA_ROOT_DIRECTORY / SYSTEM_ROOT_DIRECTORY"),
            ],
        )

    # 6. Dataset name that doesn't exist.
    if cause_type == "DatasetNotFoundError":
        return CalmError(
            title="that dataset doesn't exist yet",
            why=message,
            fixes=[
                ("List", "cognee-cli datasets list"),
                ("Create", 'cognee-cli add <file> --dataset-name "<name>"'),
            ],
        )

    # 7. Can't reach a local/custom endpoint (ollama & friends).
    if cause_type in ("ConnectError", "ConnectionError", "APIConnectionError") or (
        "Connection refused" in message
    ):
        provider, _ = _llm_context()
        fixes = [("Check", "cognee-cli doctor")]
        if provider == "ollama":
            fixes.insert(0, ("Fix", "start Ollama first:  ollama serve"))
        else:
            fixes.insert(0, ("Fix", "verify LLM_ENDPOINT is reachable"))
        return CalmError(
            title=f"can't reach the {provider} endpoint",
            why=message.splitlines()[0][:200] if message else None,
            fixes=fixes,
        )

    return None
