"""Semantic classification of pipeline failures.

Maps a raw exception to a small, stable taxonomy shared by user-facing errors
(``CognifyFailedError``), the recall warm-up marker, and onboarding analytics:
``auth`` / ``provider_quota`` / ``schema`` / ``config`` / ``loader`` /
``db_init`` / ``unknown``. Matching walks the exception's cause/context chain,
checks class names first and message substrings second; the first rule wins.

The categories are intentionally coarse: they answer "what should the user do
next", not "which line failed" — the scrubbed error message carries the detail.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineErrorInfo:
    category: str
    remedy: str


_REMEDIES = {
    "auth": (
        "The LLM provider rejected the API key. Set a valid LLM_API_KEY "
        "(and EMBEDDING_API_KEY if you use a separate embedding provider)."
    ),
    "provider_quota": (
        "The LLM provider refused the request for quota or rate reasons. Check "
        "billing/quota, or enable client-side rate limiting "
        "(LLM_RATE_LIMIT_ENABLED=true)."
    ),
    "schema": (
        "The provider rejected cognee's structured-output schema. Upgrade cognee "
        "(1.5.2+ demotes to a non-strict schema automatically) or set "
        "STRUCTURED_OUTPUT_FRAMEWORK=instructor as a stopgap."
    ),
    "config": (
        "LLM/embedding configuration looks incomplete. Configure BOTH the LLM_* "
        "and EMBEDDING_* providers — configuring only one silently defaults the "
        "other to OpenAI."
    ),
    "loader": (
        "The input could not be loaded or parsed. Check the file format, or pass "
        "preferred_loaders explicitly."
    ),
    "db_init": (
        "Local databases are not initialized or are corrupted. Retry once; if it "
        "persists, remove the .cognee_system directory and re-ingest."
    ),
    "unknown": (
        "Inspect the error message and the pipeline_runs record; run "
        "'cognee-cli doctor' for a configuration preflight."
    ),
}

# (category, exception-class-name substrings, message substrings), first match
# wins. Class names are matched against the whole cause/context chain so a
# wrapped provider error still classifies by its root. Order matters: auth and
# quota outrank the generic config bucket because their messages often also
# mention keys or providers.
_RULES = [
    (
        "auth",
        ("AuthenticationError", "LLMAPIKeyNotSetError", "InvalidApiKeyError"),
        (
            "invalid api key",
            "incorrect api key",
            "api key not set",
            "no api key",
            "authentication",
            "unauthorized",
            "error code: 401",
        ),
    ),
    (
        "provider_quota",
        ("LLMPaymentRequiredError", "LLMQuotaExceededError", "RateLimitError"),
        (
            "insufficient_quota",
            "quota_exceeded",
            "rate limit",
            "payment required",
            "error code: 429",
        ),
    ),
    (
        "schema",
        ("SchemaRejectedError",),
        (
            "response_format",
            "json_schema",
            "invalid schema",
            "'oneof'",
            "additionalproperties",
        ),
    ),
    (
        "db_init",
        ("DatabaseNotCreatedError",),
        ("database not created", "no such table", "unable to open database"),
    ),
    (
        "config",
        ("EmbeddingException", "InvalidValueError", "CogneeConfigurationError"),
        (
            "embedding",
            "llm_api_key",
            "environment variable",
            "unsupported vector database provider",
            "unsupported graph database provider",
        ),
    ),
    (
        "loader",
        ("LoaderError", "IngestionError", "UnstructuredError"),
        ("loader", "unsupported file", "could not parse", "mime type"),
    ),
]


def _exception_chain(error: BaseException) -> list[BaseException]:
    """The error plus its __cause__/__context__ chain, cycle-safe."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_pipeline_error(error: BaseException | str | None) -> PipelineErrorInfo:
    """Classify ``error`` into the pipeline failure taxonomy.

    Accepts an exception (preferred — class names in its cause chain are
    matched too) or a plain message string. Never raises.
    """
    try:
        if isinstance(error, BaseException):
            class_names = " ".join(type(e).__name__ for e in _exception_chain(error))
            text = " ".join(str(e) for e in _exception_chain(error))
        else:
            class_names = ""
            text = str(error or "")

        class_names_lower = class_names.lower()
        text_lower = text.lower()

        for category, name_parts, message_parts in _RULES:
            if any(part.lower() in class_names_lower for part in name_parts):
                return PipelineErrorInfo(category, _REMEDIES[category])
            if any(part in text_lower for part in message_parts):
                return PipelineErrorInfo(category, _REMEDIES[category])
    except Exception:
        pass
    return PipelineErrorInfo("unknown", _REMEDIES["unknown"])
