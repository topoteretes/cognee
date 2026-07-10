"""Central table of default Instructor modes per LLM provider.

Single source of truth that replaces the per-adapter ``default_instructor_mode``
class attributes previously scattered across the individual adapter modules.
"""

import instructor

# Provider -> default instructor mode. Values match what each adapter used
# historically. ``llama_cpp`` uses the enum member directly, as it did before.
# (Azure is intentionally absent: AzureOpenAIAdapter subclasses OpenAIAdapter and
# therefore inherits the "openai" default.)
INSTRUCTOR_MODE_TABLE: dict[str, object] = {
    "openai": "json_schema_mode",
    "anthropic": "anthropic_tools",
    "gemini": "json_mode",
    "bedrock": "json_schema_mode",
    "ollama": "json_mode",
    "mistral": "mistral_tools",
    "generic": "json_mode",
    "llama_cpp": instructor.Mode.JSON,
}

# Fallback used when a provider has no explicit entry above.
DEFAULT_INSTRUCTOR_MODE = "json_mode"


def get_instructor_mode(provider: str):
    """Return the default instructor mode for ``provider``.

    See ``INSTRUCTOR_MODE_TABLE``; unknown providers fall back to
    ``DEFAULT_INSTRUCTOR_MODE``.
    """
    return INSTRUCTOR_MODE_TABLE.get(provider, DEFAULT_INSTRUCTOR_MODE)
