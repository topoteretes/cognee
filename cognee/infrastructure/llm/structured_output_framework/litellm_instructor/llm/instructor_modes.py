"""Single source of truth for each provider's default Instructor mode.

Replaces the ``default_instructor_mode`` class attributes that were previously
duplicated across the individual adapter modules.
"""

import instructor

# Values match each adapter's historical default. Azure is absent on purpose:
# AzureOpenAIAdapter subclasses OpenAIAdapter and inherits the "openai" mode.
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

DEFAULT_INSTRUCTOR_MODE = "json_mode"


def get_instructor_mode(provider: str):
    """Return ``provider``'s default instructor mode, or the fallback if unknown."""
    return INSTRUCTOR_MODE_TABLE.get(provider, DEFAULT_INSTRUCTOR_MODE)
