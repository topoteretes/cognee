"""Single source of truth for each provider's default Instructor mode.

Replaces the ``default_instructor_mode`` class attributes that were previously
duplicated across the individual adapter modules.
"""

import instructor

# Values match each adapter's historical default. Azure is absent on purpose:
# AzureOpenAIAdapter subclasses OpenAIAdapter and inherits the "openai" mode.
INSTRUCTOR_MODE_TABLE: dict[str, str | instructor.Mode] = {
    "openai": "json_schema_mode",
    "anthropic": "anthropic_tools",
    "gemini": "json_mode",
    "bedrock": "json_schema_mode",
    "ollama": "json_mode",
    "mistral": "mistral_tools",
    "generic": "json_mode",
    "llama_cpp": instructor.Mode.JSON,
}


def get_instructor_mode(provider: str) -> str | instructor.Mode:
    """Return ``provider``'s default instructor mode.

    Raises ``KeyError`` for a provider not in ``INSTRUCTOR_MODE_TABLE`` so a
    missing entry fails loudly at import rather than silently picking a default.
    """
    return INSTRUCTOR_MODE_TABLE[provider]
