from enum import Enum

from cognee.exceptions import CogneeValidationError


class ContextFormat(str, Enum):
    """Shape of an ``only_context`` search result.

    ``CONTEXT`` is the historical shape: the bare retrieval context and nothing else.
    ``PROMPT`` is the full envelope a completion would have received — question, context,
    session layer, and the rendered user and system prompts.

    A closed knob, so it lives next to ``SearchType`` as a ``(str, Enum)`` — the same
    convention as ``ContextProfile`` — and validates in one place for every entry point.
    """

    CONTEXT = "context"
    PROMPT = "prompt"

    @classmethod
    def parse(cls, value: "ContextFormat | str | None") -> "ContextFormat":
        """Coerce a caller-supplied value, raising the one shared validation error."""
        if value is None:
            return cls.CONTEXT
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            raise CogneeValidationError(
                message=(
                    f"Invalid context_format {value!r}. "
                    f"Valid values: {[member.value for member in cls]}."
                ),
                name="InvalidContextFormatError",
            ) from None
