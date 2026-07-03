"""Abstract interface for the litellm_native structured output framework.

This module defines the base class that any adapter within the ``litellm_native``
framework must implement.  It intentionally mirrors the method signatures used by
``litellm_instructor/llm/llm_interface.py`` so that ``LLMGateway`` can invoke any
framework adapter through the same contract.

The key difference from the instructor interface is that this one *only* requires
``acreate_structured_output``.  Transcription and image transcription remain with
the instructor adapters because they don't benefit from response_format routing.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel | str")


class LLMNativeInterface(ABC):
    """Contract that every litellm_native adapter must satisfy.

    The only required capability is ``acreate_structured_output``, which takes
    free-form text, a system prompt, and a Pydantic response model and returns
    a validated instance of that model (or a plain ``str`` when the caller asks
    for unstructured text).

    Instance variables that implementations are expected to expose:
    - max_completion_tokens (int)
    """

    max_completion_tokens: int

    @abstractmethod
    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        """Produce validated structured output from an LLM call.

        Args:
            text_input: The user-supplied content to send to the model.
            system_prompt: System-level instructions that guide the model.
            response_model: A Pydantic ``BaseModel`` subclass describing the
                expected JSON shape, or the literal type ``str`` for plain text.
            **kwargs: Extra keyword arguments forwarded to the underlying LLM
                completion call (e.g. ``temperature``, ``max_tokens``).

        Returns:
            A validated instance of *response_model*, or a plain ``str`` when
            ``response_model is str``.

        Raises:
            litellm.exceptions.RateLimitError: Propagated immediately on quota
                exhaustion — never retried.
            litellm.exceptions.AuthenticationError: Propagated immediately —
                never retried.
            pydantic.ValidationError: After all retry attempts are exhausted.
        """
        raise NotImplementedError
