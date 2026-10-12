"""Make sure litellm knows the models the LoCoMo run uses, and that they exist.

Two separate problems:

1. **Capability table.** cognee's ``litellm_native`` structured-output path asks
   ``litellm.supports_response_schema(model)``. For a model id missing from litellm's
   bundled table that returns ``False`` and every extraction call silently falls back to
   prompted JSON. ``ensure_model_registered`` copies a sibling model's capability row so a
   newer OpenAI id (e.g. a ``gpt-5.x-mini`` released after the pinned litellm) still takes the
   schema-native path.
2. **Existence.** Registering a row does not make the model real. ``probe_model`` issues one
   tiny completion so a typo or an unavailable model fails in seconds, not after an hour of
   ingestion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Capability rows are copied from these when the requested id is unknown.
_TEMPLATE_BY_PREFIX = (
    ("gpt-5", "gpt-5-mini"),
    ("o", "gpt-5-mini"),
)


def _bare(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "openai"


def ensure_model_registered(model: str) -> dict[str, Any] | None:
    """Register ``model`` (both ``provider/x`` and bare ``x`` keys) if litellm lacks it.

    Returns the registered row, or ``None`` when nothing had to be done.
    """
    import litellm

    if model in litellm.model_cost and _bare(model) in litellm.model_cost:
        return None
    if _provider(model) != "openai":
        return None

    bare = _bare(model)
    template_key = next(
        (template for prefix, template in _TEMPLATE_BY_PREFIX if bare.startswith(prefix)), None
    )
    if template_key is None or template_key not in litellm.model_cost:
        return None

    row = dict(litellm.model_cost[template_key])
    row["litellm_provider"] = "openai"
    row.setdefault("mode", "chat")
    registered = {model: row, bare: row}
    litellm.register_model(registered)
    return row


@dataclass
class ProbeResult:
    model: str
    ok: bool
    detail: str
    resolved_model: str | None = None


async def probe_model(model: str, *, api_key: str | None = None) -> ProbeResult:
    """One minimal completion; a NotFound / auth error comes back as ``ok=False``."""
    import litellm

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            api_key=api_key or os.getenv("LLM_API_KEY"),
            max_completion_tokens=256,
        )
        content = (response.choices[0].message.content or "").strip()
        return ProbeResult(
            model=model,
            ok=True,
            detail=f"replied {content!r}",
            resolved_model=getattr(response, "model", None),
        )
    except Exception as error:
        return ProbeResult(model=model, ok=False, detail=f"{type(error).__name__}: {error}")


def describe_capabilities(model: str) -> dict[str, Any]:
    import litellm

    try:
        schema = bool(litellm.supports_response_schema(model=model))
    except Exception:
        schema = False
    return {
        "model": model,
        "known_to_litellm": model in litellm.model_cost or _bare(model) in litellm.model_cost,
        "supports_response_schema": schema,
    }
