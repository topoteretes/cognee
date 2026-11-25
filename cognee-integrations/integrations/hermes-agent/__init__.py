"""Directory-plugin entry point for Hermes Agent."""

try:
    from .cognee_integration_hermes import CogneeMemoryProvider, transform_llm_output
except ImportError:  # pragma: no cover - supports direct pytest/path imports.
    import sys
    from pathlib import Path

    plugin_dir = Path(__file__).resolve().parent
    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
    from cognee_integration_hermes import CogneeMemoryProvider, transform_llm_output


def register(ctx) -> None:
    """Register Cognee as the Hermes memory provider, with its memory-credit hook."""
    ctx.register_memory_provider(CogneeMemoryProvider())
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        register_hook("transform_llm_output", transform_llm_output)
