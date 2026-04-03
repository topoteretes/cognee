# ruff: noqa: E402
from cognee.version import get_cognee_version

# NOTE: __version__ extraction must be at the top of the __init__.py otherwise
#       there will be circular import issues
__version__ = get_cognee_version()

# Load environment variable settings has to be before setting up logging for LOG_LEVEL value
import dotenv

dotenv.load_dotenv(override=True)

# NOTE: Log level can be set with the LOG_LEVEL env variable
from cognee.shared.logging_utils import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Lazy imports — heavy modules are loaded on first access, not at import time.
# This keeps `import cognee` fast for scripts that only use a subset of the API.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # V1 API
    "add": ".api.v1.add",
    "delete": ".api.v1.delete",
    "cognify": ".api.v1.cognify",
    "memify": ".modules.memify",
    "run_custom_pipeline": ".modules.run_custom_pipeline",
    "update": ".api.v1.update",
    "config": ".api.v1.config.config",
    "datasets": ".api.v1.datasets.datasets",
    "prune": ".api.v1.prune",
    "SearchType": ".api.v1.search",
    "search": ".api.v1.search",
    "visualize_graph": ".api.v1.visualize",
    "start_visualization_server": ".api.v1.visualize",
    "cognee_network_visualization": "cognee.modules.visualization.cognee_network_visualization",
    "start_ui": ".api.v1.ui",
    "session": ".api.v1.session",
    # V2 memory-oriented API
    "remember": ".api.v2",
    "RememberResult": ".api.v2",
    "recall": ".api.v2",
    "improve": ".api.v2",
    "forget": ".api.v2",
    # Pipelines
    "pipelines": ".modules",
    "Drop": ".pipelines",
    # Migrations
    "run_migrations": "cognee.run_migrations",
    # Tracing / Observability
    "enable_tracing": "cognee.modules.observability.trace_context",
    "disable_tracing": "cognee.modules.observability.trace_context",
    "get_last_trace": "cognee.modules.observability.trace_context",
    "get_all_traces": "cognee.modules.observability.trace_context",
    "clear_traces": "cognee.modules.observability.trace_context",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        if module_path.startswith("."):
            import importlib

            mod = importlib.import_module(module_path, package="cognee")
        else:
            import importlib

            mod = importlib.import_module(module_path)
        attr = getattr(mod, name)
        # Cache on the module so __getattr__ is only called once per name
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'cognee' has no attribute {name!r}")


# Explicit list for `from cognee import *` and IDE autocompletion
__all__ = list(_LAZY_IMPORTS.keys()) + ["__version__", "logger"]
