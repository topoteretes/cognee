"""Dynamic registry of optional graph database adapters."""

supported_databases = {}

try:  # Optional community adapter for Ladybug
    from cognee_community_graph_adapter_ladybug.ladybug_adapter import LadybugAdapter

    supported_databases["ladybug"] = LadybugAdapter
except ImportError:
    # Adapter is installed via the cognee-community package; skip if unavailable
    pass
