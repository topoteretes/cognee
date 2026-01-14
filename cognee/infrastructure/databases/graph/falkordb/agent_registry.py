"""Agent registry for FalkorDB multi-agent isolation.

This module provides functions to load and query the agent-to-graph mapping,
allowing multiple agents to share the same graph or have isolated graphs.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("FalkorDBAgentRegistry")

# Default registry location
REGISTRY_FILE = Path(__file__).parent / "falkordb_registry.json"


@lru_cache(maxsize=1)
def load_registry() -> Dict:
    """Load the agent registry from falkordb_registry.json.

    Returns:
        Dict containing 'default' graph name and 'agents' mapping.
    """
    registry_path = os.environ.get("FALKORDB_REGISTRY_PATH", str(REGISTRY_FILE))

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
            logger.debug("Loaded FalkorDB registry from %s", registry_path)
            return registry
    except FileNotFoundError:
        logger.warning(
            "FalkorDB registry not found at %s, using defaults", registry_path
        )
        return {"default": "CogneeGraph", "agents": {}}
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in FalkorDB registry: %s", e)
        return {"default": "CogneeGraph", "agents": {}}


def get_graph_name_for_agent(agent_name: Optional[str]) -> str:
    """Get the graph name for a given agent.

    Args:
        agent_name: Name of the agent (from X-Agent-Name header or similar).

    Returns:
        Graph name to use for this agent. Falls back to 'default' if agent
        not found in registry.
    """
    registry = load_registry()

    if not agent_name:
        return registry.get("default", "CogneeGraph")

    agents = registry.get("agents", {})
    if agent_name in agents:
        graph_name = agents[agent_name]
        logger.debug("Agent '%s' mapped to graph '%s'", agent_name, graph_name)
        return graph_name

    # Agent not in registry, use default
    default_graph = registry.get("default", "CogneeGraph")
    logger.debug(
        "Agent '%s' not in registry, using default graph '%s'",
        agent_name,
        default_graph,
    )
    return default_graph


def clear_registry_cache():
    """Clear the cached registry to force reload on next access."""
    load_registry.cache_clear()
