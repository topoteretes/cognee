"""Legacy import path for the Ladybug graph adapter."""

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


KuzuAdapter = LadybugAdapter

__all__ = ["KuzuAdapter"]
