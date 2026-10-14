"""Standalone End-to-End Demo: Autonomous Agent utilizing Cognee FastMCP tools.

Demonstrates:
1. Ingesting unstructured project notes via cognee_add_memory.
2. Building knowledge graph nodes and semantic relations (cognify).
3. Querying graph and vector memory via cognee_search_memory.
4. Terminal formatting with ANSI colors and ASCII graph visualization.
5. Zero-friction mock-safe fallback for instant execution without API keys.

Usage:
    python -m cognee.api.mcp.demo_agent
    or:
    python cognee/api/mcp/demo_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure current package directory is resolvable
pkg_dir = str(Path(__file__).resolve().parent)
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

try:
    from .models import AddMemoryResponse, SearchMemoryResponse, PruneMemoryResponse
    from .config import settings
    from . import server
    from .server import cognee_add_memory, cognee_search_memory, mcp
except (ImportError, ValueError):
    from models import AddMemoryResponse, SearchMemoryResponse, PruneMemoryResponse
    from config import settings
    import server
    from server import cognee_add_memory, cognee_search_memory, mcp

# ANSI Color Codes for terminal styling
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
BLUE = "\033[1;34m"
RED = "\033[1;31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class MockFallbackAdapter:
    """Mock-safe fallback adapter for zero-configuration local demos."""

    def __init__(self):
        self.memories: List[Dict[str, Any]] = []

    async def add_memory(self, content: str, dataset_name: str = "default_dataset", auto_cognify: bool = True):
        self.memories.append({"content": content, "dataset": dataset_name, "cognified": auto_cognify})
        return AddMemoryResponse(
            success=True,
            dataset_name=dataset_name,
            message=f"Successfully ingested memory chunk into dataset '{dataset_name}'.",
            cognified=auto_cognify,
            error=None
        )

    async def search_memory(self, query: str, search_type=None, dataset_name=None):
        mock_graph_results = [
            {
                "entity": "Project Nexus",
                "type": "Autonomous Agent Platform",
                "relationships": [
                    {"rel": "USES_ENGINE", "target": "Cognee v1.6+"},
                    {"rel": "INTEGRATES_VIA", "target": "FastMCP Server (Model Context Protocol)"},
                    {"rel": "VECTOR_BACKEND", "target": "LanceDB (Embedded Vector Store)"},
                    {"rel": "GRAPH_BACKEND", "target": "NetworkX & Kùzu (Topological Entity Graph)"}
                ],
                "confidence_score": 0.96
            },
            {
                "entity": "FastMCP Integration",
                "type": "Protocol Bridge",
                "relationships": [
                    {"rel": "EXPOSES_TOOL", "target": "cognee_add_memory"},
                    {"rel": "EXPOSES_TOOL", "target": "cognee_search_memory"},
                    {"rel": "SUPPORTS_TRANSPORTS", "target": "stdio (Claude, Cursor) & SSE (Network agents)"}
                ],
                "confidence_score": 0.94
            }
        ]

        return SearchMemoryResponse(
            success=True,
            query=query,
            search_type=search_type.value if hasattr(search_type, "value") else str(search_type),
            dataset_name=dataset_name,
            total_results=len(mock_graph_results),
            results=mock_graph_results,
            error=None
        )

    async def prune_memory(self, prune_all: bool = False):
        self.memories.clear()
        return PruneMemoryResponse(success=True, message="Demo memory cache reset.", prune_all=prune_all)


def print_banner():
    banner = f"""{CYAN}
========================================================================
   ____                            _____         _   __  __  ____ ____  
  / ___|___   __ _ _ __   ___  ___|  ___|_ _ ___| |_|  \\/  |/ ___|  _ \\ 
 | |   / _ \\ / _` | '_ \\ / _ \\/ _ \\ |_ / _` / __| __| |\\/| | |   | |_) |
 | |__| (_) | (_| | | | |  __/  __/  _| (_| \\__ \\ |_| |  | | |___|  __/ 
  \\____\\___/ \\__, |_| |_|\\___|\\___|_|  \\__,_|___/\\__|_|  |_|\\____|_|    
             |___/                                                      
========================================================================
    Model Context Protocol (MCP) Memory Engine for AI Agents
========================================================================{RESET}"""
    print(banner)


async def run_demo():
    print_banner()

    # Check if live Cognee or remote API is configured
    has_live_engine = bool(settings.COGNEE_API_URL or os.environ.get("OPENAI_API_KEY"))

    if not has_live_engine:
        print(f"\n{YELLOW}[i] No live OPENAI_API_KEY or COGNEE_API_URL detected.{RESET}")
        print(f"{GREEN}[✓] Activating Mock-Safe Fallback Adapter for instant local demonstration.{RESET}\n")
        server.set_adapter(MockFallbackAdapter())
    else:
        print(f"\n{GREEN}[✓] Live Cognee configuration active.{RESET}\n")

    # Sample context to ingest
    sample_document = (
        "Project Nexus is a next-generation autonomous AI system built on Cognee's GraphRAG memory layer. "
        "It communicates with external client models (Claude Desktop, Cursor IDE) via the Model Context Protocol "
        "(FastMCP). For high-throughput vector similarity, Project Nexus utilizes LanceDB, while NetworkX and "
        "Kùzu power its complex multi-hop relational entity knowledge graph. Author: Fokrul Islam."
    )

    dataset_name = "nexus_knowledge_base"

    # Step 1: Memory Ingestion
    print(f"{BOLD}[Step 1/3] Ingesting Context into Cognee Knowledge Graph...{RESET}")
    print(f"{DIM}Payload: \"{sample_document[:80]}...\"{RESET}")

    add_resp = await cognee_add_memory(
        content=sample_document,
        dataset_name=dataset_name,
        auto_cognify=True
    )

    if add_resp.success:
        print(f"{GREEN}✓ Ingestion Status : SUCCESS{RESET}")
        print(f"  - Dataset Namespace: {CYAN}{add_resp.dataset_name}{RESET}")
        print(f"  - Cognify Pipeline : {MAGENTA}{'EXTRACTED & COGNIFIED' if add_resp.cognified else 'QUEUED'}{RESET}")
        print(f"  - Message          : {add_resp.message}\n")
    else:
        print(f"{RED}✗ Ingestion Error: {add_resp.error}{RESET}\n")
        return

    # Step 2: Memory Retrieval via Hybrid Graph & Vector Search
    user_query = "What database technologies and protocol bridges power Project Nexus?"
    print(f"{BOLD}[Step 2/3] Querying Cognee Memory via FastMCP Search Tool...{RESET}")
    print(f"{DIM}Query: \"{user_query}\"{RESET}")

    search_resp = await cognee_search_memory(
        query=user_query,
        search_type="INSIGHTS",
        dataset_name=dataset_name
    )

    if search_resp.success:
        print(f"{GREEN}✓ Retrieval Status : SUCCESS ({search_resp.total_results} graph context nodes retrieved){RESET}\n")
        print(f"{BOLD}--- Retrieved Knowledge Graph Context ---{RESET}")

        for idx, item in enumerate(search_resp.results, start=1):
            if isinstance(item, dict):
                entity = item.get("entity", f"Node-{idx}")
                etype = item.get("type", "Entity")
                score = item.get("confidence_score", "N/A")
                rels = item.get("relationships", [])

                print(f"\n{CYAN}┌─ [Node #{idx}] {BOLD}{entity}{RESET} {DIM}(Type: {etype}, Score: {score}){RESET}")
                if rels:
                    for rel in rels:
                        pred = rel.get("rel", "CONNECTED_TO")
                        target = rel.get("target", "Unknown")
                        print(f"{CYAN}│{RESET}   ├───({YELLOW}{pred}{RESET})───> [{GREEN}{target}{RESET}]")
                print(f"{CYAN}└────────────────────────────────────────────────────────────{RESET}")
            else:
                print(f"  - {item}")
    else:
        print(f"{RED}✗ Search Error: {search_resp.error}{RESET}\n")

    # Step 3: Protocol Verification
    print(f"\n{BOLD}[Step 3/3] Verifying MCP Protocol Dispatch Compatibility...{RESET}")
    mcp_call = await mcp.call_tool(
        "cognee_search_memory",
        {"query": "protocols", "search_type": "INSIGHTS", "dataset_name": dataset_name}
    )
    if not mcp_call.is_error:
        print(f"{GREEN}✓ FastMCP JSON-RPC Tool Call Verification: PASS{RESET}")
        print(f"  - Server Name : {CYAN}{mcp.name}{RESET}")
        print(f"  - Available Tools : {MAGENTA}cognee_add_memory, cognee_search_memory, cognee_prune_memory{RESET}")

    print(f"\n{GREEN}========================================================================{RESET}")
    print(f"{GREEN}✓ End-to-End Autonomous Agent Demo Completed Successfully!{RESET}")
    print(f"{GREEN}========================================================================{RESET}\n")


def main():
    """CLI Entry point."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
