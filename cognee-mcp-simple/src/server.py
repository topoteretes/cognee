"""
Cognee Memory — minimal MCP server with 2 tools.

Tools:
  remember      — Store information into memory (instant session cache + background graph).
  search_memory — Retrieve from memory (hybrid: session + graph).

Design:
  - `remember` dual-writes: session cache for instant recall, then kicks off
    add + cognify in the background so the knowledge graph stays up to date.
  - `search_memory` queries the session layer first (fast, recent), then the
    graph layer (deep, semantic), and merges the results.
  - memify runs automatically in the background after cognify finishes.
"""

import sys
import asyncio
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Optional

import mcp.types as types
from mcp.server import FastMCP

from cognee.shared.logging_utils import get_logger, setup_logging

logger = get_logger()

mcp = FastMCP("Cognee Memory")

# Background task error log for status reporting
_task_errors: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_session_manager():
    """Return a SessionManager wired to the current cache engine."""
    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
    from cognee.infrastructure.session.session_manager import SessionManager

    cache_engine = get_cache_engine()
    return SessionManager(cache_engine=cache_engine)


async def _get_default_user():
    from cognee.modules.users.methods import get_default_user

    return await get_default_user()


async def _background_cognify_and_memify(data: str):
    """Add data, cognify, then memify — all in the background."""
    with redirect_stdout(sys.stderr):
        try:
            import cognee

            logger.info("Background: adding data to cognee...")
            await cognee.add(data, dataset_name="main_dataset")

            logger.info("Background: running cognify...")
            await cognee.cognify()
            logger.info("Background: cognify complete.")

            logger.info("Background: running memify...")
            await cognee.memify()
            logger.info("Background: memify complete.")
        except Exception as e:
            timestamp = datetime.now(timezone.utc).isoformat()
            _task_errors[timestamp] = str(e)
            logger.error(f"Background cognify+memify failed: {e}")


# ---------------------------------------------------------------------------
# Tool: remember
# ---------------------------------------------------------------------------

@mcp.tool()
async def remember(information: str, session_id: Optional[str] = None) -> list:
    """Store information into Cognee's memory.

    This tool saves the provided information so it can be recalled later.
    The information is immediately available for search (via session cache)
    and is also processed in the background into the knowledge graph for
    deeper semantic retrieval.

    Use this whenever the user says something worth remembering — facts,
    preferences, decisions, context, or any information that should persist
    across conversations.

    Parameters
    ----------
    information : str
        The text to remember. Can be a fact, a conversation excerpt,
        a user preference, a decision, or any free-form text.
    session_id : str, optional
        Session identifier for grouping related memories.
        Defaults to "default_session" if not provided.

    Returns
    -------
    list
        Confirmation that the information was stored.
    """
    with redirect_stdout(sys.stderr):
        # 1. Instant write to session cache (available for immediate recall)
        try:
            session_mgr = await _get_session_manager()
            user = await _get_default_user()

            if session_mgr.is_available:
                await session_mgr.add_qa(
                    user_id=str(user.id),
                    question="[remember]",
                    context=information,
                    answer=information,
                    session_id=session_id,
                )
                logger.info("remember: stored in session cache.")
            else:
                logger.info("remember: session cache unavailable, skipping instant store.")
        except Exception as e:
            logger.warning(f"remember: session cache write failed ({e}), continuing with graph.")

        # 2. Background: add + cognify + memify
        asyncio.create_task(_background_cognify_and_memify(information))

    return [
        types.TextContent(
            type="text",
            text="Remembered. The information is stored and will be available for search.",
        )
    ]


# ---------------------------------------------------------------------------
# Tool: search_memory
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_memory(
    query: str,
    session_id: Optional[str] = None,
    top_k: int = 5,
) -> list:
    """Search Cognee's memory for relevant information.

    This tool retrieves information previously stored via the `remember` tool
    or any data processed through Cognee. It searches both recent session
    memory (fast, exact) and the long-term knowledge graph (semantic, deep).

    Use this whenever the user asks a question that might be answered by
    previously stored information, or when context from past conversations
    would be helpful.

    Parameters
    ----------
    query : str
        The search query in natural language.
        Examples:
        - "What did the user say about their preferences?"
        - "What are the key decisions we made?"
        - "Find information about project deadlines"
    session_id : str, optional
        Session to search within for recent memories.
        Defaults to "default_session" if not provided.
    top_k : int, optional
        Maximum number of graph results to return (default: 5).

    Returns
    -------
    list
        Combined results from session memory and knowledge graph.
    """
    results_parts = []

    with redirect_stdout(sys.stderr):
        # --- Layer 1: Session cache (recent, fast) ---
        try:
            session_mgr = await _get_session_manager()
            user = await _get_default_user()

            if session_mgr.is_available:
                session_entries = await session_mgr.get_session(
                    user_id=str(user.id),
                    session_id=session_id,
                    formatted=False,
                    last_n=20,
                )

                if isinstance(session_entries, list) and session_entries:
                    # Simple keyword matching on session entries for relevance
                    query_lower = query.lower()
                    query_words = set(query_lower.split())

                    relevant = []
                    for entry in session_entries:
                        content = entry.get("context", "") or entry.get("answer", "")
                        content_lower = content.lower()
                        # Score by word overlap
                        score = sum(1 for w in query_words if w in content_lower)
                        if score > 0:
                            relevant.append((score, content))

                    relevant.sort(key=lambda x: x[0], reverse=True)

                    if relevant:
                        session_texts = [text for _, text in relevant[:5]]
                        results_parts.append(
                            "## Recent Memory (Session)\n" + "\n---\n".join(session_texts)
                        )
                        logger.info(f"search_memory: found {len(session_texts)} session matches.")
        except Exception as e:
            logger.warning(f"search_memory: session search failed ({e}), continuing with graph.")

        # --- Layer 2: Knowledge graph (deep, semantic) ---
        try:
            import cognee
            from cognee.modules.search.types import SearchType

            graph_results = await cognee.search(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
                top_k=top_k,
            )

            if graph_results:
                if isinstance(graph_results, list) and len(graph_results) > 0:
                    graph_text = str(graph_results[0])
                else:
                    graph_text = str(graph_results)

                if graph_text.strip():
                    results_parts.append("## Long-term Memory (Knowledge Graph)\n" + graph_text)
                    logger.info("search_memory: got graph results.")
        except Exception as e:
            logger.warning(f"search_memory: graph search failed ({e}).")

    # Combine results
    if results_parts:
        combined = "\n\n".join(results_parts)
    else:
        combined = "No relevant memories found. Try storing information first with the `remember` tool."

    return [types.TextContent(type="text", text=combined)]


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Cognee Memory — minimal MCP server")

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE/HTTP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE/HTTP (default: 8000)")
    parser.add_argument(
        "--no-migration",
        default=False,
        action="store_true",
        help="Skip database migration on startup",
    )

    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    if not args.no_migration:
        from cognee.modules.engine.operations.setup import setup
        from cognee.run_migrations import run_migrations

        logger.info("Running database migrations...")
        await setup()
        await run_migrations()
        logger.info("Database migrations done.")

    match args.transport.lower():
        case "sse":
            logger.info(f"Cognee Memory MCP running on SSE {args.host}:{args.port}")
            await mcp.run_sse_async()
        case "http":
            logger.info(f"Cognee Memory MCP running on HTTP {args.host}:{args.port}")
            await mcp.run_streamable_http_async()
        case _:
            logger.info("Cognee Memory MCP running on stdio")
            await mcp.run_stdio_async()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee Memory MCP server: {e}")
        raise
