"""Interactive CLI adapter for development and testing.

This adapter requires **zero** external tokens or platform accounts.
It simulates a support channel in the terminal with commands:

  - Type any text        → triggers triage (recall similar issues)
  - !resolve <id> <msgs> → simulate resolving a thread
  - !forget <id>         → forget a specific thread
  - !optout              → opt out of future ingestion
  - !quit / !exit        → stop the bot
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import ChannelAdapter, Message

logger = logging.getLogger(__name__)

# Pre-seeded threads for demonstration.
# These are resolved support threads that the bot "remembers" for recall.
SEED_THREADS = [
    {
        "thread_id": "T001",
        "channel_id": "cli",
        "reporter": "alice",
        "problem_summary": "Auth timeout after token refresh — users get 401 after refreshing their session token",
        "resolution_summary": "Bumped token TTL from 1h to 24h in auth config. The short TTL caused race conditions during concurrent refresh requests.",
        "messages": [
            "Users are getting 401 errors after token refresh",
            "Looks like the token TTL is too short at 1 hour",
            "Fixed by bumping TTL to 24h in auth config — the short TTL caused race conditions during concurrent refresh",
        ],
        "thread_url": "https://support.example.com/threads/T001",
    },
    {
        "thread_id": "T002",
        "channel_id": "cli",
        "reporter": "bob",
        "problem_summary": "Session expiry on mobile app — mobile SDK uses a shorter TTL than web",
        "resolution_summary": "Same root cause as T001 — mobile SDK had its own shorter TTL of 30min. Aligned mobile TTL with web TTL (24h).",
        "messages": [
            "Mobile app sessions expire way too fast",
            "Found that mobile SDK uses a 30min TTL vs 1h on web",
            "Aligned mobile TTL with web (24h) — same root cause as the auth timeout issue",
        ],
        "thread_url": "https://support.example.com/threads/T002",
    },
    {
        "thread_id": "T003",
        "channel_id": "cli",
        "reporter": "carol",
        "problem_summary": "Database connection pool exhaustion during peak traffic — PostgreSQL max_connections exceeded",
        "resolution_summary": "Increased PgBouncer pool size from 20 to 100 and added connection timeout of 30s. Also added monitoring alerts.",
        "messages": [
            "Getting 'too many connections' errors during peak hours",
            "PgBouncer pool was set to 20, way too low for our traffic",
            "Increased pool to 100, added 30s timeout, and set up monitoring alerts for connection count",
        ],
        "thread_url": "https://support.example.com/threads/T003",
    },
]


class CLIAdapter(ChannelAdapter):
    """Interactive terminal adapter for development/testing."""

    def __init__(self) -> None:
        self._threads: dict[str, list[Message]] = {}
        self._output_buffer: list[str] = []

    async def start(self) -> None:
        """Not used by CLI — the event loop is driven by run_cli()."""
        pass

    async def send_reply(
        self,
        channel_id: str,
        thread_id: str,
        text: str,
        ephemeral_user: Optional[str] = None,
    ) -> None:
        """Print the reply to stdout."""
        prefix = f"[ephemeral → {ephemeral_user}] " if ephemeral_user else ""
        output = f"\n{prefix}📬 Bot Reply (thread={thread_id}):\n{text}\n"
        print(output)
        self._output_buffer.append(output)

    async def fetch_thread_messages(
        self, channel_id: str, thread_id: str
    ) -> list[Message]:
        """Return messages from the in-memory thread store."""
        return self._threads.get(thread_id, [])

    async def get_thread_permalink(
        self, channel_id: str, thread_id: str
    ) -> str:
        """Return a simulated permalink."""
        return f"https://support.example.com/threads/{thread_id}"

    def add_thread(self, thread_id: str, messages: list[Message]) -> None:
        """Add a thread to the in-memory store (for testing)."""
        self._threads[thread_id] = messages
