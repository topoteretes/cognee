"""Event handlers for the Support-Triage Bot.

Each handler implements one of the bot's core workflows:
- TriageHandler: Recall similar issues and suggest answers
- IngestHandler: Store resolved threads in memory
- ForgetHandler: Remove threads from memory
- OptOutHandler: Block future ingestion for a user
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from citation_builder import format_triage_result
from config import BotConfig
from memory_adapter import MemoryAdapter
from models import Citation, SupportThread, TriageResult

logger = logging.getLogger(__name__)


class TriageHandler:
    """Handle new support messages by recalling similar past issues.

    Workflow:
        1. Receive new support message
        2. Call cognee.recall() via the adapter
        3. Build citations from results
        4. Format ephemeral reply (human-in-the-loop)
    """

    def __init__(self, adapter: MemoryAdapter, config: BotConfig) -> None:
        self._adapter = adapter
        self._config = config

    async def handle(self, query: str, channel_id: str) -> TriageResult:
        """Triage a new support issue.

        Args:
            query: The support question/issue text.
            channel_id: Channel where the issue was posted.

        Returns:
            TriageResult with citations and suggested reply.
        """
        citations = await self._adapter.find_similar_issues(query, channel_id)

        # Build suggested reply from citations
        suggested_reply = ""
        confidence = 0.0

        if citations:
            # Build a suggestion from the top citation
            top = citations[0]
            suggested_reply = (
                f"Based on past resolutions, this may be related to: "
                f"{top.resolution_summary}"
            )
            # Confidence is based on best score (if available)
            if top.similarity_score is not None:
                confidence = min(top.similarity_score, 1.0)
            else:
                confidence = 0.5  # Default when no score

        result = TriageResult(
            query=query,
            citations=citations,
            suggested_reply=suggested_reply,
            confidence=confidence,
        )

        logger.info(
            "Triage complete: query='%.60s…' citations=%d confidence=%.2f",
            query,
            len(citations),
            confidence,
        )
        return result

    def format_reply(self, result: TriageResult) -> str:
        """Format the triage result into a human-readable message."""
        return format_triage_result(result)


class IngestHandler:
    """Handle resolved support threads by ingesting them into memory.

    Workflow:
        1. Receive resolve signal (✅ reaction or !resolve command)
        2. Build SupportThread from thread messages
        3. Call cognee.remember() via the adapter
        4. Persist thread_id ↔ data_id mapping
    """

    def __init__(
        self,
        adapter: MemoryAdapter,
        config: BotConfig,
        opt_out_list: set[str],
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._opt_out_list = opt_out_list

    async def handle(
        self,
        thread_id: str,
        channel_id: str,
        reporter: str,
        messages: list[str],
        thread_url: str = "",
    ) -> dict:
        """Ingest a resolved thread into memory.

        Args:
            thread_id: Platform thread identifier.
            channel_id: Channel where the thread lives.
            reporter: User who reported the issue.
            messages: Ordered messages in the thread.
            thread_url: Permalink to the thread.

        Returns:
            Ingestion result dict.
        """
        # Check opt-out
        if reporter in self._opt_out_list:
            logger.info(
                "Skipping ingestion for opted-out user: %s (thread %s)",
                reporter,
                thread_id,
            )
            return {"status": "skipped", "reason": "user_opted_out"}

        # Build thread document
        # Use first message as problem, last as resolution
        problem = messages[0] if messages else "Unknown problem"
        resolution = messages[-1] if len(messages) > 1 else "Resolution not recorded"

        thread = SupportThread(
            thread_id=thread_id,
            channel_id=channel_id,
            reporter=reporter,
            problem_summary=problem,
            resolution_summary=resolution,
            messages=messages,
            resolved_at=datetime.utcnow(),
            thread_url=thread_url,
        )

        result = await self._adapter.ingest_resolved_thread(thread)
        logger.info("Ingested thread %s: %s", thread_id, result.get("status"))
        return result


class ForgetHandler:
    """Handle !forget commands by removing threads from memory.

    Workflow:
        1. Parse thread_id from command
        2. Look up data_id from mapping
        3. Call cognee.forget() via the adapter
    """

    def __init__(self, adapter: MemoryAdapter) -> None:
        self._adapter = adapter

    async def handle(self, thread_id: str) -> dict:
        """Forget a specific thread.

        Args:
            thread_id: The platform thread identifier to forget.

        Returns:
            Dict with deletion result or error info.
        """
        try:
            result = await self._adapter.forget_thread(thread_id)
            logger.info("Forgot thread %s", thread_id)
            return {"status": "success", "thread_id": thread_id, **result}
        except KeyError as e:
            logger.warning("Forget failed: %s", e)
            return {"status": "error", "message": str(e)}


class OptOutHandler:
    """Handle !optout commands by adding users to the blocklist.

    Once opted out, the IngestHandler will skip ingestion for that
    user's resolved threads.
    """

    def __init__(self, opt_out_list: set[str]) -> None:
        self._opt_out_list = opt_out_list

    def handle(self, user_id: str) -> dict:
        """Register a user's opt-out.

        Args:
            user_id: The user identifier to opt out.

        Returns:
            Dict confirming the opt-out.
        """
        self._opt_out_list.add(user_id)
        logger.info("User %s opted out of support thread ingestion", user_id)
        return {
            "status": "success",
            "message": f"User {user_id} has opted out. "
            f"Future resolved threads will not be ingested.",
        }
