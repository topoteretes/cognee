"""Print a deterministic JSON demo of session-context growth.

Run with:

    uv run python examples/demos/session_context_growth_demo.py

This demo isolates the session loop. It does not call a live LLM or retriever. Instead, it feeds
simulated feedback-analysis results into ``SessionManager.prepare_session_turn`` and prints how
context is learned, served, and rated across a short conversation.
"""

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

os.environ.setdefault("COGNEE_CLI_MODE", "true")
os.environ.setdefault("COGNEE_LOG_FILE", "false")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LOG_LEVEL", "ERROR")

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult
from cognee.infrastructure.session.session_context_builder import build_active_context_block
from cognee.infrastructure.session.session_context_models import (
    CandidateContextUpdate,
    ServedContextRating,
)
from cognee.infrastructure.session.session_manager import SessionManager, SessionTurnPreparation

USER_ID = "demo-user"
SESSION_ID = "demo-session"


@dataclass
class DemoUser:
    id: str = USER_ID


@dataclass
class DemoTurn:
    user_message: str
    simulated_analysis: FeedbackDetectionResult
    generated_answer: str | None = None


def create_demo_turn(
    *,
    user_message: str,
    generated_answer: str | None = None,
    response_to_user: str | None = None,
    query_to_answer: str | None = None,
    learned_section: str | None = None,
    learned_content: str | None = None,
    learned_confidence: float = 0.9,
    rated_context_id: str | None = None,
    context_rating: str | None = None,
) -> DemoTurn:
    candidate_updates = []
    if learned_section and learned_content:
        candidate_updates.append(
            CandidateContextUpdate(
                section=learned_section,
                content=learned_content,
                confidence=learned_confidence,
            )
        )

    served_context_ratings = []
    if rated_context_id and context_rating:
        served_context_ratings.append(
            ServedContextRating(entry_id=rated_context_id, rating=context_rating)
        )

    return DemoTurn(
        user_message=user_message,
        generated_answer=generated_answer,
        simulated_analysis=FeedbackDetectionResult(
            response_to_user=response_to_user,
            query_to_answer=query_to_answer,
            candidate_context_updates=candidate_updates,
            served_context_ratings=served_context_ratings,
        ),
    )


def demo_turns() -> list[DemoTurn]:
    """Conversation script plus the feedback-analysis result for each user turn."""
    turns = []

    user_message = "How should we format API examples?"
    answer = "Use short examples that show the request and the expected response."
    turns.append(
        create_demo_turn(
            user_message=user_message,
            generated_answer=answer,
            query_to_answer=user_message,
        )
    )

    user_message = "That was helpful. For future answers, prefer bullet points."
    learned_preference = "Prefer bullet points in future answers."
    turns.append(
        create_demo_turn(
            user_message=user_message,
            response_to_user="Thanks for your feedback.",
            learned_section="preferences",
            learned_content=learned_preference,
            learned_confidence=0.92,
        )
    )

    user_message = "How should I explain the retry flow?"
    answer = "Start with the trigger, then list retry limits and failure behavior."
    turns.append(
        create_demo_turn(
            user_message=user_message,
            generated_answer=answer,
            query_to_answer=user_message,
        )
    )

    user_message = "That answer was too verbose, but what should I test?"
    query_to_answer = "What should I test?"
    learned_rule = "Keep implementation plans concise."
    answer = "Test feedback detection, context updates, and served context IDs."
    turns.append(
        create_demo_turn(
            user_message=user_message,
            generated_answer=answer,
            response_to_user="Thanks, noted.",
            query_to_answer=query_to_answer,
            learned_section="rules",
            learned_content=learned_rule,
            learned_confidence=0.88,
            rated_context_id="ctx-1",
            context_rating="helpful",
        )
    )

    return turns


async def run_demo() -> list[dict]:
    manager = SessionManager(cache_engine=InMemorySessionCache())
    token = session_user.set(DemoUser())

    try:
        turns = demo_turns()
        with (
            patch("cognee.infrastructure.session.session_manager.CacheConfig", demo_cache_config),
            patch(
                "cognee.infrastructure.session.session_manager.analyze_turn_for_session_context",
                new_callable=AsyncMock,
                side_effect=[turn.simulated_analysis for turn in turns],
            ),
            patch(
                "cognee.infrastructure.session.session_manager.uuid.uuid4",
                side_effect=["qa-1", "feedback-1", "qa-2", "feedback-2", "qa-3"],
            ),
            patch(
                "cognee.infrastructure.session.session_context_builder.uuid4",
                side_effect=["ctx-1", "ctx-2"],
            ),
        ):
            output = []
            for turn_number, turn in enumerate(turns, start=1):
                output.append(await run_turn(manager, turn_number, turn))
            return output
    finally:
        session_user.reset(token)


async def run_turn(manager: SessionManager, turn_number: int, turn: DemoTurn) -> dict:
    preparation = await manager.prepare_session_turn(
        query=turn.user_message,
        session_id=SESSION_ID,
    )

    served_context = {"ids": [], "block": ""}
    stored_qa = None
    if preparation.should_answer and turn.generated_answer:
        served_context = await build_context_for_answer(manager, preparation)
        stored_qa = await store_demo_answer(manager, turn, served_context["ids"])

    return {
        "turn": turn_number,
        "user_message": turn.user_message,
        "simulated_feedback_analysis": serialize_analysis(turn.simulated_analysis),
        "decision": serialize_decision(preparation),
        "context_served_to_answer": served_context,
        "stored_qa": stored_qa,
        "session_context_after_turn": await context_snapshot(manager),
    }


async def build_context_for_answer(
    manager: SessionManager,
    preparation: SessionTurnPreparation,
) -> dict:
    block, served_ids = await build_active_context_block(
        session_manager=manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        query=preparation.effective_query,
    )
    return {"ids": served_ids, "block": block}


async def store_demo_answer(
    manager: SessionManager,
    turn: DemoTurn,
    served_ids: list[str],
) -> dict:
    await manager.add_qa(
        user_id=USER_ID,
        session_id=SESSION_ID,
        question=turn.user_message,
        context="",
        answer=turn.generated_answer or "",
        used_session_context_ids=served_ids or None,
    )
    return {
        "question": turn.user_message,
        "answer": turn.generated_answer,
        "used_session_context_ids": served_ids,
    }


def serialize_analysis(analysis: FeedbackDetectionResult) -> dict:
    return {
        "query_to_answer": analysis.query_to_answer,
        "response_to_user": analysis.response_to_user,
        "candidate_context_updates": [
            candidate.model_dump() for candidate in analysis.candidate_context_updates
        ],
        "served_context_ratings": [
            rating.model_dump() for rating in analysis.served_context_ratings
        ],
    }


def serialize_decision(preparation: SessionTurnPreparation) -> dict:
    return {
        "should_answer": preparation.should_answer,
        "acknowledgement": preparation.response_to_user,
        "effective_query": preparation.effective_query,
        "previous_qa_id": preparation.previous_qa_id,
        "accepted_context_ids": preparation.accepted_context_ids,
    }


async def context_snapshot(manager: SessionManager) -> list[dict]:
    entries = await manager.get_session_context_entries(user_id=USER_ID, session_id=SESSION_ID)
    return [
        {
            "id": entry["id"],
            "section": entry["section"],
            "content": entry["content"],
            "helpful_count": entry.get("helpful_count", 0),
            "harmful_count": entry.get("harmful_count", 0),
        }
        for entry in entries
        if entry.get("kind", "context") == "context"
    ]


def demo_cache_config():
    return SimpleNamespace(caching=True, auto_feedback=True, max_session_context_chars=None)


class InMemorySessionCache:
    """Tiny async cache with only the methods ``SessionManager`` uses in this demo."""

    def __init__(self):
        self.qa_entries = defaultdict(list)
        self.context_entries = defaultdict(list)
        self._cache = {}
        self.session_ttl_seconds = None

    @staticmethod
    def _key(user_id: str, session_id: str) -> tuple[str, str]:
        return user_id, session_id

    async def create_qa_entry(
        self,
        user_id,
        session_id,
        qa_id,
        question,
        context,
        answer,
        feedback_text=None,
        feedback_score=None,
        used_graph_element_ids=None,
        used_session_context_ids=None,
    ):
        self.qa_entries[self._key(user_id, session_id)].append(
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                qa_id=qa_id,
                question=question,
                context=context,
                answer=answer,
                feedback_text=feedback_text,
                feedback_score=feedback_score,
                used_graph_element_ids=used_graph_element_ids,
                used_session_context_ids=used_session_context_ids,
            )
        )

    async def get_all_qa_entries(self, user_id, session_id):
        return list(self.qa_entries[self._key(user_id, session_id)])

    async def get_latest_qa_entries(self, user_id, session_id, last_n=1):
        return list(self.qa_entries[self._key(user_id, session_id)][-last_n:])

    async def update_qa_entry(
        self,
        user_id,
        session_id,
        qa_id,
        question=None,
        context=None,
        answer=None,
        feedback_text=None,
        feedback_score=None,
        used_graph_element_ids=None,
        memify_metadata=None,
        used_session_context_ids=None,
    ):
        for index, entry in enumerate(self.qa_entries[self._key(user_id, session_id)]):
            if entry.qa_id != qa_id:
                continue
            data = entry.model_dump()
            for field, value in {
                "question": question,
                "context": context,
                "answer": answer,
                "feedback_text": feedback_text,
                "feedback_score": feedback_score,
                "used_graph_element_ids": used_graph_element_ids,
                "memify_metadata": memify_metadata,
                "used_session_context_ids": used_session_context_ids,
            }.items():
                if value is not None:
                    data[field] = value
            self.qa_entries[self._key(user_id, session_id)][index] = SessionQAEntry(**data)
            return True
        return False

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        self.context_entries[self._key(user_id, session_id)].append(dict(entry_dump))
        return True

    async def get_session_context_entries(self, user_id, session_id):
        return [dict(entry) for entry in self.context_entries[self._key(user_id, session_id)]]

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        for entry in self.context_entries[self._key(user_id, session_id)]:
            if entry.get("id") == entry_id:
                entry.update(merge)
                return True
        return False

    async def delete_session_context(self, user_id, session_id):
        self.context_entries.pop(self._key(user_id, session_id), None)
        return True


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), indent=2))
