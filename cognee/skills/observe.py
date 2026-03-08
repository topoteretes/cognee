"""Observe layer: record SkillRun executions to short-term cache.

Runs are stored as QA entries in SessionManager under a namespaced session
(skill_runs:{session_id}). Use promote_skill_runs() to move worthy runs
into the long-term graph.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from cognee.low_level import setup
from cognee.infrastructure.session.get_session_manager import get_session_manager


logger = logging.getLogger(__name__)

CACHE_USER_ID = "skill_runs_user"


def _session_key(session_id: str) -> str:
    return f"skill_runs:{session_id}"


async def record_skill_run(
    session_id: str,
    task_text: str,
    selected_skill_id: str,
    task_pattern_id: str = "",
    result_summary: str = "",
    success_score: float = 0.0,
    candidate_skills: Optional[List[Dict[str, Any]]] = None,
    router_version: str = "",
    tool_trace: Optional[List[Dict[str, Any]]] = None,
    feedback: float = 0.0,
    error_type: str = "",
    error_message: str = "",
    cognee_session_id: str = "",
    latency_ms: int = 0,
) -> dict:
    """Record a skill execution to the short-term cache.

    Returns the run data dict (including cache_qa_id for later promotion).
    """
    await setup()

    run_data = {
        "session_id": session_id,
        "task_text": task_text,
        "selected_skill_id": selected_skill_id,
        "task_pattern_id": task_pattern_id,
        "result_summary": result_summary,
        "success_score": success_score,
        "candidate_skills": candidate_skills or [],
        "router_version": router_version,
        "tool_trace": tool_trace or [],
        "feedback": feedback,
        "error_type": error_type,
        "error_message": error_message,
        "cognee_session_id": cognee_session_id,
        "latency_ms": latency_ms,
        "started_at_ms": int(time.time() * 1000),
    }

    sm = get_session_manager()
    qa_id = await sm.add_qa(
        user_id=CACHE_USER_ID,
        question=task_text,
        context=json.dumps(run_data),
        answer=json.dumps({"result_summary": result_summary, "success_score": success_score}),
        session_id=_session_key(session_id),
    )

    run_data["cache_qa_id"] = qa_id or ""

    logger.info(
        "Cached SkillRun: session=%s skill=%s score=%.2f qa_id=%s",
        session_id,
        selected_skill_id,
        success_score,
        qa_id,
    )
    return run_data
