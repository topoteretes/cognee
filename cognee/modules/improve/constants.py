"""Names and defaults shared by the self-improvement loop.

One declaration per literal that the improve stages, their pipelines and the
visualization layer used to spell out by hand (plan Appendix D). Nothing here
changes a formula: ``DEFAULT_FEEDBACK_ALPHA`` is the learning rate the
feedback-weight task has always used. (The session-context confidence
threshold, ``GATE_CONFIDENCE``, lives with the models it gates in
``cognee.infrastructure.session.session_context_models`` — the loop never
reads it.)
"""

# Node sets written by the loop's stages.
USER_SESSIONS_NODE_SET = "user_sessions_from_cache"  # stage 2: persisted session Q&A
AGENT_TRACE_FEEDBACKS_NODE_SET = "agent_trace_feedbacks"  # stage 3: persisted tool-call traces
SESSION_LEARNINGS_NODE_SET = "session_learnings"  # stage 5: distilled lessons
USER_PREFERENCES_NODE_SET = "user_preferences"  # stage 6: per-user preference subgraph
SKILLS_NODE_SET = "skills"  # skill ingestion / skill improvement

# Learning rate for streaming feedback-weight updates, in (0, 1].
DEFAULT_FEEDBACK_ALPHA = 0.1

__all__ = [
    "AGENT_TRACE_FEEDBACKS_NODE_SET",
    "DEFAULT_FEEDBACK_ALPHA",
    "SESSION_LEARNINGS_NODE_SET",
    "SKILLS_NODE_SET",
    "USER_PREFERENCES_NODE_SET",
    "USER_SESSIONS_NODE_SET",
]
