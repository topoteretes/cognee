"""Unit-test patch coverage for operations gated by LLM availability."""

KEYLESS_GATE_PATCH_TARGETS = (
    "cognee.infrastructure.session.agent_context_extraction.llm_available",
    "cognee.infrastructure.session.feedback_detection.llm_available",
    "cognee.infrastructure.session.session_agent_trace.llm_available",
    "cognee.modules.improve.stages.llm_available",
)

# Recall must observe the real keyless configuration because its CHUNKS default
# is itself under test; so must the telemetry settings payload, which only
# reports whether the LLM is usable (``llm.configured``) and gates nothing.
# Every other module-level import is a gated operation whose existing unit tests
# mock the downstream LLM call.
KEYLESS_GATE_IMPORT_EXCLUSIONS = {
    "cognee.api.v1.recall.recall",
    "cognee.modules.settings.get_current_settings",
}
