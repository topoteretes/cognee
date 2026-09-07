"""Every module that used to spell a loop constant by hand now reads the shared one.

Plan Appendix D: node-set names, the learning-rate default and the single confidence
threshold are declared once in ``cognee.modules.improve.constants``. These tests pin the
wiring, not the values, so a renamed node set changes in one place.
"""

import inspect

from cognee.infrastructure.session import session_context_models
from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
    persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
)
from cognee.modules.improve import constants
from cognee.modules.memify import skill_improvement
from cognee.modules.session_distillation import distill
from cognee.modules.session_distillation import models as distillation_models
from cognee.modules.truth_subspace import constants as truth_constants
from cognee.modules.user_preferences import constants as preference_constants
from cognee.modules.visualization import preprocessor
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback


def test_constants_module_imports_nothing_from_cognee():
    """It sits below every consumer, so it must never import back into the package."""
    source = inspect.getsource(constants)
    assert "import cognee" not in source
    assert "from cognee" not in source


def test_confidence_threshold_is_declared_once():
    assert session_context_models.MIN_CANDIDATE_CONFIDENCE is constants.GATE_CONFIDENCE
    assert session_context_models.MIN_GATE_CONFIDENCE is constants.GATE_CONFIDENCE
    assert distillation_models.MIN_GATE_CONFIDENCE is constants.GATE_CONFIDENCE
    assert constants.GATE_CONFIDENCE == 0.75


def test_session_learnings_node_set_is_shared():
    assert distill.DISTILLATE_NODE_SET == [constants.SESSION_LEARNINGS_NODE_SET]
    assert truth_constants.TRUTH_NODE_SET == [constants.SESSION_LEARNINGS_NODE_SET]
    assert truth_constants.truth_session_node_set("s1") == (
        f"{constants.SESSION_LEARNINGS_NODE_SET}:s1"
    )
    assert preprocessor._DISTILLED_LEARNING_NODE_SET == constants.SESSION_LEARNINGS_NODE_SET


def test_agent_trace_node_set_is_the_default_in_task_and_pipeline():
    task_default = inspect.signature(cognify_agent_trace_feedback).parameters["node_set_name"]
    pipeline_default = inspect.signature(
        persist_agent_trace_feedbacks_in_knowledge_graph_pipeline
    ).parameters["node_set_name"]
    assert task_default.default == constants.AGENT_TRACE_FEEDBACKS_NODE_SET
    assert pipeline_default.default == constants.AGENT_TRACE_FEEDBACKS_NODE_SET


def test_user_preferences_and_skills_node_sets_are_shared():
    assert preference_constants.PREFERENCE_NODE_SET == constants.USER_PREFERENCES_NODE_SET
    assert skill_improvement._skills_node_set().name == constants.SKILLS_NODE_SET


def test_visualization_pins_colors_for_every_bridge_node_set():
    for name in (
        constants.SESSION_LEARNINGS_NODE_SET,
        constants.USER_SESSIONS_NODE_SET,
        constants.AGENT_TRACE_FEEDBACKS_NODE_SET,
        constants.SKILLS_NODE_SET,
    ):
        assert name in preprocessor._MEMORY_NODESET_COLORS
    colors = preprocessor.build_node_set_colors([constants.SKILLS_NODE_SET, "slack"])
    assert (
        colors[constants.SKILLS_NODE_SET]
        == preprocessor._MEMORY_NODESET_COLORS[constants.SKILLS_NODE_SET]
    )
