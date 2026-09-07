"""The self-improvement loop's orchestration types (plan Part 5).

Every type here is improve-local. The stage bodies stay where they are; this
package is the glue: a frozen argument bundle per run, one result shape on
``PipelineRunInfo``'s statuses plus ``skipped``, a capability probe, a config
object owning only the loop's own knobs, and the registry that is the only
description of the chain.
"""

from .capabilities import GraphCapabilities, probe_graph_capabilities, resolve_graph_capabilities
from .config import ImproveConfig, get_improve_config
from .constants import (
    AGENT_TRACE_FEEDBACKS_NODE_SET,
    DEFAULT_FEEDBACK_ALPHA,
    GATE_CONFIDENCE,
    SESSION_LEARNINGS_NODE_SET,
    SKILLS_NODE_SET,
    USER_PREFERENCES_NODE_SET,
    USER_SESSIONS_NODE_SET,
)
from .inputs import MEMIFY_PASSTHROUGH_KEYS, ImproveRunInputs
from .registry import DEFAULT_STAGES, stage_names, validate_stage_order
from .result import (
    REASON_ABORTED_BY_FATAL_STAGE,
    REASON_BACKEND_UNSUPPORTED,
    REASON_DISABLED_BY_CONFIG,
    REASON_LOCK_HELD,
    REASON_NO_SESSION_IDS,
    ImproveResult,
    StageResult,
)
from .stage import BaseStage, ImproveStage, evaluate_gate

__all__ = [
    "AGENT_TRACE_FEEDBACKS_NODE_SET",
    "BaseStage",
    "DEFAULT_FEEDBACK_ALPHA",
    "DEFAULT_STAGES",
    "GATE_CONFIDENCE",
    "GraphCapabilities",
    "ImproveConfig",
    "ImproveResult",
    "ImproveRunInputs",
    "ImproveStage",
    "MEMIFY_PASSTHROUGH_KEYS",
    "REASON_ABORTED_BY_FATAL_STAGE",
    "REASON_BACKEND_UNSUPPORTED",
    "REASON_DISABLED_BY_CONFIG",
    "REASON_LOCK_HELD",
    "REASON_NO_SESSION_IDS",
    "SESSION_LEARNINGS_NODE_SET",
    "SKILLS_NODE_SET",
    "StageResult",
    "USER_PREFERENCES_NODE_SET",
    "USER_SESSIONS_NODE_SET",
    "evaluate_gate",
    "get_improve_config",
    "probe_graph_capabilities",
    "resolve_graph_capabilities",
    "stage_names",
    "validate_stage_order",
]
