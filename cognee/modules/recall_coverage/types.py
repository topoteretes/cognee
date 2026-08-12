"""Value objects and app-level enums for recall coverage.

Two jobs:

* the string enums behind the ``Column(String)`` status/scope/source columns in
  :mod:`cognee.modules.recall_coverage.models` — kept out of the database as a
  native type on purpose, because adding one value to a Postgres ``ENUM`` needs
  raw DDL (see ``cognee/alembic/versions/1d0bb7fede17_add_pipeline_run_status.py``);
* :class:`AgentScope`, the resolved agent selector, and :class:`CoverageParams`,
  the frozen-at-run-time parameter snapshot persisted on the run row.

Import-light by design (stdlib + pydantic only) so ``models.py`` can import the
enums without pulling the config, search or embedding stacks in behind them. The
one config lookup, :meth:`CoverageParams.from_config`, imports lazily.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

# The sink is a wire-level literal, not a row in ``recall_coverage_topics``:
# questions that matched no topic carry ``topic_id = NULL`` in the database and
# are reported under this id, so the sink can never be deleted or accepted.
SINK_TOPIC_ID = "other"
SINK_TOPIC_LABEL = "Other"

# Escape character passed as SQL ``LIKE ... ESCAPE`` when matching session-id
# prefixes. ``_`` is a single-character wildcard in LIKE, so an unescaped
# ``claude_%`` would also match ``claude_desktop_...`` — a different agent.
LIKE_ESCAPE_CHAR = "\\"


class RunStatus(str, Enum):
    """Lifecycle of one ``recall_coverage_runs`` row.

    Owned by this module rather than by ``PipelineRun``: a run spans every
    dataset in the tenant, while ``PipelineRun.dataset_id`` is scalar.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class QuestionSource(str, Enum):
    """Where a question row came from."""

    OBSERVED = "observed"
    CURATED = "curated"


class SuggestionStatus(str, Enum):
    """Lifecycle of a suggested topic. Accept, dismiss — never rename or merge."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"


class CuratedScope(str, Enum):
    """Scope of a curated question.

    ``AGENT`` rows belong to one ``agent_label``. ``SHARED`` rows are the
    benchmark set — identical prompts across every agent, which is the only
    reason ``benchmark_score_pct`` is comparable between agents at all.
    """

    AGENT = "agent"
    SHARED = "shared"


class AgentScopeMode(str, Enum):
    """How an :class:`AgentScope` turns into a session-id predicate.

    ``PREFIX``  — match any of the scope's prefixes.
    ``NEGATED`` — the ``api`` label: no known prefix matched, or no session at
                  all (``session_id IS NULL``).
    ``ALL``     — no session predicate whatsoever. A first-class permanent mode
                  ("how is my memory doing overall"), and the only mode that
                  returns rows until ``Query.session_id`` ships.
    """

    PREFIX = "prefix"
    NEGATED = "negated"
    ALL = "all"


@dataclass(frozen=True)
class AgentScope:
    """A validated agent selector.

    ``prefixes`` are already LIKE-escaped and sorted longest-first, so callers
    build predicates from them directly and classification tries
    ``claude_desktop_`` before ``claude_``. Everything downstream
    (``run_recall_coverage``, the repository fetch) takes this object rather
    than a raw label string, so an unvalidated label cannot reach a query.
    """

    label: str
    prefixes: tuple[str, ...] = ()
    mode: AgentScopeMode = AgentScopeMode.ALL


class CoverageParams(BaseModel):
    """The parameter set one run executed under.

    Persisted verbatim into ``recall_coverage_runs.params`` so a historical run
    stays readable after the deployment's defaults move. Every field is
    required and has no default here — :class:`RecallCoverageConfig` is the
    single source of the defaults, and this is the resolved snapshot of them.
    Use :meth:`from_config`.
    """

    model_config = ConfigDict(extra="forbid")

    # Phase 1 — window, collapse, dedup.
    max_questions: int
    max_age_days: int
    query_types: list[str]
    fanout_window_seconds: int
    retry_cooldown_seconds: int
    dedup_threshold: float

    # Phase 2 — topic assignment and suggestions.
    assignment_threshold: float
    assignment_margin: float
    sink_cluster_threshold: float
    suggestion_dedup_threshold: float
    min_questions_per_topic: int
    min_scored_questions_per_topic: int
    max_suggestions_per_run: int
    topic_label_max_chars: int

    # Phase 3 — replay and judge.
    replay_top_k: int
    replay_max_concurrent: int
    judge_max_concurrent: int
    judge_max_retries: int
    judge_score_max: int
    judge_reason_max_chars: int
    store_context_max_chars: int

    # Phase 4 — alert thresholds.
    sink_share_alert: float
    sink_cluster_alert_size: int

    @classmethod
    def from_config(cls, config: Optional[Any] = None, **overrides: Any) -> "CoverageParams":
        """Snapshot the deployment defaults, then apply per-request overrides.

        ``config`` is a :class:`RecallCoverageConfig`; it is fetched lazily when
        omitted so importing this module stays cheap. Unknown override keys
        raise (``extra="forbid"``) instead of being silently dropped, which
        would let a typo'd request parameter look accepted.
        """
        if config is None:
            from cognee.modules.recall_coverage.config import get_recall_coverage_config

            config = get_recall_coverage_config()

        values: dict[str, Any] = {
            name: getattr(config, name) for name in cls.model_fields if name != "query_types"
        }
        values["query_types"] = config.query_types()
        values.update(overrides)
        return cls(**values)
