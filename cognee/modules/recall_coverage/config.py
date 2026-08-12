"""Configuration for recall coverage.

Every threshold the pipeline uses is a parameter here — there are no magic
numbers in the pipeline modules. The values below are reasonable arbitrary
starting points to be tuned against real data; the ones that change the meaning
of a reported number rather than just its quality are called out individually.

Env vars are prefixed ``RECALL_COVERAGE_`` (e.g.
``RECALL_COVERAGE_MAX_QUESTIONS``) so short field names like ``max_questions``
cannot collide with unrelated deployment variables.
"""

import json
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.modules.search.types import SearchType

# Tool label -> session-id prefixes. One label maps to one or more prefixes, so
# the resulting predicate is an OR group. Two further labels are literals the
# backend never mints from a prefix: "api" (unknown prefix, or no session at
# all) and "all" (no session predicate at all).
#
# Kept as a code-level default rather than a settings field because
# pydantic-settings has no mutable-collection default in this repo; the
# ``agent_prefix_map`` field below overrides it with JSON when a deployment
# needs to.
DEFAULT_AGENT_PREFIX_MAP: dict[str, tuple[str, ...]] = {
    "claude-desktop": ("claude_desktop_",),
    "claude-code": ("claude_", "cc_"),
    "codex": ("codex_",),
    "openclaw": ("openclaw_",),
    "hermes": ("hermes_",),
    "vscode": ("vscode_",),
    "cursor": ("cursor_",),
    "gemini": ("gemini_",),
    "cline": ("cline_",),
    "ui": ("search-ui-",),
}

# The search types that count as recall. CHUNKS, CHUNKS_LEXICAL, SUMMARIES,
# CYPHER and CODING_RULES are deliberately excluded: those are lookups, not a
# question memory was asked to answer, so judging them would score the wrong
# thing. Written as enum members so a renamed SearchType breaks at import.
DEFAULT_QUERY_TYPES: tuple[SearchType, ...] = (
    SearchType.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_COT,
    SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
    SearchType.GRAPH_SUMMARY_COMPLETION,
    SearchType.TEMPORAL,
    SearchType.RAG_COMPLETION,
)


class RecallCoverageConfig(BaseSettings):
    """Tunables for a recall-coverage run and its read endpoints."""

    model_config = SettingsConfigDict(env_prefix="RECALL_COVERAGE_", env_file=".env", extra="allow")

    # --- Window ------------------------------------------------------------
    # Hard cap on observed question rows per run, newest first. This is also
    # what keeps dedup non-quadratic in history size: the matmul is bounded by
    # this number, not by an index (there is no ANN index in this repo).
    max_questions: int = 150
    max_age_days: int = 30
    # Hard SQL LIMIT on the window fetch itself, far above ``max_questions``
    # because the collapse needs the raw rows to count retries and fan-outs.
    # Without it a busy deployment materializes every logged recall in the age
    # window as Python objects before the truncation runs; with it,
    # ``recall_row_count`` stays honest via a COUNT(*) when the cap is hit.
    window_row_cap: int = 50000

    # --- Collapse ----------------------------------------------------------
    # Identical text + query_type seen inside this window is one fan-out of a
    # single search. Merged only to avoid embedding the same string N times —
    # there is no fan-out counting rule.
    fanout_window_seconds: int = 5
    # Identical text from the same user against the same dataset inside this
    # window counts as ONE ask: an agent retrying is not demand. A heuristic;
    # what it swallowed is reported as ``collapsed_retry_count``.
    retry_cooldown_seconds: int = 300

    # --- Dedup -------------------------------------------------------------
    # Cosine similarity above which two questions in the same
    # (user_id, dataset_id) partition are the same question. Not a free knob:
    # once ``occurrence_count`` is displayed, this threshold *defines* the
    # demand metric.
    dedup_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    # --- Topic assignment --------------------------------------------------
    assignment_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    # Best topic must beat the runner-up by this much, otherwise the question
    # goes to the sink instead of being assigned on a coin flip.
    assignment_margin: float = Field(default=0.05, ge=0.0, le=1.0)

    # --- Topic suggestions -------------------------------------------------
    # Looser than ``dedup_threshold``: a suggestion is a theme, not a duplicate.
    sink_cluster_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # A candidate this close to an already dismissed or accepted suggestion is
    # dropped, so dismissing a suggestion actually sticks.
    suggestion_dedup_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    min_questions_per_topic: int = 5
    max_suggestions_per_run: int = 5
    topic_label_max_chars: int = 60

    # --- Curated questions ---------------------------------------------------
    # Cap per (owner, scope) bucket, enforced at creation. Curated questions are
    # a per-run cost multiplier that ``max_questions`` does not bound: each one
    # is replicated into every readable dataset partition and each row is a
    # replay plus up to three judge LLM calls.
    max_curated_questions: int = 100

    # --- Replay ------------------------------------------------------------
    replay_top_k: int = 15
    replay_max_concurrent: int = 5
    store_context_max_chars: int = 4000

    # --- Judge -------------------------------------------------------------
    judge_max_concurrent: int = 10
    judge_max_retries: int = 2
    judge_score_max: int = 5
    judge_reason_max_chars: int = 500

    # --- Aggregation -------------------------------------------------------
    # A topic with fewer scored rows than this reports avg_score null and is
    # excluded from overall_score, rather than letting one question speak for a
    # whole topic.
    min_scored_questions_per_topic: int = 3
    sink_share_alert: float = Field(default=0.30, ge=0.0, le=1.0)
    sink_cluster_alert_size: int = 10

    # --- Ceilings on per-request overrides ---------------------------------
    # ``POST /runs`` takes per-run parameter overrides, and the cost knobs are
    # among them: the window size, the age window, the retrieval depth and the two
    # concurrency limits each bill LLM calls or hold memory. Without an upper
    # bound any authenticated caller can ask for a run nobody would start —
    # ``max_questions = 200000`` is a 200000 x 200000 similarity matrix before a
    # single LLM call — and the ``(owner, agent_label)`` in-flight guard bounds
    # neither the labels nor the owners. These bound a *request*; the deployment's
    # own defaults above are trusted and are not checked against them.
    max_questions_ceiling: int = 1000
    max_age_days_ceiling: int = 365
    replay_top_k_ceiling: int = 100
    max_concurrent_ceiling: int = 50
    window_row_cap_ceiling: int = 500000
    # Retries multiply the cost of every failing LLM call, and the reason/context
    # sizes multiply prompt tokens and storage per row — all three are
    # per-request overridable, so all three need a ceiling.
    judge_max_retries_ceiling: int = 10
    judge_reason_max_chars_ceiling: int = 4000
    store_context_max_chars_ceiling: int = 100000

    # --- Runs --------------------------------------------------------------
    # How long a pending or running run keeps blocking the next run for the same
    # (owner, agent_label). A run is minutes of LLM calls, so a row still in
    # flight after this lost its process: a pod rescheduled mid-run leaves the
    # row at "running" for ever, and with no age bound the 409 guard would refuse
    # every later run for that pair with no way out — there is no cancel or
    # delete route, so the only fix would be manual SQL. The stale row is left
    # alone rather than marked failed: nobody knows how it ended.
    run_stale_after_seconds: int = 3600

    # --- Read endpoints ----------------------------------------------------
    runs_list_default_limit: int = 20
    # "I don't show 10, only show the top five."
    agents_list_default_limit: int = 5

    # --- Agent resolution --------------------------------------------------
    # JSON object mapping a tool label to a prefix string or a list of prefix
    # strings, e.g. '{"claude-code": ["claude_", "cc_"], "codex": "codex_"}'.
    # Empty (the default) means DEFAULT_AGENT_PREFIX_MAP.
    agent_prefix_map: str = ""
    # Granularity of a minted label. Only "tool" is implemented: every session
    # sharing a tool's prefix is one agent. Reserved so a future per-project
    # granularity is a config change rather than a schema change.
    agent_label_granularity: str = "tool"

    def prefix_map(self) -> dict[str, tuple[str, ...]]:
        """Return the effective label -> prefixes map.

        Malformed JSON or a wrong shape raises, so a misconfigured deployment
        fails loudly instead of silently resolving every label to no prefixes
        (which would report "nothing asked yet" for every agent). So does the same
        prefix appearing under two labels: "longest prefix wins" only makes
        classification a partition when each prefix has one owner, and two labels
        claiming ``claude_`` would report the same traffic twice under two names
        and replay and judge the identical window at double the LLM cost.
        """
        if not self.agent_prefix_map.strip():
            return dict(DEFAULT_AGENT_PREFIX_MAP)

        parsed = json.loads(self.agent_prefix_map)
        if not isinstance(parsed, dict):
            raise ValueError("RECALL_COVERAGE_AGENT_PREFIX_MAP must be a JSON object")

        prefix_map: dict[str, tuple[str, ...]] = {}
        for label, value in parsed.items():
            if isinstance(value, str):
                prefixes: tuple[str, ...] = (value,)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                prefixes = tuple(value)
            else:
                raise ValueError(
                    f"RECALL_COVERAGE_AGENT_PREFIX_MAP entry '{label}' must be a prefix "
                    "string or a list of prefix strings"
                )
            if not prefixes or any(not prefix for prefix in prefixes):
                raise ValueError(
                    f"RECALL_COVERAGE_AGENT_PREFIX_MAP entry '{label}' has an empty prefix"
                )
            prefix_map[label] = prefixes

        owner_of: dict[str, str] = {}
        for label, prefixes in prefix_map.items():
            for prefix in prefixes:
                if owner_of.setdefault(prefix, label) != label:
                    raise ValueError(
                        f"RECALL_COVERAGE_AGENT_PREFIX_MAP gives prefix '{prefix}' to both "
                        f"'{owner_of[prefix]}' and '{label}'; one prefix belongs to one label, "
                        "otherwise the same sessions are counted under two agents"
                    )

        return prefix_map

    def override_ceilings(self) -> dict[str, int]:
        """The most a single request may ask for, per overridable cost parameter.

        Only the parameters whose value costs money or memory are listed: the
        thresholds are already bounded to ``0..1`` by their own fields, and a
        request cannot make a run expensive by moving one. One shared ceiling
        covers both concurrency limits — they bound the same thing, in-flight
        calls, and a deployment that raises one has no reason to keep the other
        low.
        """
        return {
            "max_questions": self.max_questions_ceiling,
            "max_age_days": self.max_age_days_ceiling,
            "replay_top_k": self.replay_top_k_ceiling,
            "replay_max_concurrent": self.max_concurrent_ceiling,
            "judge_max_concurrent": self.max_concurrent_ceiling,
            "window_row_cap": self.window_row_cap_ceiling,
            "judge_max_retries": self.judge_max_retries_ceiling,
            "judge_reason_max_chars": self.judge_reason_max_chars_ceiling,
            "store_context_max_chars": self.store_context_max_chars_ceiling,
        }

    def query_types(self) -> list[str]:
        """Search types that count as recall, as the strings stored in ``queries``.

        ``Query.query_type`` is written from ``query_type.value``, so comparing
        against these values matches the stored column exactly.
        """
        return [query_type.value for query_type in DEFAULT_QUERY_TYPES]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_questions": self.max_questions,
            "max_age_days": self.max_age_days,
            "window_row_cap": self.window_row_cap,
            "max_curated_questions": self.max_curated_questions,
            "fanout_window_seconds": self.fanout_window_seconds,
            "retry_cooldown_seconds": self.retry_cooldown_seconds,
            "dedup_threshold": self.dedup_threshold,
            "assignment_threshold": self.assignment_threshold,
            "assignment_margin": self.assignment_margin,
            "sink_cluster_threshold": self.sink_cluster_threshold,
            "suggestion_dedup_threshold": self.suggestion_dedup_threshold,
            "min_questions_per_topic": self.min_questions_per_topic,
            "max_suggestions_per_run": self.max_suggestions_per_run,
            "topic_label_max_chars": self.topic_label_max_chars,
            "replay_top_k": self.replay_top_k,
            "replay_max_concurrent": self.replay_max_concurrent,
            "store_context_max_chars": self.store_context_max_chars,
            "judge_max_concurrent": self.judge_max_concurrent,
            "judge_max_retries": self.judge_max_retries,
            "judge_score_max": self.judge_score_max,
            "judge_reason_max_chars": self.judge_reason_max_chars,
            "min_scored_questions_per_topic": self.min_scored_questions_per_topic,
            "sink_share_alert": self.sink_share_alert,
            "sink_cluster_alert_size": self.sink_cluster_alert_size,
            "run_stale_after_seconds": self.run_stale_after_seconds,
            "override_ceilings": self.override_ceilings(),
            "runs_list_default_limit": self.runs_list_default_limit,
            "agents_list_default_limit": self.agents_list_default_limit,
            "agent_label_granularity": self.agent_label_granularity,
            "query_types": self.query_types(),
            "agent_prefix_map": {
                label: list(prefixes) for label, prefixes in self.prefix_map().items()
            },
        }


@lru_cache
def get_recall_coverage_config() -> RecallCoverageConfig:
    return RecallCoverageConfig()


def clear_recall_coverage_config_cache() -> None:
    """Clear the cached config. For tests that override thresholds."""
    get_recall_coverage_config.cache_clear()
