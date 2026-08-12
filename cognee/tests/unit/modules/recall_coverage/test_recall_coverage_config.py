"""Config, value-object and exception guards for recall coverage.

Every threshold the pipeline uses has to be a config parameter — that is the
whole point of ``RecallCoverageConfig`` — and :class:`CoverageParams` has to be a
faithful, JSON-round-trippable snapshot of it, because it is persisted onto the
run row so a historical run stays readable after the defaults move.

Config objects are constructed with ``_env_file=None`` throughout: a developer's
local ``.env`` must not be able to change what these assertions mean.
"""

import json

import pytest
from dataclasses import FrozenInstanceError

from cognee.modules.recall_coverage import config as config_module
from cognee.modules.recall_coverage.config import (
    DEFAULT_AGENT_PREFIX_MAP,
    RecallCoverageConfig,
    clear_recall_coverage_config_cache,
    get_recall_coverage_config,
)
from cognee.modules.recall_coverage.exceptions import (
    CoverageRunInFlightError,
    CoverageRunNotFoundError,
    CoverageSuggestionNotFoundError,
    CoverageSuggestionNotPendingError,
    CoverageTopicNotFoundError,
    CuratedQuestionNotFoundError,
    DegenerateEmbeddingError,
    DuplicateCuratedQuestionError,
    EmbeddingFingerprintMismatchError,
    InvalidCuratedQuestionScopeError,
    SinkTopicNotEditableError,
    UnknownAgentLabelError,
)
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_ID,
    AgentScope,
    AgentScopeMode,
    CoverageParams,
    CuratedScope,
    QuestionSource,
    RunStatus,
    SuggestionStatus,
)


def _config(**overrides) -> RecallCoverageConfig:
    return RecallCoverageConfig(_env_file=None, **overrides)


def test_documented_defaults():
    """The defaults the spec fixes by number, so a silent retune is visible."""
    config = _config()
    assert config.max_questions == 150
    assert config.max_age_days == 30
    assert config.replay_top_k == 15
    assert config.judge_score_max == 5
    assert config.min_questions_per_topic == 5
    assert config.agents_list_default_limit == 5
    assert config.fanout_window_seconds == 5
    assert config.retry_cooldown_seconds == 300


def test_similarity_thresholds_are_bounded():
    with pytest.raises(ValueError):
        _config(dedup_threshold=1.5)
    with pytest.raises(ValueError):
        _config(assignment_threshold=-0.1)


def test_env_vars_are_namespaced(monkeypatch):
    """Short names like max_questions must not collide with unrelated env vars."""
    monkeypatch.setenv("MAX_QUESTIONS", "3")
    assert _config().max_questions == 150

    monkeypatch.setenv("RECALL_COVERAGE_MAX_QUESTIONS", "7")
    assert _config().max_questions == 7


def test_default_prefix_map_keeps_claude_code_and_desktop_separate():
    prefix_map = _config().prefix_map()
    assert prefix_map["claude-code"] == ("claude_", "cc_")
    assert prefix_map["claude-desktop"] == ("claude_desktop_",)
    assert prefix_map["ui"] == ("search-ui-",)
    assert set(prefix_map) == set(DEFAULT_AGENT_PREFIX_MAP)

    # "api" and "all" are literals the backend never mints from a prefix.
    assert "api" not in prefix_map
    assert "all" not in prefix_map


def test_prefix_map_override_accepts_a_string_or_a_list():
    config = _config(
        agent_prefix_map=json.dumps({"claude-code": ["claude_", "cc_"], "codex": "codex_"})
    )
    assert config.prefix_map() == {"claude-code": ("claude_", "cc_"), "codex": ("codex_",)}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps(["claude-code"]),
        json.dumps({"claude-code": 5}),
        json.dumps({"claude-code": ["claude_", 5]}),
        json.dumps({"claude-code": []}),
        json.dumps({"claude-code": [""]}),
    ],
)
def test_malformed_prefix_map_fails_loudly(raw):
    """A misconfigured map must not degrade into "this agent asked nothing"."""
    with pytest.raises(ValueError):
        _config(agent_prefix_map=raw).prefix_map()


def test_query_types_are_recall_not_lookups():
    query_types = _config().query_types()
    assert query_types == [
        "GRAPH_COMPLETION",
        "GRAPH_COMPLETION_COT",
        "GRAPH_COMPLETION_CONTEXT_EXTENSION",
        "GRAPH_SUMMARY_COMPLETION",
        "TEMPORAL",
        "RAG_COMPLETION",
    ]
    for excluded in ("CHUNKS", "CHUNKS_LEXICAL", "SUMMARIES", "CYPHER", "CODING_RULES"):
        assert excluded not in query_types


def test_to_dict_is_json_serializable():
    dumped = _config().to_dict()
    assert json.loads(json.dumps(dumped))["agent_prefix_map"]["codex"] == ["codex_"]


def test_config_getter_is_cached_and_clearable(monkeypatch):
    clear_recall_coverage_config_cache()
    first = get_recall_coverage_config()
    assert get_recall_coverage_config() is first

    clear_recall_coverage_config_cache()
    monkeypatch.setattr(config_module, "RecallCoverageConfig", lambda: "sentinel")
    assert get_recall_coverage_config() == "sentinel"
    clear_recall_coverage_config_cache()


def test_coverage_params_snapshot_every_config_threshold():
    """Params carry the values, config owns the defaults — no duplicated numbers."""
    config = _config(max_questions=11, dedup_threshold=0.5)
    params = CoverageParams.from_config(config)

    assert params.max_questions == 11
    assert params.dedup_threshold == 0.5
    assert params.query_types == config.query_types()

    for name in CoverageParams.model_fields:
        if name == "query_types":
            continue
        assert getattr(params, name) == getattr(config, name), name


def test_coverage_params_apply_overrides_and_reject_typos():
    config = _config()
    params = CoverageParams.from_config(config, max_questions=3)
    assert params.max_questions == 3

    with pytest.raises(ValueError):
        CoverageParams.from_config(config, max_questsions=3)


def test_coverage_params_round_trip_through_json():
    """params is stored as JSON on the run row and read back later."""
    params = CoverageParams.from_config(_config())
    restored = CoverageParams(**json.loads(json.dumps(params.model_dump())))
    assert restored == params


def test_agent_scope_is_an_immutable_value_object():
    scope = AgentScope(label="claude-code", prefixes=("claude\\_", "cc\\_"))
    assert scope.mode is AgentScopeMode.PREFIX or scope.mode is AgentScopeMode.ALL
    with pytest.raises(FrozenInstanceError):
        scope.label = "codex"


def test_agent_scope_defaults_to_all():
    """Omitting agent_label means "how is my memory doing overall"."""
    scope = AgentScope(label="all")
    assert scope.mode is AgentScopeMode.ALL
    assert scope.prefixes == ()


def test_enum_wire_values():
    """These strings are written into String columns and read by the UI."""
    assert [status.value for status in RunStatus] == ["pending", "running", "complete", "failed"]
    assert [source.value for source in QuestionSource] == ["observed", "curated"]
    assert [status.value for status in SuggestionStatus] == ["pending", "accepted", "dismissed"]
    assert [scope.value for scope in CuratedScope] == ["agent", "shared"]
    assert SINK_TOPIC_ID == "other"


@pytest.mark.parametrize(
    "error_class,status_code",
    [
        (UnknownAgentLabelError, 404),
        (CoverageRunNotFoundError, 404),
        (CoverageTopicNotFoundError, 404),
        (CoverageSuggestionNotFoundError, 404),
        (CuratedQuestionNotFoundError, 404),
        (CoverageRunInFlightError, 409),
        (CoverageSuggestionNotPendingError, 409),
        (DuplicateCuratedQuestionError, 409),
        (EmbeddingFingerprintMismatchError, 409),
        (SinkTopicNotEditableError, 422),
        (InvalidCuratedQuestionScopeError, 422),
        (DegenerateEmbeddingError, 500),
    ],
)
def test_exception_status_codes(error_class, status_code):
    """Id-keyed lookups 404 on an owner mismatch — never 403, which leaks existence."""
    error = error_class()
    assert error.status_code == status_code
    assert error.args == (error.message, error.name)
