"""Guards on turning an ``agent_label`` into a session-id scope, and back.

Three invariants that silently corrupt a coverage report when broken:

* a prefix reaching a ``LIKE`` pattern unescaped, because ``_`` is a wildcard —
  ``claude_%`` would report Claude Desktop's traffic as Claude Code's;
* an unknown label resolving to "no prefixes" instead of 422ing, which is
  indistinguishable from a real agent that has asked nothing yet;
* the two directions disagreeing. ``resolve_agent_scope`` decides which rows a
  label selects and :func:`classify_session` decides which label a row reports,
  so a drift between them files a row under a label whose own predicate does not
  select it. ``test_the_classifier_agrees_with_the_sql_predicate`` pins that.

The SQL half of the escaping contract is exercised in
``cognee/tests/unit/modules/search/test_get_queries_window.py``.
"""

import pytest
from fastapi import status
from sqlalchemy import Column, MetaData, String, Table, create_engine, select

from cognee.modules.recall_coverage.agent_scope import (
    LABEL_ALL,
    LABEL_API,
    LABEL_UI,
    classify_session,
    escape_like_prefix,
    resolve_agent_scope,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import UnknownAgentLabelError
from cognee.modules.recall_coverage.types import AgentScopeMode
from cognee.modules.search.operations.get_queries import build_session_predicate


def _config(**overrides) -> RecallCoverageConfig:
    """A config that ignores the developer's ``.env``."""
    return RecallCoverageConfig(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("claude_", "claude\\_"),
        ("cc_", "cc\\_"),
        ("search-ui-", "search-ui-"),
        ("100%_", "100\\%\\_"),
        # The escape character itself is escaped first, so the backslashes this
        # function introduces are not doubled a second time.
        ("a\\b_", "a\\\\b\\_"),
    ],
)
def test_like_metacharacters_are_escaped(raw, expected):
    assert escape_like_prefix(raw) == expected


def test_default_map_prefixes_are_escaped_and_longest_first():
    """``claude_desktop_`` must be tried before ``claude_`` when classifying."""
    scope = resolve_agent_scope("claude-code", config=_config())

    assert scope.mode is AgentScopeMode.PREFIX
    assert scope.prefixes == ("claude\\_", "cc\\_")


def test_longer_prefixes_of_other_labels_are_subtracted():
    """Escaping cannot separate ``claude_`` from ``claude_desktop_``; exclusion can."""
    scope = resolve_agent_scope("claude-code", config=_config())

    assert scope.excluded_prefixes == ("claude\\_desktop\\_",)
    # Claude Desktop owns the longer prefix, so it has nothing to subtract.
    assert resolve_agent_scope("claude-desktop", config=_config()).excluded_prefixes == ()


def test_a_labels_own_longer_prefix_is_not_subtracted():
    """Otherwise a label owning both ``a_`` and ``a_b_`` would exclude half its own."""
    config = _config(agent_prefix_map='{"widget": ["w_", "w_pro_"], "other": ["o_"]}')

    scope = resolve_agent_scope("widget", config=config)

    assert scope.prefixes == ("w\\_pro\\_", "w\\_")
    assert scope.excluded_prefixes == ()


def test_omitted_label_defaults_to_all():
    for label in (None, "", "   "):
        scope = resolve_agent_scope(label, config=_config())

        assert scope.label == LABEL_ALL
        assert scope.mode is AgentScopeMode.ALL
        # "all" is the absence of a session predicate, so it carries no prefixes
        # to build one from.
        assert scope.prefixes == ()


def test_api_carries_every_known_prefix_to_negate():
    config = _config()
    scope = resolve_agent_scope(LABEL_API, config=config)

    assert scope.mode is AgentScopeMode.NEGATED
    expected = {
        escape_like_prefix(prefix)
        for prefixes in config.prefix_map().values()
        for prefix in prefixes
    }
    assert set(scope.prefixes) == expected
    # The UI's prefixes are among them: "api" is the complement of the *whole*
    # map, so a label that resolves through the map must also be negated by it,
    # or its traffic would be counted twice.
    assert "ui\\_" in scope.prefixes
    assert "search-ui-" in scope.prefixes
    # Longest raw prefix first, so a classifier walking these cannot let
    # "claude_" claim a "claude_desktop_" session.
    lengths = [len(prefix.replace("\\", "")) for prefix in scope.prefixes]
    assert lengths == sorted(lengths, reverse=True)


def test_unknown_label_is_a_422_not_an_empty_scope():
    """A typo must not be indistinguishable from "this agent asked nothing yet"."""
    with pytest.raises(UnknownAgentLabelError) as raised:
        resolve_agent_scope("claude-cod", config=_config())

    assert raised.value.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_deployment_prefix_map_overrides_the_default():
    config = _config(agent_prefix_map='{"widget": ["widget_", "wg_"]}')

    scope = resolve_agent_scope("widget", config=config)
    assert scope.prefixes == ("widget\\_", "wg\\_")

    # A label from the built-in default is no longer valid once a deployment
    # supplies its own map.
    with pytest.raises(UnknownAgentLabelError):
        resolve_agent_scope("claude-code", config=config)


# --- the reserved ``ui`` label ------------------------------------------------


def test_ui_is_a_prefix_label_not_a_second_complement():
    """``ui`` resolves through the ordinary map; a complement scope is ``api``'s job.

    Giving ``ui`` a NEGATED scope would give two labels one predicate, and the
    same traffic would be reported — and replayed, and judged — under both.
    """
    scope = resolve_agent_scope(LABEL_UI, config=_config())

    assert scope.label == LABEL_UI
    assert scope.mode is AgentScopeMode.PREFIX
    assert scope.prefixes == ("search-ui-", "ui\\_")


def test_ui_carries_the_shipped_frontend_prefix_and_the_reserved_one():
    """``search-ui-`` is what the frontend mints today; ``ui_`` is the reserved name."""
    assert _config().prefix_map()[LABEL_UI] == ("search-ui-", "ui_")


# --- one session id -> one label ---------------------------------------------


@pytest.mark.parametrize(
    "session_id, expected",
    [
        # Longest prefix wins, and it must agree with the SQL exclusion set.
        ("claude_desktop_a1", "claude-desktop"),
        ("claude_a1", "claude-code"),
        ("cc_a1", "claude-code"),
        ("codex_a1", "codex"),
        ("search-ui-1712345678901", LABEL_UI),
        ("ui_a1", LABEL_UI),
        # "_" is a literal here, not a wildcard, so this is nobody's cc_ session.
        ("ccx_a1", LABEL_API),
        ("weird_a1", LABEL_API),
        ("", LABEL_API),
        (None, LABEL_API),
    ],
)
def test_classify_session_walks_the_raw_prefixes_longest_first(session_id, expected):
    assert classify_session(session_id, _config()) == expected


def test_classify_session_never_returns_all():
    """``all`` is the absence of a classification, not one of its outcomes."""
    for session_id in (None, "", "claude_a1", "nothing_recognisable"):
        assert classify_session(session_id, _config()) != LABEL_ALL


def test_classify_session_follows_a_deployment_prefix_map():
    config = _config(agent_prefix_map='{"widget": ["widget_", "wg_"]}')

    assert classify_session("widget_7", config) == "widget"
    # ``claude_`` belongs to nobody under this map, so it is api by definition.
    assert classify_session("claude_a1", config) == LABEL_API


_stub_metadata = MetaData()
CLASSIFIED_SESSIONS = Table(
    "classified_sessions",
    _stub_metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=True),
)

# One session id per interesting shape, including the two traps: "claude_" must
# not claim "claude_desktop_...", and "cc_" must not claim "ccx...".
CROSS_CHECK_SESSIONS = {
    "claude_code": "claude_a1",
    "claude_code_alt": "cc_a1",
    "claude_desktop": "claude_desktop_a1",
    "codex": "codex_a1",
    "ui_shipped": "search-ui-1712345678901",
    "ui_reserved": "ui_a1",
    "wildcard_trap": "ccx_a1",
    "unknown_prefix": "weird_a1",
    "no_session": None,
}


def _selected_by(agent_label: str) -> set[str]:
    """Keys of ``CROSS_CHECK_SESSIONS`` the label's SQL predicate selects."""
    scope = resolve_agent_scope(agent_label, config=_config())
    predicate = build_session_predicate(
        CLASSIFIED_SESSIONS.c.session_id, scope.mode, scope.prefixes, scope.excluded_prefixes
    )

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            _stub_metadata.create_all(connection)
            connection.execute(
                CLASSIFIED_SESSIONS.insert(),
                [{"id": key, "session_id": value} for key, value in CROSS_CHECK_SESSIONS.items()],
            )
            statement = select(CLASSIFIED_SESSIONS.c.id)
            if predicate is not None:
                statement = statement.where(predicate)
            return {row.id for row in connection.execute(statement)}
    finally:
        engine.dispose()


def test_the_classifier_agrees_with_the_sql_predicate():
    """``classify_session(s) == L`` iff ``L``'s predicate selects ``s``.

    The invariant the two directions exist to share. Without it a row can be
    stored with ``agent_label = "claude-code"`` while a ``claude-code`` run's own
    window does not contain it — a report that contradicts itself, and one no
    single-direction test can see.
    """
    config = _config()
    labels = list(config.prefix_map()) + [LABEL_API]

    selected_by_label = {label: _selected_by(label) for label in labels}

    for key, session_id in CROSS_CHECK_SESSIONS.items():
        expected = classify_session(session_id, config)
        owners = [label for label, keys in selected_by_label.items() if key in keys]

        assert owners == [expected], (key, session_id, owners, expected)
