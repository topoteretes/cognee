"""Guards on turning an ``agent_label`` into a session-id scope.

Two invariants that silently corrupt a coverage report when broken:

* a prefix reaching a ``LIKE`` pattern unescaped, because ``_`` is a wildcard —
  ``claude_%`` would report Claude Desktop's traffic as Claude Code's;
* an unknown label resolving to "no prefixes" instead of 404ing, which is
  indistinguishable from a real agent that has asked nothing yet.

The SQL half of the escaping contract is exercised in
``cognee/tests/unit/modules/search/test_get_queries_window.py``.
"""

import pytest
from fastapi import status

from cognee.modules.recall_coverage.agent_scope import (
    LABEL_ALL,
    LABEL_API,
    escape_like_prefix,
    resolve_agent_scope,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import UnknownAgentLabelError
from cognee.modules.recall_coverage.types import AgentScopeMode


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
    # Longest raw prefix first, so a classifier walking these cannot let
    # "claude_" claim a "claude_desktop_" session.
    lengths = [len(prefix.replace("\\", "")) for prefix in scope.prefixes]
    assert lengths == sorted(lengths, reverse=True)


def test_unknown_label_is_a_404_not_an_empty_scope():
    with pytest.raises(UnknownAgentLabelError) as raised:
        resolve_agent_scope("claude-cod", config=_config())

    assert raised.value.status_code == status.HTTP_404_NOT_FOUND


def test_deployment_prefix_map_overrides_the_default():
    config = _config(agent_prefix_map='{"widget": ["widget_", "wg_"]}')

    scope = resolve_agent_scope("widget", config=config)
    assert scope.prefixes == ("widget\\_", "wg\\_")

    # A label from the built-in default is no longer valid once a deployment
    # supplies its own map.
    with pytest.raises(UnknownAgentLabelError):
        resolve_agent_scope("claude-code", config=config)
