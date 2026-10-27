"""CODE_GRAPH_DOCS: the code routes store docstrings unless told otherwise."""

import pytest

from cognee.modules.cognify import config as cognify_config


def test_docstrings_are_on_by_default(monkeypatch):
    monkeypatch.delenv("CODE_GRAPH_DOCS", raising=False)
    assert cognify_config.CognifyConfig().code_graph_docs is True


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_the_setting_switches_them_off(monkeypatch, value):
    monkeypatch.setenv("CODE_GRAPH_DOCS", value)
    assert cognify_config.CognifyConfig().code_graph_docs is False


def test_the_argument_wins_over_the_setting():
    on = cognify_config.CognifyConfig(code_graph_docs=True)
    off = cognify_config.CognifyConfig(code_graph_docs=False)

    assert cognify_config.resolve_code_docs(None, on) is True
    assert cognify_config.resolve_code_docs(None, off) is False
    assert cognify_config.resolve_code_docs(False, on) is False
    assert cognify_config.resolve_code_docs(True, off) is True


def test_a_non_bool_argument_is_rejected():
    """cognify() must not take a string here: "false" would read as on."""
    with pytest.raises(ValueError, match="code_docs"):
        cognify_config.resolve_code_docs("false", cognify_config.CognifyConfig())


def test_the_setting_is_reported():
    config = cognify_config.CognifyConfig(code_graph_docs=False)
    assert config.to_dict()["code_graph_docs"] is False
