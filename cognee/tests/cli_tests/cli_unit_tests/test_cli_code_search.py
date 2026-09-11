"""CLI support for SearchType.CODE: --code-query, --diagram, --diagram-out."""

import argparse
import asyncio
import json
import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from cognee.cli.code_search import (
    build_code_query,
    first_diagram,
    infer_diagram_format,
    iter_code_results,
    print_code_results,
    write_diagram,
)
from cognee.cli.commands.recall_command import RecallCommand
from cognee.cli.commands.search_command import SearchCommand
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException

MERMAID = '---\ntitle: "architecture: shop"\n---\nflowchart LR\n    n0[["(root)"]]\n    n1[["inventory"]]\n    n0 -- "calls x2" --> n1\n'
DOT = 'digraph code_graph {\n    rankdir=LR;\n    n0 [label="(root)" shape=component];\n    n1 [label="inventory" shape=component];\n    n0 -> n1 [label="calls x2"];\n}\n'


def _code_result(diagram_format="mermaid", source=MERMAID):
    return {
        "operation": "architecture",
        "repos": ["shop"],
        "nodes": [{"id": "m0", "kind": "module", "name": "."}],
        "edges": [],
        "stats": {"nodes_total": 2, "nodes_shown": 2, "edges_shown": 1, "truncated": False},
        "diagram": {"format": diagram_format, "source": source, "nodes": 2, "edges": 1},
    }


def _search_response(result=None):
    """search() wraps the CODE result per dataset."""
    return [
        {
            "dataset_id": "11111111-1111-1111-1111-111111111111",
            "dataset_name": "shop",
            "search_result": [result or _code_result()],
        }
    ]


def _args(**overrides):
    base = dict(
        query_text="",
        query_type="CODE",
        datasets=["shop"],
        top_k=10,
        system_prompt=None,
        output_format="pretty",
        code_query=None,
        diagram=None,
        diagram_out=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _mock_run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --- option assembly -----------------------------------------------------------


def test_build_code_query_is_none_without_code_options():
    assert build_code_query(_args(), "CODE") is None
    assert build_code_query(_args(query_type="CHUNKS"), "CHUNKS") is None


def test_build_code_query_parses_json_and_applies_diagram_options():
    args = _args(code_query='{"operation": "impact_analysis", "name": "pkg.X"}', diagram="dot")
    assert build_code_query(args, "CODE") == {
        "operation": "impact_analysis",
        "name": "pkg.X",
        "diagram": "dot",
    }

    # --diagram-out alone implies a diagram, format inferred from the extension.
    assert build_code_query(_args(diagram_out="arch.html"), "CODE") == {"diagram": "mermaid"}
    assert build_code_query(_args(diagram_out="arch.svg"), "CODE") == {"diagram": "dot"}
    assert build_code_query(_args(diagram_out="arch.gv"), "CODE") == {"diagram": "dot"}
    # An explicit diagram in the JSON wins over the inferred one.
    args = _args(code_query='{"operation": "explore", "diagram": false}', diagram_out="x.html")
    assert build_code_query(args, "CODE") == {"operation": "explore", "diagram": False}


@pytest.mark.parametrize(
    "extension, expected",
    [(".mmd", "mermaid"), (".dot", "dot"), (".png", "dot"), (".html", "mermaid"), ("", "mermaid")],
)
def test_infer_diagram_format(extension, expected):
    assert infer_diagram_format(f"diagram{extension}") == expected


def test_build_code_query_rejects_non_code_types_and_bad_json():
    with pytest.raises(CliCommandInnerException, match="require --query-type CODE"):
        build_code_query(_args(query_type="CHUNKS", diagram="mermaid"), "CHUNKS")
    with pytest.raises(CliCommandInnerException, match="require --query-type CODE"):
        build_code_query(_args(query_type=None, code_query="{}"), None)
    with pytest.raises(CliCommandInnerException, match="not valid JSON"):
        build_code_query(_args(code_query="{not json"), "CODE")
    with pytest.raises(CliCommandInnerException, match="JSON object"):
        build_code_query(_args(code_query='["architecture"]'), "CODE")


# --- result discovery ----------------------------------------------------------


def test_iter_code_results_finds_results_in_search_and_recall_shapes():
    nested = _search_response()
    assert [result["operation"] for result in iter_code_results(nested)] == ["architecture"]
    # recall() items: a plain list of dicts, possibly mixed with other sources.
    recall_like = [{"_source": "session", "question": "q"}, _code_result()]
    assert len(list(iter_code_results(recall_like))) == 1
    # recall() items are pydantic models whose ``raw`` keeps the CODE payload.
    from pydantic import BaseModel

    class _Item(BaseModel):
        text: str
        raw: dict

    assert [r["operation"] for r in iter_code_results([_Item(text="x", raw=_code_result())])] == [
        "architecture"
    ]
    assert list(iter_code_results(["just text", 42, None])) == []
    assert first_diagram(nested)["format"] == "mermaid"
    assert first_diagram([{"operation": "delta", "diagram": {"source": None}}]) is None


def test_print_code_results_shows_payload_and_fenced_diagram(capsys):
    assert print_code_results(_search_response()) is True
    out = capsys.readouterr().out
    assert "architecture" in out
    assert '"repos"' in out
    assert "```mermaid" in out
    assert 'title: "architecture: shop"' in out
    assert "nodes_shown=2" in out
    # The diagram source is shown once, in the fence, not repeated in the JSON.
    assert out.count("flowchart LR") == 1

    assert print_code_results(["plain", "strings"]) is False


# --- writing diagrams ------------------------------------------------------------


def test_write_diagram_raw_and_html(tmp_path):
    raw_path = write_diagram(_search_response(), str(tmp_path / "out" / "arch.mmd"))
    assert open(raw_path, encoding="utf-8").read() == MERMAID

    html_path = write_diagram(_search_response(), str(tmp_path / "arch.html"))
    page = open(html_path, encoding="utf-8").read()
    assert page.startswith("<!doctype html>")
    assert "<title>architecture: shop</title>" in page
    assert 'class="mermaid"' in page
    assert "mermaid.min.js" in page
    assert "mermaid.initialize" in page
    # Source is HTML-escaped inside the page; the label quotes survive as entities.
    assert "n0[[&quot;(root)&quot;]]" in page

    with pytest.raises(CliCommandInnerException, match="Mermaid format"):
        write_diagram(_search_response(_code_result("dot", DOT)), str(tmp_path / "arch.html"))
    with pytest.raises(CliCommandInnerException, match="No diagram"):
        write_diagram(
            [{"operation": "delta", "diagram": {"source": None}}], str(tmp_path / "x.mmd")
        )


def test_write_diagram_renders_dot_with_graphviz(tmp_path, monkeypatch):
    import shutil

    dot_response = _search_response(_code_result("dot", DOT))

    raw = write_diagram(dot_response, str(tmp_path / "arch.dot"))
    assert open(raw, encoding="utf-8").read() == DOT

    with pytest.raises(CliCommandInnerException, match="DOT format"):
        write_diagram(_search_response(), str(tmp_path / "arch.svg"))

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(CliCommandInnerException, match="Graphviz"):
        write_diagram(dot_response, str(tmp_path / "arch.svg"))

    monkeypatch.undo()
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")
    svg = write_diagram(dot_response, str(tmp_path / "arch.svg"))
    assert open(svg, encoding="utf-8").read().lstrip().startswith("<?xml")


# --- commands ------------------------------------------------------------------------


def test_search_command_registers_code_options():
    parser = argparse.ArgumentParser()
    SearchCommand().configure_parser(parser)
    actions = {action.dest for action in parser._actions}
    assert {"code_query", "diagram", "diagram_out"} <= actions

    parsed = parser.parse_args(
        ["", "-t", "CODE", "--code-query", '{"operation": "architecture"}', "--diagram", "dot"]
    )
    assert parsed.code_query == '{"operation": "architecture"}'
    assert parsed.diagram == "dot"

    recall_parser = argparse.ArgumentParser()
    RecallCommand().configure_parser(recall_parser)
    assert {"code_query", "diagram", "diagram_out"} <= {a.dest for a in recall_parser._actions}


@patch(
    "cognee.cli.user_resolution.resolve_cli_user",
    new_callable=lambda: AsyncMock(return_value=MagicMock(id="user")),
)
@patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
def test_search_command_passes_code_query_prints_diagram_and_writes_file(
    _mock_run_patch, _mock_resolve, tmp_path, capsys
):
    mock_cognee = MagicMock()
    mock_cognee.search = AsyncMock(return_value=_search_response())
    out_path = tmp_path / "arch.html"

    with patch.dict(sys.modules, {"cognee": mock_cognee}):
        SearchCommand().execute(
            _args(code_query='{"operation": "architecture"}', diagram_out=str(out_path))
        )

    mock_cognee.search.assert_awaited_once_with(
        query_text="",
        query_type=ANY,
        user=ANY,
        datasets=["shop"],
        system_prompt_path="answer_simple_question.txt",
        top_k=10,
        session_id=None,
        code_query={"operation": "architecture", "diagram": "mermaid"},
    )
    assert mock_cognee.search.await_args.kwargs["query_type"].name == "CODE"
    out = capsys.readouterr().out
    assert "```mermaid" in out
    assert f"Diagram written to {out_path}" in out
    assert out_path.is_file()


@patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run)
def test_search_command_without_code_options_does_not_send_code_query(_mock_run_patch):
    mock_cognee = MagicMock()
    mock_cognee.search = AsyncMock(return_value=["r"])
    with (
        patch.dict(sys.modules, {"cognee": mock_cognee}),
        patch(
            "cognee.cli.user_resolution.resolve_cli_user",
            new_callable=lambda: AsyncMock(return_value=MagicMock()),
        ),
    ):
        SearchCommand().execute(_args(query_type="CHUNKS"))
    assert "code_query" not in mock_cognee.search.await_args.kwargs


def test_search_command_rejects_diagram_without_code_type():
    with pytest.raises(CliCommandException, match="require --query-type CODE"):
        SearchCommand().execute(_args(query_type="CHUNKS", diagram="mermaid"))


@patch("cognee.cli.commands.recall_command.asyncio.run", side_effect=_mock_run)
def test_recall_command_routes_code_query_through_the_code_scope(_mock_run_patch, capsys):
    mock_cognee = MagicMock()
    mock_cognee.recall = AsyncMock(return_value=[_code_result()])

    with patch.dict(sys.modules, {"cognee": mock_cognee}):
        RecallCommand().execute(
            argparse.Namespace(
                query_text="",
                query_type="CODE",
                datasets=["shop"],
                top_k=10,
                system_prompt=None,
                session_id=None,
                output_format="pretty",
                code_query='{"operation": "architecture"}',
                diagram="mermaid",
                diagram_out=None,
            )
        )

    kwargs = mock_cognee.recall.await_args.kwargs
    assert kwargs["code_query"] == {"operation": "architecture", "diagram": "mermaid"}
    assert kwargs["scope"] == ["code"]
    assert kwargs["query_type"].name == "CODE"
    assert "```mermaid" in capsys.readouterr().out


def test_json_output_keeps_the_diagram_inline(capsys):
    """--output-format json is for scripting: the diagram stays inside the payload."""
    with (
        patch("cognee.cli.commands.search_command.asyncio.run", side_effect=_mock_run),
        patch(
            "cognee.cli.user_resolution.resolve_cli_user",
            new_callable=lambda: AsyncMock(return_value=MagicMock()),
        ),
    ):
        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(return_value=_search_response())
        with patch.dict(sys.modules, {"cognee": mock_cognee}):
            SearchCommand().execute(_args(diagram="mermaid", output_format="json"))
    out = capsys.readouterr().out
    # The status line precedes the JSON document, which starts on its own line.
    payload = json.loads(out[out.index("\n[") + 1 :])
    assert payload[0]["search_result"][0]["diagram"]["format"] == "mermaid"
