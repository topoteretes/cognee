"""Crediting Cognee memory in the answer (memory_credit), as the Cursor and Codex plugins do.

Recalled items are tagged ``[mN · origin]`` with one numbering per conversation,
an instruction asks the agent to end an answer that used them with a plain
``> Cognee: …`` line, the ``transform_llm_output`` hook draws it as a purple
COGNEE wordmark in Hermes Desktop, and the credit is noted and stripped before
the turn is stored.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import memory_credit  # noqa: E402

GRAPH_CONTEXT = (
    "## Relevant passages\n"
    "Session ID: codex_abc\n\nQuestion: what editor?\nAnswer: Helix."
    "\n---\n"
    "# Session learning 2026-10-03 (session claude_x)\n\nUse the server, not the CLI."
    "\n\n## Relevant entities\n### mika (person)\n- Mika likes tennis."
    "\n\n## Related facts\n- Mika plays on Saturdays."
)


@pytest.fixture
def cite_on(monkeypatch):
    monkeypatch.setenv("COGNEE_PLUGIN_CITE_MEMORY", "on")


def test_each_item_gets_a_tag_and_its_origin():
    tags = memory_credit.MemoryTags()
    tags.begin_turn()
    out = tags.tag_graph(GRAPH_CONTEXT)
    assert "[m1 · Codex session]" in out
    assert "[m2 · Claude Code session learning, 2026-10-03]" in out
    assert "## Relevant entities [m3 · knowledge graph entities]" in out
    assert "## Related facts [m4 · knowledge graph facts]" in out
    assert tags.found == ["m1", "m2", "m3", "m4"]


def test_a_memory_recalled_again_keeps_its_tag():
    tags = memory_credit.MemoryTags()
    tags.begin_turn()
    tags.tag_graph(GRAPH_CONTEXT)
    tags.begin_turn()
    assert tags.found == []
    again = tags.label("facts", "knowledge graph facts", "- Mika plays on Saturdays.")
    new = tags.label("recall", "Cognee recall", "something new")
    assert again == "[m4 · knowledge graph facts]"
    assert new == "[m5 · Cognee recall]"


def test_check_separates_valid_invented_and_earlier_tags():
    tags = memory_credit.MemoryTags()
    tags.begin_turn()
    tags.label("recall", "x", "first")
    tags.begin_turn()
    tags.label("recall", "x", "second")
    result = tags.check("Done.\n> 🧠 Cognee memory [m2, m1, m9]: you said so.")
    assert result == {
        "credited": True,
        "cited": ["m2", "m1", "m9"],
        "valid": ["m2", "m1"],
        "invented": ["m9"],
        "earlier": ["m1"],
        "available": 1,
    }
    assert tags.credited is True


def test_code_that_looks_like_a_tag_is_not_a_citation():
    tags = memory_credit.MemoryTags()
    tags.begin_turn()
    tags.label("recall", "x", "first")
    assert tags.check("print(arr[m1])")["cited"] == []
    assert tags.credited is False


PLAIN = "Mika likes Yuji.\n\n> Cognee: Mika likes Yuji and volleyball, based on agent_sessions."
RENDERED = (
    "Mika likes Yuji.\n\n"
    f"{memory_credit.WORDMARK[:-1]}\\enspace$"
    " Mika likes Yuji and volleyball, based on agent_sessions."
)
OLD_BRAILLE = (
    "Mika likes Yuji.\n\n"
    f"> {memory_credit.LOGOTYPE[0]} Mika likes Yuji, based on agent_sessions.\\\n"
    f"> {memory_credit.LOGOTYPE[1]}"
)


def test_the_instruction_asks_for_a_plain_line_with_the_source():
    text = memory_credit.credit_instruction(False, 1, "agent_sessions")
    assert '"> Cognee:"' in text
    assert "> Cognee: you prefer the server over the CLI, based on dataset agent_sessions." in text
    assert 'Write "dataset agent_sessions" exactly so.' in text
    assert "Do not copy the example" in text
    assert "\u2800" not in text and not any("\u2800" <= c <= "\u28ff" for c in text)


def test_desktop_gets_the_pixel_logo(cite_on):
    logo = memory_credit.WORDMARK
    assert logo.startswith("$\\color{#BC9BFF}{") and logo.endswith("}$")
    assert logo.count("\\rule[") > 20
    assert memory_credit.render_credit(PLAIN, "desktop") == RENDERED


def test_desktop_sets_the_named_datasets_as_code(cite_on):
    answer = (
        "Maya wears EU 39.\n\n> Cognee: Maya's orders, based on dataset "
        "live_context_support_agent and dataset agent_sessions."
    )
    line = memory_credit.render_credit(answer, "desktop").splitlines()[-1]
    assert line.endswith(
        " Maya's orders, based on dataset `live_context_support_agent`"
        " and dataset `agent_sessions`."
    )
    assert memory_credit.has_credit(memory_credit.render_credit(answer, "desktop"))
    assert memory_credit.strip_credit_lines(memory_credit.render_credit(answer, "desktop")) == (
        "Maya wears EU 39."
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "based on Cognee memory.",
        "based on dataset `x`.",
        "a dataset is a set of documents, based on Cognee memory.",
    ],
)
def test_only_a_named_dataset_is_set_as_code(sentence):
    assert memory_credit.highlight_datasets(sentence) == sentence


def test_three_datasets_are_each_set_as_code():
    sentence = "based on dataset a, dataset b and dataset c."
    assert memory_credit.highlight_datasets(sentence) == (
        "based on dataset `a`, dataset `b` and dataset `c`."
    )


def test_the_pixel_logo_follows_the_grid():
    # One rule per vertical run of squares; a single-letter check keeps it readable.
    # E: a full left column, then squares in rows 0, 2 and 4 (raised 0.8, 0.4, 0).
    logo = memory_credit.pixel_logo("E")
    assert logo == (
        "$\\color{#BC9BFF}{\\rule[0em]{0.2em}{1em}"
        "\\rule[0.8em]{0.2em}{0.2em}\\kern-0.2em"
        "\\rule[0.4em]{0.2em}{0.2em}\\kern-0.2em"
        "\\rule[0em]{0.2em}{0.2em}}$"
    )


def test_the_earlier_bold_wordmark_is_still_stripped():
    old = "Mika likes Yuji.\n\n> $\\color{#9F7AEA}{\\textbf{COGNEE}}$ Mika likes Yuji, based on x."
    assert memory_credit.has_credit(old)
    assert memory_credit.strip_credit_lines(old) == "Mika likes Yuji."


@pytest.mark.parametrize("platform", ["telegram", "discord", ""])
def test_other_platforms_keep_the_plain_line_for_now(platform):
    assert memory_credit.render_credit(PLAIN, platform) is None


def test_nothing_to_draw_without_a_credit_line_or_with_credit_off(monkeypatch):
    assert memory_credit.render_credit("Mika likes Yuji.", "desktop") is None
    monkeypatch.setenv("COGNEE_PLUGIN_CITE_MEMORY", "off")
    assert memory_credit.render_credit(PLAIN, "desktop") is None


def test_every_credit_form_counts_and_is_stripped():
    for answer in (PLAIN, RENDERED, OLD_BRAILLE):
        assert memory_credit.has_credit(answer), answer
        assert memory_credit.strip_credit_lines(answer) == "Mika likes Yuji."


def test_a_bare_label_is_not_a_credit():
    assert not memory_credit.has_credit("> Cognee:")
    assert not memory_credit.has_credit("Cognee: is a memory layer")


def test_the_credit_marks_the_session_as_credited():
    tags = memory_credit.MemoryTags()
    tags.begin_turn()
    tags.label("recall", "x", "Mika likes Yuji.")
    assert tags.check(RENDERED)["credited"] is True
    assert tags.credited is True


def test_older_credit_lines_are_stripped_too():
    answer = "Mika likes tennis.\n\n> 🧠 Cognee memory [m3]: from a Codex session."
    assert memory_credit.strip_credit_lines(answer) == "Mika likes tennis."
    assert memory_credit.strip_credit_lines("x = arr[m1]") == "x = arr[m1]"


def test_the_plugin_registers_the_hook(cite_on):
    from cognee_integration_hermes import register, transform_llm_output

    registered = {}

    class Ctx:
        def register_memory_provider(self, provider):
            registered["provider"] = provider

        def register_hook(self, name, callback):
            registered[name] = callback

    register(Ctx())
    assert registered["transform_llm_output"] is transform_llm_output
    assert transform_llm_output(response_text=PLAIN, platform="desktop", session_id="s") == RENDERED


def test_the_first_credit_asks_how_it_connects():
    assert memory_credit.FIRST_CREDIT_ADDENDUM in memory_credit.credit_instruction(True, 2)
    assert memory_credit.FIRST_CREDIT_ADDENDUM not in memory_credit.credit_instruction(False, 1)
    assert "found 1 memory " in memory_credit.credit_instruction(False, 1)


def test_off_switch(monkeypatch):
    monkeypatch.setenv("COGNEE_PLUGIN_CITE_MEMORY", "off")
    assert memory_credit.credit_enabled() is False
    monkeypatch.delenv("COGNEE_PLUGIN_CITE_MEMORY")
    assert memory_credit.credit_enabled() is True  # on by default, as in Cursor


def test_auto_recall_carries_tags_and_the_instruction(cite_on):
    with fake_backend() as fake:
        fake.results["recall"] = [{"text": "Mika likes tennis.", "source": "graph"}]
        provider = make_provider()
        provider.queue_prefetch("what does mika like")
        out = provider.prefetch("what does mika like")
    assert "[m1 · Cognee memory]" in out
    assert "Crediting Cognee memory: Cognee found 1 memory" in out
    assert memory_credit.FIRST_CREDIT_ADDENDUM in out


def test_nothing_recalled_means_no_instruction(cite_on):
    with fake_backend() as fake:
        fake.results["recall"] = []
        provider = make_provider()
        provider.queue_prefetch("q")
        assert provider.prefetch("q") == ""


def test_recall_tool_results_are_tagged(cite_on):
    with fake_backend() as fake:
        fake.results["recall"] = [{"text": "Mika likes tennis.", "source": "graph"}]
        provider = make_provider()
        out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "mika"}))
    assert out["results"][0]["tag"] == "[m1 · Cognee recall]"
    assert '"> Cognee:"' in out["credit"]


@pytest.mark.parametrize("pinned", [False, True])
def test_the_credit_names_the_dataset_only_when_recall_read_it(cite_on, pinned):
    # Auto recall may answer from any readable dataset, so naming the active
    # one would credit the wrong source.
    expected = "based on dataset hermes" if pinned else "based on Cognee memory"
    with fake_backend() as fake:
        fake.results["recall"] = [{"text": "Mika likes tennis.", "source": "graph"}]
        provider = make_provider()
        provider._recall_pinned = pinned
        provider.queue_prefetch("what does mika like")
        prompt = provider.prefetch("what does mika like")
        tool = json.loads(provider.handle_tool_call("cognee_recall", {"query": "mika"}))
    assert expected in prompt
    assert expected in tool["credit"]
    if not pinned:
        assert "based on dataset" not in prompt + tool["credit"]


def test_auto_recall_tags_and_credits_the_dataset_the_server_named(cite_on):
    # A routed result carries the dataset it came from (SearchResult.dataset_name):
    # each tag names it, and the credit asks for the datasets the used items' tags
    # name, not "Cognee memory" and not the conversation's own dataset.
    with fake_backend() as fake:
        fake.results["recall"] = [
            {"text": "Maya ordered boots.", "source": "graph", "dataset_name": "shop_support"},
            {"text": "Leo swapped sizes.", "source": "graph", "dataset_name": "returns"},
        ]
        provider = make_provider()
        provider.queue_prefetch("what did maya order")
        prompt = provider.prefetch("what did maya order")
        tool = json.loads(provider.handle_tool_call("cognee_recall", {"query": "maya"}))
    assert "· dataset shop_support]" in prompt
    assert "· dataset returns]" in prompt
    assert "dataset named in the tag of each item you used" in prompt
    assert "based on dataset shop_support" in prompt  # the example
    assert "dataset hermes" not in prompt  # the conversation's dataset is not credited
    assert [item["tag"].endswith("· dataset shop_support]") for item in tool["results"]] == [
        True,
        False,
    ]
    assert "dataset named in the tag of each item you used" in tool["credit"]
    # A credit line naming the datasets is still the answer's credit.
    answer = (
        "Maya ordered boots.\n\n> Maya's boots, based on dataset shop_support and dataset returns."
    )
    assert memory_credit.has_credit(answer)


def test_auto_recall_falls_back_to_cognee_memory_when_a_result_names_no_dataset(cite_on):
    with fake_backend() as fake:
        fake.results["recall"] = [
            {"text": "Maya ordered boots.", "source": "graph", "dataset_name": "shop_support"},
            {"text": "Leo swapped sizes.", "source": "graph"},
        ]
        provider = make_provider()
        provider.queue_prefetch("what did maya order")
        prompt = provider.prefetch("what did maya order")
    assert "based on Cognee memory" in prompt
    assert "dataset named in the tag" not in prompt


def test_the_stored_answer_has_no_credit_line(cite_on):
    with fake_backend() as fake:
        provider = make_provider()
        provider.sync_turn("what does mika like?", RENDERED)
        assert fake.wait("remember_session")
        stored = json.dumps(
            [fake.kwargs_for(name) for name in ("remember_session", "remember_entry")],
            ensure_ascii=False,
        )
    assert "Mika likes Yuji." in stored
    assert "COGNEE" not in stored and "Cognee:" not in stored


def test_the_plugin_root_registers_the_hook_too():
    # Directory installs (the catalog) load the plugin root's __init__.py, not the package.
    import importlib.util

    spec = importlib.util.spec_from_file_location("hermes_plugin_root", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registered = {}

    class Ctx:
        def register_memory_provider(self, provider):
            registered["provider"] = provider

        def register_hook(self, name, callback):
            registered[name] = callback

    module.register(Ctx())
    assert "provider" in registered and "transform_llm_output" in registered


def test_the_drawn_line_is_not_a_blockquote_and_leaves_space(cite_on):
    # Half an em after the logo, about two spaces.
    # Desktop draws a blockquote with a quotation mark, and markdown collapses spaces.
    line = memory_credit.render_credit(PLAIN, "desktop").splitlines()[-1]
    assert line.startswith("$\\color{#BC9BFF}{")
    assert "}\\enspace$ Mika likes" in line
    assert not memory_credit.has_credit("Cognee: is a memory layer")


def test_the_terminal_ui_gets_the_logo_appended(cite_on):
    # The TUI always streams and keeps the streamed text unless the final starts
    # with it, so a replaced credit line would show the answer twice.
    out = memory_credit.render_credit(PLAIN, "tui")
    top, bottom = memory_credit.LOGOTYPE
    # PLAIN names no "dataset <name>", so the logo signs alone.
    assert out == f"{PLAIN}\n\n{top}\n{bottom}"
    assert out.startswith(PLAIN)


def test_the_streaming_classic_cli_gets_the_logo_appended(cite_on, monkeypatch):
    # A streamed answer is already on screen; Hermes shows only an appended suffix
    # cleanly (a replaced answer would be printed a second time).
    monkeypatch.setattr(memory_credit, "_cli_streams", lambda: True)
    top, bottom = memory_credit.LOGOTYPE
    # The CLI prints the suffix as is under the answer's closing line: one newline.
    assert memory_credit.render_credit(PLAIN, "cli") == f"{PLAIN}\n{top}\n{bottom}"


def test_the_non_streaming_classic_cli_gets_the_line_replaced(cite_on, monkeypatch):
    monkeypatch.setattr(memory_credit, "_cli_streams", lambda: False)
    out = memory_credit.render_credit(PLAIN, "cli")
    top, bottom = memory_credit.LOGOTYPE
    assert out.endswith(
        f"{top}  Mika likes Yuji and volleyball, based on agent_sessions.  \n{bottom}"
    )
    assert "> " not in out


@pytest.mark.parametrize("streams", [True, False])
def test_a_braille_credit_counts_and_both_rows_are_stripped(cite_on, monkeypatch, streams):
    monkeypatch.setattr(memory_credit, "_cli_streams", lambda: streams)
    out = memory_credit.render_credit(PLAIN, "cli")
    assert memory_credit.has_credit(out)
    assert memory_credit.strip_credit_lines(out) == "Mika likes Yuji."
    # A logo row alone, or braille inside a sentence, is not a credit.
    assert not memory_credit.has_credit(memory_credit.LOGOTYPE[0])
    assert not memory_credit.has_credit(f"see {memory_credit.LOGOTYPE[0]} here")


def test_streaming_defaults_to_on_outside_hermes():
    # No hermes_cli to ask (tests, a bare install): assume the default, streaming.
    assert memory_credit._cli_streams() is True


SIGNED = "Mika likes Yuji.\n\n> Mika likes Yuji and volleyball, based on dataset agent_sessions."


@pytest.mark.parametrize("platform", ["tui", "cli"])
def test_the_terminal_instruction_asks_for_an_unlabelled_line(platform):
    text = memory_credit.credit_instruction(False, 1, "agent_sessions", platform)
    assert "> you prefer the server over the CLI, based on dataset agent_sessions." in text
    assert "> Cognee:" not in text
    assert "> Cognee:" not in memory_credit.tool_credit_instruction("agent_sessions", platform)
    # Desktop and messaging apps keep the labelled line.
    assert "> Cognee:" in memory_credit.credit_instruction(False, 1, "agent_sessions", "desktop")


def test_the_terminal_signs_an_unlabelled_line_with_the_logo(cite_on, monkeypatch):
    monkeypatch.setattr(memory_credit, "_cli_streams", lambda: True)
    top, bottom = memory_credit.LOGOTYPE
    for platform, gap in (("tui", "\n\n"), ("cli", "\n")):
        out = memory_credit.render_credit(SIGNED, platform)
        assert out == f"{SIGNED}{gap}{top}  based on agent_sessions dataset\n{bottom}"
        assert memory_credit.has_credit(out)
        assert memory_credit.strip_credit_lines(out) == "Mika likes Yuji."
    monkeypatch.setattr(memory_credit, "_cli_streams", lambda: False)
    out = memory_credit.render_credit(SIGNED, "cli")
    assert out.endswith(
        f"{top}  Mika likes Yuji and volleyball, based on dataset agent_sessions.  \n{bottom}"
    )


def test_an_ordinary_quote_is_not_signed(cite_on):
    assert memory_credit.render_credit("> To be or not to be.", "tui") is None
    assert not memory_credit.has_credit("> To be or not to be.")


def test_the_provider_shapes_the_credit_for_its_platform(cite_on):
    with fake_backend() as fake:
        fake.results["recall"] = [{"text": "Mika likes tennis.", "source": "graph"}]
        provider = make_provider()
        provider._platform = "tui"
        provider.queue_prefetch("what does mika like")
        out = provider.prefetch("what does mika like")
    assert "the Cognee logo is added under it" in out


@pytest.mark.parametrize(
    "line",
    [
        "> Mika's likes come from earlier sessions stored in dataset agent_sessions.",
        "> From dataset agent_sessions: Mika likes Yuji.",
        "> Mika likes Yuji, as remembered in Cognee memory.",
    ],
)
def test_the_signed_line_is_recognised_however_it_is_worded(cite_on, line):
    # Seen live: the agent wrote "stored in dataset …", not "based on dataset …".
    answer = f"Mika likes Yuji.\n\n{line}"
    top, bottom = memory_credit.LOGOTYPE
    out = memory_credit.render_credit(answer, "tui")
    assert out.startswith(f"{answer}\n\n{top}")
    assert out.endswith(f"\n{bottom}")
    assert memory_credit.has_credit(out)
    assert memory_credit.strip_credit_lines(out) == "Mika likes Yuji."


def test_a_quote_naming_a_dataset_mid_answer_is_left_alone(cite_on):
    answer = "> Load dataset agent_sessions first.\n\nThen run the import."
    assert memory_credit.render_credit(answer, "tui") is None
    assert not memory_credit.has_credit(answer)
    assert memory_credit.strip_credit_lines(answer) == answer


@pytest.mark.parametrize(
    ("line", "signed"),
    [
        (
            "> Alice's story, based on dataset alice_in_wonderland.",
            "  based on alice_in_wonderland dataset",
        ),
        (
            "> Maya's orders, based on dataset shop and dataset returns.",
            "  based on shop and returns datasets",
        ),
        ("> Notes, based on dataset a, dataset b and dataset c.", "  based on a, b and c datasets"),
        ("> Mika likes Yuji, as remembered in Cognee memory.", ""),
    ],
)
def test_the_terminal_logo_names_the_credited_datasets(cite_on, monkeypatch, line, signed):
    monkeypatch.setattr(memory_credit, "_cli_streams", lambda: True)
    top, bottom = memory_credit.LOGOTYPE
    answer = f"Answer.\n\n{line}"
    out = memory_credit.render_credit(answer, "cli")
    assert out == f"{answer}\n{top}{signed}\n{bottom}"
    assert memory_credit.has_credit(out)
    assert memory_credit.strip_credit_lines(out) == "Answer."
