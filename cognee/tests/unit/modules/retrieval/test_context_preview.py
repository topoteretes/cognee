"""Tests for the ``only_context`` prompt preview.

The preview exists to answer "what would the LLM have received?", so its contract is as
much about what it must *not* do — call an LLM, write to the session — as about the
strings it returns.
"""

import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval import context_preview as context_preview_module
from cognee.modules.retrieval.context_preview import (
    CONTEXT_FORMAT_CONTEXT,
    CONTEXT_FORMAT_PROMPT,
    CONTEXT_FORMATS,
    ContextPreview,
    build_context_preview,
    load_read_only_session_prompt,
    render_context_for_prompt,
)
from cognee.modules.retrieval.utils.completion import build_completion_prompts

# Resolve module objects explicitly and patch with patch.object. Dotted-string targets
# are unreliable here: `cognee.infrastructure.session.__init__` re-exports the *function*
# `get_session_manager`, shadowing the same-named submodule, so a string target resolves
# to the function or the module depending on which import ran first — passing in
# isolation and failing in a full-suite run. Same reason for LLMGateway, a module holding
# a class of the same name. (See the identical note in test_get_retriever_output.py.)
get_session_manager_module = importlib.import_module(
    "cognee.infrastructure.session.get_session_manager"
)
session_turn_module = importlib.import_module("cognee.infrastructure.session.session_turn")
session_context_builder_module = importlib.import_module(
    "cognee.infrastructure.session.session_context_builder"
)
llm_gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")


class _PromptRetriever:
    """A retriever with the prompt attributes the completion path reads."""

    user_prompt_path = "graph_context_for_question.txt"
    system_prompt_path = "answer_simple_question.txt"
    system_prompt = None
    session_id = "session-1"


class _NonGenerativeRetriever:
    """CHUNKS/SUMMARIES/CODE shape: no prompt templates at all."""

    session_id = "session-1"


class _FakeSessionManager:
    """Minimal session manager: records reads, refuses to be written to."""

    def __init__(self, *, available=True, auto_feedback=True, history="Previous conversation:\n"):
        self._available = available
        self._auto_feedback = auto_feedback
        self.history = history
        self.updated_entries = []
        self.added_qas = []

    def is_session_available_for_completion(self, user_id):
        return self._available

    def is_auto_feedback_enabled(self):
        return self._auto_feedback

    def resolve_session_id(self, session_id):
        return session_id or "default-session"

    async def update_session_context_entry(self, **kwargs):
        self.updated_entries.append(kwargs)

    async def add_qa(self, **kwargs):
        self.added_qas.append(kwargs)


@contextmanager
def patched_session(session_manager, *, block="", preference_lines=None, history=None):
    """Stub the read-only session primitives the preview imports lazily.

    Yields the ``build_active_context_block`` mock, the one collaborator tests assert on.
    """
    history_text = session_manager.history if history is None else history
    with (
        patch.object(
            get_session_manager_module,
            "get_session_manager",
            return_value=session_manager,
        ),
        patch.object(
            session_turn_module,
            "select_session_history",
            new_callable=AsyncMock,
            return_value=history_text,
        ),
        patch.object(
            session_turn_module,
            "load_preference_lines_safe",
            new_callable=AsyncMock,
            return_value=preference_lines or [],
        ),
        patch.object(
            session_context_builder_module,
            "build_active_context_block",
            new_callable=AsyncMock,
            return_value=(block, ["entry-1"]),
        ) as build_block,
    ):
        yield build_block


@pytest.fixture
def as_user():
    """Put a user on the context var the session lookup reads."""
    token = context_preview_module.session_user.set(SimpleNamespace(id="user-1"))
    yield
    context_preview_module.session_user.reset(token)


def test_context_formats_are_the_two_documented_values():
    assert CONTEXT_FORMATS == {CONTEXT_FORMAT_CONTEXT, CONTEXT_FORMAT_PROMPT}


def test_render_context_for_prompt_joins_lists_and_passes_strings_through():
    assert render_context_for_prompt("plain") == "plain"
    assert render_context_for_prompt(["a", "b"]) == "a\n---\nb"
    assert render_context_for_prompt(None) is None


@pytest.mark.asyncio
async def test_session_prompt_puts_guidance_ahead_of_history(as_user):
    manager = _FakeSessionManager(history="Previous conversation:\n\nQUESTION: q\nANSWER: a\n")
    with patched_session(manager, block="## Active session guidance\n- be terse"):
        prompt = await load_read_only_session_prompt(_PromptRetriever(), "why?")

    assert prompt.startswith("## Active session guidance")
    assert "QUESTION: q" in prompt
    assert prompt.index("be terse") < prompt.index("QUESTION: q")


@pytest.mark.asyncio
async def test_session_prompt_performs_no_session_writes(as_user):
    """The guidance block must be built with stamp_served=False."""
    manager = _FakeSessionManager()
    with patched_session(manager, block="## Active session guidance\n- x") as build_block:
        await load_read_only_session_prompt(_PromptRetriever(), "why?")

    assert build_block.await_args.kwargs["stamp_served"] is False
    assert manager.updated_entries == []
    assert manager.added_qas == []


@pytest.mark.asyncio
async def test_session_prompt_renders_preferences_when_auto_feedback_is_off(as_user):
    """Mirrors generate_session_answer: preferences still reach the prompt."""
    manager = _FakeSessionManager(auto_feedback=False)
    with patched_session(manager, preference_lines=["Answer in Portuguese"]) as build_block:
        prompt = await load_read_only_session_prompt(_PromptRetriever(), "why?")

    build_block.assert_not_awaited()
    assert "Answer in Portuguese" in prompt


@pytest.mark.asyncio
async def test_session_prompt_is_empty_without_an_available_session(as_user):
    manager = _FakeSessionManager(available=False)
    with patched_session(manager, block="## Active session guidance\n- x"):
        assert await load_read_only_session_prompt(_PromptRetriever(), "why?") == ""


@pytest.mark.asyncio
async def test_session_prompt_is_empty_without_a_user():
    assert await load_read_only_session_prompt(_PromptRetriever(), "why?") == ""


@pytest.mark.asyncio
async def test_session_prompt_fails_open(as_user):
    with patch.object(
        get_session_manager_module,
        "get_session_manager",
        side_effect=RuntimeError("cache down"),
    ):
        assert await load_read_only_session_prompt(_PromptRetriever(), "why?") == ""


@pytest.mark.asyncio
async def test_preview_matches_what_generate_completion_would_build(as_user):
    """The whole point: preview strings equal the real assembly, not a lookalike."""
    manager = _FakeSessionManager(history="Previous conversation:\n\nQUESTION: q\nANSWER: a\n")
    with patched_session(manager, block="## Active session guidance\n- be terse"):
        preview = await build_context_preview(
            _PromptRetriever(), query="why?", context="node1 -- rel -- node2"
        )

    expected_user, expected_system = build_completion_prompts(
        query="why?",
        context="node1 -- rel -- node2",
        user_prompt_path=_PromptRetriever.user_prompt_path,
        system_prompt_path=_PromptRetriever.system_prompt_path,
        system_prompt=None,
        conversation_history=preview.session_context,
    )
    assert preview.user_prompt == expected_user
    assert preview.system_prompt == expected_system
    # The ticket's specific complaint: the question is now in the output.
    assert "why?" in preview.user_prompt
    assert "node1 -- rel -- node2" in preview.user_prompt
    # And the session layer rides on the system prompt, as it does for a real answer.
    assert "be terse" in preview.system_prompt
    assert "\nTASK:" in preview.system_prompt


@pytest.mark.asyncio
async def test_preview_never_calls_an_llm(as_user):
    manager = _FakeSessionManager()
    with (
        patch.object(
            llm_gateway_module.LLMGateway,
            "acreate_structured_output",
            new_callable=MagicMock,
        ) as llm,
        patched_session(manager, block=""),
    ):
        await build_context_preview(_PromptRetriever(), query="why?", context="ctx")

    llm.assert_not_called()


@pytest.mark.asyncio
async def test_preview_omits_prompts_for_non_generative_retrievers(as_user):
    manager = _FakeSessionManager()
    with patched_session(manager, block="## Active session guidance\n- x"):
        preview = await build_context_preview(
            _NonGenerativeRetriever(), query="why?", context=["chunk-a", "chunk-b"]
        )

    assert preview.user_prompt is None
    assert preview.system_prompt is None
    # The session layer is still reported — it exists independently of a prompt template.
    assert "## Active session guidance" in preview.session_context


@pytest.mark.asyncio
async def test_preview_fails_open_when_rendering_raises(as_user):
    class _BadPromptRetriever(_PromptRetriever):
        user_prompt_path = "this_template_does_not_exist.txt"

    manager = _FakeSessionManager()
    with patched_session(manager, block="## Active session guidance\n- x"):
        preview = await build_context_preview(_BadPromptRetriever(), query="why?", context="ctx")

    assert isinstance(preview, ContextPreview)
    assert preview.user_prompt is None
    assert "## Active session guidance" in preview.session_context
