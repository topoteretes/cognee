"""Tests for the ``only_context`` prompt preview.

The preview exists to answer "what would the LLM have received?", so its contract is as
much about what it must *not* do — call an LLM, write to the session — as about the
strings it returns. To make the no-write assertions load-bearing, the real session
assembly (``build_session_prompt`` -> ``build_active_context_block``) runs against a fake
session manager that records every write; only the embedding-backed vector recall is
stubbed.
"""

import asyncio
import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from jinja2 import TemplateNotFound

from cognee.infrastructure.session.session_context_builder import build_active_context_block
from cognee.infrastructure.session.session_context_models import ContextSection
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.modules.retrieval import context_preview as context_preview_module
from cognee.modules.retrieval.context_preview import (
    ContextPreview,
    SharedSessionHistory,
    build_context_preview,
    load_read_only_session_prompt,
    render_context_for_prompt,
)
from cognee.modules.retrieval.utils.completion import build_completion_prompts

# Resolve module objects explicitly and patch with patch.object: package __init__ files
# re-export same-named functions/classes that shadow these submodules, so dotted-string
# patch targets resolve differently depending on import order.
session_turn_module = importlib.import_module("cognee.infrastructure.session.session_turn")
llm_gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")

GUIDANCE_LINE = "Answer in one short sentence"
PREVIOUS_QUESTION = "What is the capital of Germany?"


class _PromptRetriever:
    """A retriever with the prompt attributes the completion path reads."""

    user_prompt_path = "graph_context_for_question.txt"
    system_prompt_path = "answer_simple_question.txt"
    system_prompt = None
    session_id = "retriever-session"


class _NonGenerativeRetriever:
    """CHUNKS/SUMMARIES/CODE shape: no prompt templates, and no ``session_id`` attribute.

    ``ChunksRetriever.__init__`` does not keep the session id it is constructed with, so
    the preview has to take the caller's id explicitly rather than read it back here.
    """


class _OptedOutRetriever(_PromptRetriever):
    """Cypher/agentic shape: has the template attributes but never sends that prompt."""

    supports_prompt_preview = False


class _FakeSessionManager:
    """Enough of SessionManager for the real assembly to run; records every write."""

    session_history_last_n = 5

    def __init__(
        self, *, available=True, auto_feedback=True, qa_entries=None, context_entries=None
    ):
        self._available = available
        self._auto_feedback = auto_feedback
        self.qa_entries = (
            qa_entries
            if qa_entries is not None
            else [
                {
                    "qa_id": str(uuid4()),
                    "question": PREVIOUS_QUESTION,
                    "answer": "Berlin.",
                    "time": "2026-08-25T07:00:00+00:00",
                }
            ]
        )
        self.context_entries = (
            context_entries
            if context_entries is not None
            else [
                {
                    "id": str(uuid4()),
                    "section": ContextSection.PREFERENCES.value,
                    "context_profile": "qa",
                    "content": GUIDANCE_LINE,
                    "created_at": "2026-08-25T07:00:00+00:00",
                }
            ]
        )
        self.updated_entries = []
        self.added_qas = []
        self.seen_session_ids = []

    def is_session_available_for_completion(self, user_id):
        return self._available

    def is_auto_feedback_enabled(self):
        return self._auto_feedback

    def resolve_session_id(self, session_id):
        return session_id or "default-session"

    async def get_session(
        self, *, user_id, session_id, formatted=False, last_n=None, include_context=True
    ):
        self.seen_session_ids.append(session_id)
        entries = self.qa_entries[-last_n:] if last_n else self.qa_entries
        if formatted:
            return SessionManager.format_entries(entries, include_context=include_context)
        return entries

    async def get_session_entries_by_ids(self, *, user_id, session_id, qa_ids):
        return [entry for entry in self.qa_entries if entry["qa_id"] in qa_ids]

    @staticmethod
    def format_entries(entries, include_context=True):
        return SessionManager.format_entries(entries, include_context=include_context)

    async def get_session_context_entries(self, *, user_id, session_id):
        self.seen_session_ids.append(session_id)
        return self.context_entries

    async def update_session_context_entry(self, **kwargs):
        self.updated_entries.append(kwargs)

    async def add_qa(self, **kwargs):
        self.added_qas.append(kwargs)


@contextmanager
def patched_session(session_manager, *, caching=True, preference_lines=None, preference_text=""):
    """Stub only the boundaries: the manager lookup, config, and the two preference reads.

    The embedding-backed ``search_session_qa_ids`` is replaced (it is the session layer's
    one billed step); everything above it — history selection, guidance-block ranking and
    rendering, prompt composition — is the real code.
    """
    with (
        patch.object(context_preview_module, "get_session_manager", return_value=session_manager),
        patch.object(
            context_preview_module, "CacheConfig", return_value=SimpleNamespace(caching=caching)
        ),
        patch.object(
            context_preview_module,
            "load_preference_text",
            new_callable=AsyncMock,
            return_value=preference_text,
        ) as preference_text_mock,
        patch.object(
            session_turn_module,
            "search_session_qa_ids",
            new_callable=AsyncMock,
            return_value=[],
        ) as vector_recall,
        patch.object(
            session_turn_module,
            "load_preference_lines_safe",
            new_callable=AsyncMock,
            return_value=preference_lines or [],
        ),
    ):
        yield SimpleNamespace(vector_recall=vector_recall, preference_text=preference_text_mock)


@pytest.fixture
def as_user():
    """Put a user on the context var the session lookup reads."""
    token = context_preview_module.session_user.set(SimpleNamespace(id="user-1"))
    yield
    context_preview_module.session_user.reset(token)


def test_render_context_for_prompt_joins_lists_and_passes_strings_through():
    assert render_context_for_prompt("plain") == "plain"
    assert render_context_for_prompt(["a", "b"]) == "a\n---\nb"
    assert render_context_for_prompt(None) is None


# --- session layer -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_prompt_puts_guidance_ahead_of_history(as_user):
    manager = _FakeSessionManager()
    with patched_session(manager):
        prompt = await load_read_only_session_prompt("why?", session_id="s1")

    assert prompt.startswith("## Active session guidance")
    assert GUIDANCE_LINE in prompt
    assert f"QUESTION: {PREVIOUS_QUESTION}" in prompt
    assert prompt.index(GUIDANCE_LINE) < prompt.index("QUESTION:")


@pytest.mark.asyncio
async def test_session_prompt_performs_no_session_writes(as_user):
    """The real guidance-block builder runs; the preview must not stamp its entries."""
    manager = _FakeSessionManager()
    with patched_session(manager):
        prompt = await load_read_only_session_prompt("why?", session_id="s1")

    # The entry was rendered, so the stamping code path was reachable — and skipped.
    assert GUIDANCE_LINE in prompt
    assert manager.updated_entries == []
    assert manager.added_qas == []


@pytest.mark.asyncio
async def test_no_write_assertion_is_load_bearing(as_user):
    """Control: the same builder with its default DOES write, so the assertion above bites."""
    manager = _FakeSessionManager()
    with patched_session(manager):
        block, served_ids = await build_active_context_block(
            session_manager=manager, user_id="user-1", session_id="s1", query="why?"
        )

    assert GUIDANCE_LINE in block
    assert served_ids
    assert [entry["entry_id"] for entry in manager.updated_entries] == served_ids


@pytest.mark.asyncio
async def test_session_prompt_renders_preferences_when_auto_feedback_is_off(as_user):
    """Mirrors generate_session_answer: durable preferences still reach the prompt."""
    manager = _FakeSessionManager(auto_feedback=False)
    with patched_session(manager, preference_lines=["Answer in Portuguese"]):
        prompt = await load_read_only_session_prompt("why?", session_id="s1")

    assert "Answer in Portuguese" in prompt
    assert GUIDANCE_LINE not in prompt  # the stored-entry layer is off


@pytest.mark.asyncio
async def test_sessionless_caller_gets_the_preference_block(as_user):
    """CACHING=false: the real completion sends load_preference_text(); so does the preview."""
    manager = _FakeSessionManager()
    with patched_session(manager, caching=False, preference_text="PREFERENCES:\n- Portuguese") as m:
        prompt = await load_read_only_session_prompt("why?", session_id="s1")

    assert prompt == "PREFERENCES:\n- Portuguese"
    m.preference_text.assert_awaited_once()
    m.vector_recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_user_takes_the_sessionless_branch():
    manager = _FakeSessionManager()
    with patched_session(manager, preference_text="") as m:
        assert await load_read_only_session_prompt("why?", session_id="s1") == ""
    m.preference_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_unavailable_means_bare_prompt(as_user):
    """Caching on but the backend down: _run_session_turn sends no session layer at all."""
    manager = _FakeSessionManager(available=False)
    with patched_session(manager, preference_text="PREFERENCES:\n- x") as m:
        assert await load_read_only_session_prompt("why?", session_id="s1") == ""
    m.preference_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_prompt_fails_open(as_user):
    with (
        patch.object(
            context_preview_module, "CacheConfig", return_value=SimpleNamespace(caching=True)
        ),
        patch.object(
            context_preview_module, "get_session_manager", side_effect=RuntimeError("cache down")
        ),
    ):
        assert await load_read_only_session_prompt("why?", session_id="s1") == ""


@pytest.mark.asyncio
async def test_preview_prefers_the_callers_session_id_over_the_retrievers(as_user):
    manager = _FakeSessionManager()
    with patched_session(manager):
        await build_context_preview(
            _PromptRetriever(), query="why?", context="ctx", session_id="callers-session"
        )

    assert set(manager.seen_session_ids) == {"callers-session"}


@pytest.mark.asyncio
async def test_preview_falls_back_to_the_retrievers_session_id(as_user):
    manager = _FakeSessionManager()
    with patched_session(manager):
        await build_context_preview(_PromptRetriever(), query="why?", context="ctx")

    assert set(manager.seen_session_ids) == {_PromptRetriever.session_id}


@pytest.mark.asyncio
async def test_shared_history_reads_the_conversation_once_across_a_fan_out(as_user):
    """The embedding-backed recall is the one billed step; a fan-out must pay it once."""
    manager = _FakeSessionManager()
    shared = SharedSessionHistory(query="why?", session_id="s1")
    with patched_session(manager) as m:
        prompts = await asyncio.gather(
            *[
                load_read_only_session_prompt("why?", session_id="s1", shared_history=shared)
                for _ in range(3)
            ]
        )

    assert m.vector_recall.await_count == 1
    assert len(set(prompts)) == 1
    assert f"QUESTION: {PREVIOUS_QUESTION}" in prompts[0]


# --- prompt pair ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_matches_the_real_assembly_and_joins_list_context(as_user):
    """Preview strings equal the real assembly, including the list-context join it adds."""
    manager = _FakeSessionManager()
    with patched_session(manager):
        preview = await build_context_preview(
            _PromptRetriever(),
            query="why?",
            context=["node1 -- rel -- node2", "node2 -- rel -- node3"],
            session_id="s1",
        )

    # The transformation the preview layers on top of the shared helper is observable.
    assert "node1 -- rel -- node2\n---\nnode2 -- rel -- node3" in preview.user_prompt
    assert "['node1" not in preview.user_prompt  # never a Python repr of the list
    assert "why?" in preview.user_prompt

    expected_user, expected_system = build_completion_prompts(
        query="why?",
        context="node1 -- rel -- node2\n---\nnode2 -- rel -- node3",
        user_prompt_path=_PromptRetriever.user_prompt_path,
        system_prompt_path=_PromptRetriever.system_prompt_path,
        system_prompt=None,
        conversation_history=preview.session_context,
    )
    assert preview.user_prompt == expected_user
    assert preview.system_prompt == expected_system
    # The session layer rides on the system prompt, ahead of the TASK: join.
    assert preview.system_prompt.startswith(preview.session_context)
    assert GUIDANCE_LINE in preview.system_prompt
    assert "\nTASK:" in preview.system_prompt


@pytest.mark.asyncio
async def test_preview_never_calls_an_llm_and_embeds_exactly_once(as_user):
    manager = _FakeSessionManager()
    with (
        patch.object(
            llm_gateway_module.LLMGateway, "acreate_structured_output", new_callable=MagicMock
        ) as llm,
        patched_session(manager) as m,
    ):
        await build_context_preview(
            _PromptRetriever(), query="why?", context="ctx", session_id="s1"
        )

    llm.assert_not_called()
    assert m.vector_recall.await_count == 1


@pytest.mark.asyncio
async def test_preview_omits_prompts_for_non_generative_retrievers(as_user):
    """CHUNKS-shaped retriever: no prompt fields, but the *requested* session's layer."""
    manager = _FakeSessionManager()
    with patched_session(manager):
        preview = await build_context_preview(
            _NonGenerativeRetriever(),
            query="why?",
            context=["chunk-a", "chunk-b"],
            session_id="callers-session",
        )

    assert preview.user_prompt is None
    assert preview.system_prompt is None
    assert GUIDANCE_LINE in preview.session_context
    assert set(manager.seen_session_ids) == {"callers-session"}


@pytest.mark.asyncio
async def test_preview_honours_supports_prompt_preview_opt_out(as_user):
    """Cypher has the template attributes but never sends that prompt; report none."""
    manager = _FakeSessionManager()
    with patched_session(manager):
        preview = await build_context_preview(
            _OptedOutRetriever(), query="MATCH (n) RETURN n", context=None, session_id="s1"
        )

    assert preview.user_prompt is None
    assert preview.system_prompt is None
    assert GUIDANCE_LINE in preview.session_context


@pytest.mark.asyncio
async def test_missing_system_template_is_not_swallowed(as_user):
    class _BadSystemTemplate(_PromptRetriever):
        system_prompt_path = "this_template_does_not_exist.txt"

    manager = _FakeSessionManager()
    with patched_session(manager), pytest.raises(FileNotFoundError, match="this_template"):
        await build_context_preview(_BadSystemTemplate(), query="why?", context="ctx")


@pytest.mark.asyncio
async def test_missing_user_template_is_not_swallowed(as_user):
    class _BadUserTemplate(_PromptRetriever):
        user_prompt_path = "this_template_does_not_exist.txt"

    manager = _FakeSessionManager()
    with patched_session(manager), pytest.raises(TemplateNotFound):
        await build_context_preview(_BadUserTemplate(), query="why?", context="ctx")


def test_context_preview_defaults_are_the_no_prompt_sentinel():
    assert ContextPreview() == ContextPreview(
        session_context="", user_prompt=None, system_prompt=None
    )
