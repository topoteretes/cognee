"""Unit tests for the coding unit-of-work extraction (DISTILLATION_MODE=coding, SDK-1335).

Fake session manager and fake LLM throughout: the tests cover the rendered trace lines,
the repeated-error grouping and its pitfall keys, the section layout and limits of the
extraction input, and the once-per-unit guarantee (watermark plus lock).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from cognee.infrastructure.session import coding_unit_extraction as cue
from cognee.infrastructure.session.agent_context_extraction import (
    TRACE_EXTRACTION_STATE_ID,
    TRACE_EXTRACTION_STATE_KIND,
)
from cognee.infrastructure.session.coding_unit_extraction import (
    FAILURE_CANDIDATE_CONFIDENCE,
    MAX_INPUT_CHARS,
    MAX_TRACE_INPUT_CHARS,
    UnitOfWork,
    build_extraction_inputs,
    build_failure_candidates,
    extract_coding_unit_context,
    group_repeated_errors,
    pitfall_key,
    relative_path,
    render_trace_input,
    render_trace_line,
    shell_program,
)
from cognee.infrastructure.session.session_context_models import (
    AgentContextExtraction,
    SessionContextEntry,
)

REPO = "/Users/dev/projects/acme"
_COUNTER = iter(range(1, 100_000))


def _trace(tool, status="success", params=None, error="", output=None, feedback=""):
    return SimpleNamespace(
        trace_id=f"t{next(_COUNTER)}",
        origin_function=tool,
        status=status,
        method_params=params or {},
        method_return_value=output,
        error_message=error,
        session_feedback=feedback,
    )


def _edit_failure(path=f"{REPO}/src/app.py"):
    return _trace(
        "Edit",
        status="error",
        params={"file_path": path, "old_string": "foo", "new_string": "foo"},
        error="No changes to make: old_string and new_string are exactly the same",
    )


def _lesson_entry(content, created_at, section="success_patterns"):
    return {
        "id": f"e-{content}",
        "kind": "context",
        "context_profile": "agent",
        "section": section,
        "content": content,
        "normalized_content": content.lower(),
        "confidence": 0.9,
        "created_at": created_at,
    }


class FakeSessionManager:
    """In-memory trace store, Q&A store and context-entry CRUD."""

    def __init__(self, traces=(), qa=None, store=None):
        self.traces = list(traces)
        self.qa = qa
        self.store = list(store or [])

    async def get_agent_trace_count(self, *, user_id, session_id):
        return len(self.traces)

    async def get_agent_trace_session(self, *, user_id, session_id, last_n=None):
        return list(self.traces[-last_n:]) if last_n else list(self.traces)

    async def get_session(self, *, user_id, session_id, last_n=None, formatted=False):
        return [self.qa] if self.qa is not None else []

    async def get_session_context_entries(self, *, user_id, session_id):
        return list(self.store)

    async def create_session_context_entry(self, *, user_id, session_id, entry_dump):
        self.store.append(dict(entry_dump))
        return True

    async def update_session_context_entry(self, *, user_id, session_id, entry_id, merge):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    def watermark(self):
        for row in self.store:
            if row.get("id") == TRACE_EXTRACTION_STATE_ID:
                return row.get("processed_trace_count")
        return None

    def lessons(self):
        return [row for row in self.store if row.get("kind") == "context"]


def _llm(monkeypatch, lessons=(), *, delay=0.0):
    """Fake structured-output call; records every input it was given."""
    calls = []

    async def fake(*, text_input, system_prompt, response_model):
        calls.append(text_input)
        if delay:
            await asyncio.sleep(delay)
        assert response_model is AgentContextExtraction
        return AgentContextExtraction(lessons=list(lessons))

    monkeypatch.setattr(cue.LLMGateway, "acreate_structured_output", fake)
    monkeypatch.setattr(cue, "llm_available", lambda: True)
    monkeypatch.setattr(cue, "read_query_prompt", lambda name: "coding prompt")
    return calls


# -- Rendering ----------------------------------------------------------------


class TestRenderTraceInput:
    def test_paths_are_repo_relative_and_edit_snippets_are_short(self):
        entry = _trace(
            "Edit",
            params={"file_path": f"{REPO}/src/app.py", "old_string": "a" * 90, "new_string": "b"},
        )
        rendered = render_trace_input(entry, REPO)
        assert rendered.startswith("src/app.py ")
        assert REPO not in rendered
        assert "a" * 61 not in rendered

    def test_home_directory_is_stripped_without_a_repo_root(self):
        assert relative_path("/home/alice/proj/x.py", "") == "proj/x.py"
        assert relative_path("relative/x.py", REPO) == "relative/x.py"

    def test_command_is_clipped_and_input_capped(self):
        entry = _trace("Bash", params={"command": "pytest " + "x" * 500})
        rendered = render_trace_input(entry, REPO)
        assert len(rendered) <= MAX_TRACE_INPUT_CHARS
        assert rendered.startswith("pytest xxx")

    def test_secrets_are_redacted_from_inputs_and_outputs(self):
        entry = _trace(
            "Bash",
            params={"command": "curl -H 'Authorization: Bearer abc.def.ghi' https://x"},
            output={"token": "eyJabc.eyJdef.sig"},
        )
        line = render_trace_line(entry, REPO)
        assert "abc.def.ghi" not in line
        assert "eyJabc" not in line
        assert line.startswith("- Bash curl")

    def test_line_has_tool_input_status_and_error(self):
        line = render_trace_line(_edit_failure(), REPO)
        assert line.startswith(
            "- Edit src/app.py 'foo' -> 'foo' [error] | error: No changes to make"
        )

    def test_shell_program_skips_setup_words(self):
        assert shell_program("cd /x && pytest -q") == "pytest"
        assert shell_program("FOO=1 sudo make all") == "make"
        assert shell_program("echo hi") == ""


# -- Repeated errors ----------------------------------------------------------


class TestRepeatedErrors:
    def test_six_edit_failures_fixed_by_a_script_make_one_group_and_one_candidate(self):
        traces = [_edit_failure() for _ in range(6)]
        traces.append(
            _trace(
                "Bash",
                params={"command": f"python - <<'EOF'\nopen('{REPO}/src/app.py','w')...\nEOF"},
            )
        )
        groups = group_repeated_errors(traces)
        assert len(groups) == 1
        assert len(groups[0].failures) == 6
        assert groups[0].fixed and groups[0].fix.origin_function == "Bash"
        assert pitfall_key(groups[0]) == "pitfall:Edit:no-changes-to-make:py"

        candidates = build_failure_candidates(groups, REPO)
        assert len(candidates) == 1
        source_id, candidate = candidates[0]
        assert source_id == traces[0].trace_id
        assert candidate.section == "failure_lessons"
        assert candidate.confidence == FAILURE_CANDIDATE_CONFIDENCE
        assert candidate.content.startswith("[state-key: pitfall:Edit:no-changes-to-make:py] ")
        assert "→ Bash python" in candidate.content
        # Deterministic: a second run over the same unit renders the same content.
        assert build_failure_candidates(groups, REPO)[0][1].content == candidate.content

    def test_bash_error_key_uses_the_program_as_qualifier(self):
        failing = {"command": "cat -x notes.txt"}
        traces = [
            _trace("Bash", status="error", params=failing, error="cat: illegal option -- x"),
            _trace("Bash", status="error", params=failing, error="cat: illegal option -- x\nusage"),
            _trace("Bash", params={"command": "cat notes.txt"}),
        ]
        (group,) = group_repeated_errors(traces)
        assert pitfall_key(group) == "pitfall:Bash:cat-illegal-option:cat"
        assert group.fixed

    def test_single_unfixed_failure_is_not_a_group(self):
        assert group_repeated_errors([_edit_failure()]) == []

    def test_single_failure_fixed_by_the_same_tool_is_a_group(self):
        path = f"{REPO}/src/app.py"
        traces = [_edit_failure(path), _trace("Edit", params={"file_path": path})]
        assert len(group_repeated_errors(traces)) == 1

    def test_a_read_of_the_file_is_not_a_fix(self):
        path = f"{REPO}/src/app.py"
        traces = [_edit_failure(path), _trace("Read", params={"file_path": path})]
        assert group_repeated_errors(traces) == []

    def test_unfixed_repeated_group_produces_no_candidate(self):
        groups = group_repeated_errors([_edit_failure(), _edit_failure()])
        assert len(groups) == 1 and not groups[0].fixed
        assert build_failure_candidates(groups, REPO) == []


# -- Extraction input -----------------------------------------------------------


def _unit(traces, request="Fix the bug", answer="Done.", repo_root=REPO):
    return UnitOfWork(request=request, final_answer=answer, traces=traces, repo_root=repo_root)


class TestExtractionInput:
    def test_sections_appear_in_order(self):
        traces = [_edit_failure(), _trace("Edit", params={"file_path": f"{REPO}/src/app.py"})]
        existing = [
            SessionContextEntry(
                id="e1",
                section="tool_rules",
                context_profile="agent",
                content="Known rule.",
                confidence=0.9,
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
        (text,) = build_extraction_inputs(
            _unit(traces),
            existing,
            group_repeated_errors(traces),
            memory_lessons=["Edit on identical strings fails → use a script."],
        )
        positions = [
            text.index("EXISTING LESSONS"),
            text.index("LESSONS ALREADY IN MEMORY"),
            text.index("TRACES (part 1 of 1, phase change)"),
            text.index("REPEATED ERRORS AND WHAT WORKED"),
            text.index("FINAL MESSAGE"),
        ]
        assert positions == sorted(positions)
        assert "User request: Fix the bug" in text
        assert "Agent's final answer: Done." in text
        assert "What worked: Edit src/app.py" in text

    def test_request_and_answer_are_capped(self):
        unit = _unit([_trace("Read")], request="r" * 5_000, answer="a" * 20_000)
        (text,) = build_extraction_inputs(unit, [], [])
        assert "r" * 1_200 not in text
        assert "a" * 6_000 not in text
        assert len(text) < 10_000

    def test_long_units_are_split_with_every_section_in_each_part(self):
        traces = [
            _trace("Bash", params={"command": "x" * 150}, output="o" * 600) for _ in range(300)
        ]
        inputs = build_extraction_inputs(_unit(traces), [], [], max_chars=60_000)
        assert len(inputs) > 1
        for index, text in enumerate(inputs, start=1):
            assert len(text) <= 60_000
            assert "TRACES (part 1 of 1, phase verify)" in text  # one part, many chunks
            assert "FINAL MESSAGE" in text
        del index
        rendered_lines = sum(text.count("\n- Bash") for text in inputs)
        assert rendered_lines == 300

    @pytest.mark.asyncio
    async def test_existing_lessons_show_the_40_most_recent(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            _lesson_entry(f"lesson {index:02d}", (start + timedelta(minutes=index)).isoformat())
            for index in range(41)
        ]
        manager = FakeSessionManager(store=list(reversed(rows)))  # storage order is not time order
        shown = await cue.recent_agent_lessons(manager, "u", "s")
        assert len(shown) == 40
        assert shown[0].content == "lesson 01"
        assert shown[-1].content == "lesson 40"


# -- Once per unit ------------------------------------------------------------


def _unit_manager(trace_count=3):
    traces = [_trace("Read", params={"file_path": f"{REPO}/a{i}.py"}) for i in range(trace_count)]
    qa = SimpleNamespace(
        question="Add tests",
        answer="Added them.",
        context=f'{{"cwd": "{REPO}", "date": "2026-11-03"}}',
    )
    return FakeSessionManager(traces=traces, qa=qa)


def _candidate(content):
    return {"section": "success_patterns", "content": content, "confidence": 0.9}


@pytest.mark.asyncio
async def test_extracts_the_unit_once_and_seals_the_watermark(monkeypatch):
    manager = _unit_manager()
    calls = _llm(monkeypatch, [_candidate("Tests live in tests/.")])

    touched = await extract_coding_unit_context(
        session_manager=manager, user_id="u", session_id="s"
    )

    assert len(touched) == 1
    assert len(calls) == 1
    assert "TRACES (part 1 of 1, phase explore)" in calls[0]
    assert "LESSONS ALREADY IN MEMORY" not in calls[0]  # no dataset given
    assert "a0.py" in calls[0] and REPO not in calls[0]
    assert "User request: Add tests" in calls[0]
    assert manager.watermark() == 3
    assert [row["content"] for row in manager.lessons()] == ["Tests live in tests/."]


@pytest.mark.asyncio
async def test_second_sequential_trigger_makes_no_llm_call(monkeypatch):
    manager = _unit_manager()
    calls = _llm(monkeypatch, [_candidate("Tests live in tests/.")])

    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")
    again = await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")

    assert again == []
    assert len(calls) == 1
    assert len(manager.lessons()) == 1


@pytest.mark.asyncio
async def test_concurrent_triggers_extract_once(monkeypatch):
    manager = _unit_manager()
    calls = _llm(monkeypatch, [_candidate("Tests live in tests/.")], delay=0.01)

    results = await asyncio.gather(
        extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s"),
        extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s"),
    )

    assert sorted(len(result) for result in results) == [0, 1]
    assert len(calls) == 1
    assert len(manager.lessons()) == 1


@pytest.mark.asyncio
async def test_new_traces_after_the_watermark_form_the_next_unit(monkeypatch):
    manager = _unit_manager(trace_count=2)
    calls = _llm(monkeypatch)
    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")
    manager.traces.append(_trace("Read", params={"file_path": f"{REPO}/later.py"}))

    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")

    assert len(calls) == 2
    assert "later.py" in calls[1] and "a0.py" not in calls[1]
    assert manager.watermark() == 3


@pytest.mark.asyncio
async def test_failure_candidates_are_stored_once_across_runs(monkeypatch):
    path = f"{REPO}/src/app.py"
    traces = [_edit_failure(path) for _ in range(6)]
    traces.append(_trace("Bash", params={"command": f"python rewrite.py {path}"}))
    manager = FakeSessionManager(
        traces=traces, qa=SimpleNamespace(question="q", answer="a", context="")
    )
    _llm(monkeypatch)

    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")
    manager.traces.append(_trace("Read", params={"file_path": path}))
    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")

    pitfalls = [row for row in manager.lessons() if row["section"] == "failure_lessons"]
    assert len(pitfalls) == 1
    assert pitfalls[0]["content"].startswith("[state-key: pitfall:Edit:no-changes-to-make:py]")
    assert pitfalls[0]["source_trace_ids"] == [traces[0].trace_id]


@pytest.mark.asyncio
async def test_llm_error_leaves_the_watermark_untouched(monkeypatch):
    manager = _unit_manager()
    _llm(monkeypatch)

    async def boom(*, text_input, system_prompt, response_model):
        raise RuntimeError("llm down")

    monkeypatch.setattr(cue.LLMGateway, "acreate_structured_output", boom)
    with pytest.raises(RuntimeError):
        await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")
    assert manager.watermark() is None


@pytest.mark.asyncio
async def test_no_usable_llm_extracts_nothing(monkeypatch):
    manager = _unit_manager()
    calls = _llm(monkeypatch)
    monkeypatch.setattr(cue, "llm_available", lambda: False)

    assert (
        await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")
        == []
    )
    assert calls == [] and manager.watermark() is None


@pytest.mark.asyncio
async def test_caps_a_unit_at_its_newest_400_traces(monkeypatch):
    manager = _unit_manager(trace_count=450)
    calls = _llm(monkeypatch)

    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")

    assert manager.watermark() == 450
    assert "a49.py" not in "".join(calls) and "a50.py" in "".join(calls)


def test_state_row_names_match_the_shared_trace_watermark():
    assert TRACE_EXTRACTION_STATE_KIND == "agent_context_extraction_state"
    assert cue.TRACE_EXTRACTION_WATERMARK.state_id == TRACE_EXTRACTION_STATE_ID


# -- Phase parts and memory lessons in the extraction -----------------------------


@pytest.mark.asyncio
async def test_phase_parts_become_separate_inputs_in_order(monkeypatch):
    traces = [_trace("Read", params={"file_path": f"{REPO}/a.py"}) for _ in range(6)]
    traces += [_trace("Edit", params={"file_path": f"{REPO}/a.py"}) for _ in range(6)]
    traces += [_trace("Bash", params={"command": "pytest -q"}) for _ in range(6)]
    manager = FakeSessionManager(
        traces=traces, qa=SimpleNamespace(question="q", answer="a", context="")
    )
    calls = _llm(monkeypatch)

    await extract_coding_unit_context(session_manager=manager, user_id="u", session_id="s")

    headers = [line for text in calls for line in text.splitlines() if line.startswith("TRACES")]
    assert headers == [
        "TRACES (part 1 of 3, phase explore):",
        "TRACES (part 2 of 3, phase change):",
        "TRACES (part 3 of 3, phase verify):",
    ]
    assert calls[1].count("\n- Edit") == 6 and "- Read" not in calls[1].split("TRACES")[1]


@pytest.mark.asyncio
async def test_memory_lessons_are_searched_per_part_and_rendered_second(monkeypatch):
    dataset = SimpleNamespace(id="d", owner_id="o")
    queries = []

    async def fake_search(dataset_arg, session_id, query):
        queries.append(query)
        return ["Stored lesson about a.py."]

    monkeypatch.setattr(cue, "_search_memory_lessons", fake_search)
    traces = [_trace("Read", params={"file_path": f"{REPO}/a.py"}) for _ in range(3)]
    manager = FakeSessionManager(
        traces=traces,
        qa=SimpleNamespace(question="q", answer="a", context=f'{{"cwd": "{REPO}"}}'),
    )
    calls = _llm(monkeypatch)

    await extract_coding_unit_context(
        session_manager=manager, user_id="u", session_id="s", dataset=dataset
    )

    assert queries == ["a.py"]
    text = calls[0]
    assert text.index("LESSONS ALREADY IN MEMORY") < text.index("TRACES (part")
    assert "- Stored lesson about a.py." in text


@pytest.mark.asyncio
async def test_memory_search_timeout_or_failure_leaves_the_section_out(monkeypatch):
    dataset = SimpleNamespace(id="d", owner_id="o")

    async def slow(dataset_arg, session_id, query):
        await asyncio.sleep(1)
        return ["late"]

    monkeypatch.setattr(cue, "_search_memory_lessons", slow)
    monkeypatch.setattr(cue, "MEMORY_SEARCH_TIMEOUT_SECONDS", 0.01)
    traces = [_trace("Read", params={"file_path": f"{REPO}/a.py"})]
    manager = FakeSessionManager(
        traces=traces, qa=SimpleNamespace(question="q", answer="a", context="")
    )
    calls = _llm(monkeypatch)

    await extract_coding_unit_context(
        session_manager=manager, user_id="u", session_id="s", dataset=dataset
    )
    assert "LESSONS ALREADY IN MEMORY" not in calls[0]
    assert manager.watermark() == 1  # the extraction itself still ran

    async def broken(dataset_arg, session_id, query):
        raise RuntimeError("vector store down")

    monkeypatch.setattr(cue, "_search_memory_lessons", broken)
    manager.traces.append(_trace("Read", params={"file_path": f"{REPO}/b.py"}))
    await extract_coding_unit_context(
        session_manager=manager, user_id="u", session_id="s", dataset=dataset
    )
    assert "LESSONS ALREADY IN MEMORY" not in calls[1]


def test_memory_search_query_names_paths_errors_commands_and_patterns():
    traces = [
        _trace("Read", params={"file_path": f"{REPO}/src/a.py"}),
        _trace(
            "Bash", status="error", params={"command": "pytest -q"}, error="E   assert 1 == 2\nmore"
        ),
        _trace("Grep", params={"pattern": "def parse"}),
        _trace("Read", params={"file_path": f"{REPO}/src/a.py"}),  # duplicate
    ]
    query = cue.memory_search_query(traces, REPO)
    assert query.splitlines() == ["src/a.py", "e assert 1 == 2", "pytest -q", "def parse"]


def test_memory_lessons_session_first_deduplicated_and_capped():
    session = ["Lesson A", "Lesson B"]
    dataset = ["lesson a", "Lesson C"] + [f"Lesson {index}" for index in range(10)]
    chosen = cue.select_memory_lessons(session, dataset)
    assert chosen[:3] == ["Lesson A", "Lesson B", "Lesson C"]
    assert len(chosen) == cue.MEMORY_LESSONS_SHOWN

    long = ["x" * 1_000, "y" * 600, "z" * 400]
    assert cue.select_memory_lessons(long, []) == ["x" * 1_000, "z" * 400]  # 1,500 chars in all
