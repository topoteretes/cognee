"""Unit tests for the coding distillation path (DISTILLATION_MODE=coding, SDK-1335).

Covers the mode switch on prompts and models, the pinned turns in curator batches, the
writer input (state key, recased glossary, STYLE line), the evidence rules, the typed
lesson document, the pitfall session count and the State-Key replacement on publish.
The default path is asserted unchanged where the two diverge.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.session_context_models import SessionContextEntry
from cognee.modules.session_distillation import distill as distill_module
from cognee.modules.session_distillation import lesson_store
from cognee.modules.session_distillation.distill import (
    CURATOR_PROMPT_FILE,
    WRITER_PROMPT_FILE,
    apply_evidence_rules,
    build_curator_batches,
    build_writer_input,
    candidate_state_key,
    curate_batch,
    distillation_prompt_file,
    keep_state_key_update,
    lesson_metadata,
    newest_per_state_key,
    propose_lessons,
    publish_distilled_lessons,
    recase_glossary,
    write_or_reject,
)
from cognee.modules.session_distillation.lesson_store import (
    lessons_in_scope,
    mentioned_files,
    pitfall_sessions,
    project_node_sets,
    render_coding_lesson_document,
    rows_to_delete,
)
from cognee.modules.session_distillation.models import (
    CODING_WRITER_STYLE_LINE,
    CodingCuratorBatchOutput,
    CodingProposedLesson,
    CodingWrittenLesson,
    CuratorBatchOutput,
    WrittenLesson,
)

USER = SimpleNamespace(id=uuid4())
DATASET = SimpleNamespace(id=uuid4(), name="repo", owner_id=USER.id)


def _entry(
    content, section="success_patterns", source_trace_ids=(), entry_id=None, source_qa_ids=()
):
    return SessionContextEntry(
        id=entry_id or f"e-{uuid4().hex[:6]}",
        section=section,
        context_profile="agent",
        content=content,
        confidence=0.9,
        created_at="2026-11-03T10:00:00+00:00",
        source_trace_ids=list(source_trace_ids),
        source_qa_ids=list(source_qa_ids),
    )


def _written(
    lesson_type="other",
    state_key="",
    statement="MarkTaskCommand lives in cli/tasks.py.",
    context=None,
):
    lesson = CodingWrittenLesson(
        accept=True, statement=statement, lesson_type=lesson_type, state_key=state_key
    )
    return lesson.with_turn_context(context or {})


def _row(qa_id, question, time, answer="Done.", context=""):
    return {
        "qa_id": qa_id,
        "question": question,
        "answer": answer,
        "time": time,
        "context": context,
    }


# -- Mode switch ------------------------------------------------------------------


def test_prompt_files_follow_the_mode_and_the_coding_copies_exist():
    for name in (CURATOR_PROMPT_FILE, WRITER_PROMPT_FILE, "agent_context_extraction_system.txt"):
        assert distillation_prompt_file(name, False) == name
        coding = distillation_prompt_file(name, True)
        assert coding == f"coding/{name}"
        assert read_query_prompt(coding)
        assert read_query_prompt(coding) != read_query_prompt(name)


@pytest.mark.asyncio
async def test_curator_and_writer_use_the_coding_models_only_in_coding_mode(monkeypatch):
    seen = []

    async def fake(*, text_input, system_prompt, response_model):
        seen.append((system_prompt, response_model))
        return (
            response_model(lessons=[])
            if "lessons" in response_model.model_fields
            else (response_model(accept=False, reason="not_durable"))
        )

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", fake)
    lesson = CodingProposedLesson(working_statement="x")

    await curate_batch("batch")
    await curate_batch("batch", coding=True)
    await write_or_reject(lesson, [], [], [])
    await write_or_reject(lesson, [], [], [], coding=True)

    assert [model for _prompt, model in seen] == [
        CuratorBatchOutput,
        CodingCuratorBatchOutput,
        WrittenLesson,
        CodingWrittenLesson,
    ]
    assert seen[0][0] != seen[1][0] and seen[2][0] != seen[3][0]


def test_base_models_did_not_gain_coding_fields():
    assert (
        "state_key"
        not in CuratorBatchOutput.model_json_schema()["$defs"]["ProposedLesson"]["properties"]
    )
    assert "lesson_type" not in WrittenLesson.model_fields


# -- Curator batches ----------------------------------------------------------------


def test_pinned_turns_open_every_batch_of_six_candidates():
    qa_rows = [{"question": "Add tests", "answer": "Added.", "time": "2026-11-03T09:00:00"}]
    entries = [_entry(f"candidate {index}", entry_id=f"c{index}") for index in range(13)]

    batches = build_curator_batches(qa_rows, entries, pin_turns=True)

    assert len(batches) == 3
    for batch in batches:
        assert batch.text.startswith("User: Add tests\nAssistant: Added.")
    assert [len(batch.entry_ids) for batch in batches] == [6, 6, 1]
    assert batches[0].entry_ids == [f"c{index}" for index in range(6)]


def test_each_batch_pins_only_the_turns_of_its_candidates_in_turn_order():
    qa_rows = [
        _row("q1", "First", "2026-11-03T08:00:00"),
        _row("q2", "Second", "2026-11-03T09:00:00"),
        _row("q3", "Third", "2026-11-03T10:00:00"),
    ]
    entries = [
        _entry(f"late {index}", entry_id=f"l{index}", source_qa_ids=["q3"]) for index in range(6)
    ]
    entries += [_entry("early", entry_id="e1", source_qa_ids=["q1"])]
    entries += [_entry("unlinked", entry_id="u1")]  # judged against the newest turn
    entries += [_entry("both", entry_id="b1", source_qa_ids=["q2", "q1"])]

    batches = build_curator_batches(qa_rows, entries, pin_turns=True)

    assert [batch.entry_ids for batch in batches] == [
        ["e1", "b1", "l0", "l1", "l2", "l3"],
        ["l4", "l5", "u1"],
    ]
    first, second = (batch.text for batch in batches)
    assert first.index("User: First") < first.index("User: Second") < first.index("User: Third")
    assert first.index("User: Third") < first.index("Candidate e1")
    assert "User: First" not in second and "User: Second" not in second
    assert second.startswith("User: Third")


def test_turn_rows_for_entries_are_the_linked_turns_or_the_newest():
    qa_rows = [
        _row("q1", "First", "2026-11-03T08:00:00"),
        _row("q2", "Second", "2026-11-03T09:00:00"),
    ]
    linked = [_entry("x", source_qa_ids=["q1"])]
    assert [row["qa_id"] for row in distill_module.rows_for_entries(qa_rows, linked)] == ["q1"]
    unlinked = [_entry("y"), _entry("z", source_qa_ids=["gone"])]
    assert [row["qa_id"] for row in distill_module.rows_for_entries(qa_rows, unlinked)] == ["q2"]
    assert distill_module.rows_for_entries(qa_rows, []) == []


def test_lesson_turn_context_is_the_newest_turn_of_its_members():
    qa_rows = [
        _row("q1", "First", "2026-11-03T08:00:00", context='{"end_commit": "one"}'),
        _row("q2", "Second", "2026-11-03T09:00:00", context='{"end_commit": "two"}'),
        _row("q3", "Third", "2026-11-03T10:00:00", context='{"end_commit": "three"}'),
    ]
    members = [_entry("a", source_qa_ids=["q1"]), _entry("b", source_qa_ids=["q2"])]
    assert distill_module.lesson_turn_context(members, qa_rows) == {"end_commit": "two"}
    assert distill_module.lesson_turn_context([], qa_rows) == {"end_commit": "three"}
    assert distill_module.lesson_turn_context(members, []) == {}


def test_a_capped_coding_run_keeps_every_pinned_turn():
    qa_rows = [_row("q1", "First", "2026-11-03T08:00:00")]
    entries = [_entry(f"c{index}", entry_id=f"c{index}") for index in range(8)]
    rows, chosen = distill_module.select_curator_inputs(qa_rows, entries, 1, pin_turns=True)
    assert rows == qa_rows and len(chosen) == 6


def test_default_batches_keep_the_turn_in_one_block():
    qa_rows = [{"question": "Add tests", "answer": "Added.", "time": "2026-11-03T09:00:00"}]
    entries = [_entry(f"candidate {index}") for index in range(7)]

    batches = build_curator_batches(qa_rows, entries)

    assert len(batches) == 2
    assert sum(batch.text.count("User: Add tests") for batch in batches) == 1


# -- Writer input -------------------------------------------------------------------


def test_recase_glossary_uses_the_sessions_own_spelling():
    glossary = ["marktaskcommand", "cli/tasks.py", "unknownname"]
    texts = ["MarkTaskCommand is registered in cli/tasks.py"]
    assert recase_glossary(glossary, texts) == ["MarkTaskCommand", "cli/tasks.py", "unknownname"]


def test_coding_writer_input_has_state_key_recased_glossary_and_style_line_last():
    lesson = CodingProposedLesson(working_statement="MarkTaskCommand parses flags.", state_key="")
    members = [_entry("[state-key: test-baseline] pytest -q: 12 passed")]
    text = build_writer_input(lesson, members, ["older"], ["marktaskcommand"], coding=True)

    assert "PROPOSED STATE KEY" not in text
    assert "- MarkTaskCommand" in text and "- marktaskcommand" not in text
    assert text.endswith(CODING_WRITER_STYLE_LINE)

    keyed = CodingProposedLesson(working_statement="Suite passes.", state_key="test-baseline")
    assert "PROPOSED STATE KEY:\ntest-baseline" in build_writer_input(
        keyed, [], [], [], coding=True
    )


def test_default_writer_input_is_unchanged():
    lesson = CodingProposedLesson(working_statement="x", state_key="test-baseline")
    text = build_writer_input(lesson, [], [], ["marktaskcommand"])
    assert "STYLE" not in text and "PROPOSED STATE KEY" not in text
    assert "- marktaskcommand" in text


# -- Evidence rules -----------------------------------------------------------------


class TestEvidenceRules:
    def test_repeated_error_member_makes_a_keyed_pitfall(self):
        member = _entry(
            "[state-key: pitfall:Edit:no-changes-to-make:py] Edit failed → Bash python",
            section="failure_lessons",
            source_trace_ids=["t1"],
        )
        lesson = apply_evidence_rules(_written("location"), [member])
        assert lesson.lesson_type == "pitfall"
        assert lesson.state_key == "pitfall:Edit:no-changes-to-make:py"

    def test_pitfall_without_a_failed_trace_becomes_a_change(self):
        lesson = apply_evidence_rules(_written("pitfall"), [_entry("plain candidate")])
        assert lesson.lesson_type == "change"
        assert lesson.state_key == ""

    def test_pitfall_backed_by_a_live_failure_candidate_stays_a_pitfall_without_key(self):
        member = _entry("Bash failed: exit 1", section="failure_lessons", source_trace_ids=["t9"])
        lesson = apply_evidence_rules(_written("pitfall"), [member])
        assert lesson.lesson_type == "pitfall" and lesson.state_key == ""

    def test_state_keeps_its_key_and_other_types_lose_it(self):
        assert (
            apply_evidence_rules(_written("state", "test-baseline"), []).state_key
            == "test-baseline"
        )
        assert apply_evidence_rules(_written("location", "footprint:x"), []).state_key == ""

    def test_echoed_state_key_prefix_is_stripped_from_the_statement(self):
        lesson = apply_evidence_rules(
            _written("state", "test-baseline", "[state-key: test-baseline] pytest -q passes."), []
        )
        assert lesson.statement == "pytest -q passes."

    def test_rejections_pass_through(self):
        rejected = CodingWrittenLesson(accept=False, reason="not_durable", lesson_type="pitfall")
        assert apply_evidence_rules(rejected, []) is rejected

    def test_already_known_never_rejects_a_state_key_update(self):
        proposed = CodingProposedLesson(
            working_statement="[state-key: test-baseline] pytest -q: 41 passed.",
            state_key="test-baseline",
        )
        known = CodingWrittenLesson(accept=False, reason="already_known")
        kept = keep_state_key_update(proposed, known, [])
        assert kept.accept and kept.lesson_type == "state"
        assert kept.state_key == "test-baseline"
        assert kept.statement == "pytest -q: 41 passed."

        plain = CodingProposedLesson(working_statement="Routes live in api/.")
        assert keep_state_key_update(plain, known, [_entry("plain candidate")]) is known
        unsupported = CodingWrittenLesson(accept=False, reason="unsupported")
        assert keep_state_key_update(proposed, unsupported, []) is unsupported

    def test_already_known_never_rejects_a_repeated_pitfall(self):
        # Session B hits the pitfall session A stored: the writer sees A's lesson and calls
        # it known. The pitfall must still be stored, or "seen in N sessions" never grows.
        member = _entry(
            "[state-key: pitfall:Edit:no-changes-to-make:py] Edit failed → Bash python",
            section="failure_lessons",
            source_trace_ids=["t1"],
        )
        proposed = CodingProposedLesson(
            working_statement="Edit on an unchanged string fails; rewrite with a script."
        )
        known = CodingWrittenLesson(accept=False, reason="already_known")
        kept = apply_evidence_rules(keep_state_key_update(proposed, known, [member]), [member])
        assert kept.accept and kept.lesson_type == "pitfall"
        assert kept.state_key == "pitfall:Edit:no-changes-to-make:py"
        assert kept.statement == "Edit on an unchanged string fails; rewrite with a script."

    def test_candidate_state_key_parsing(self):
        assert (
            candidate_state_key("[state-key: footprint:auth-flow] files...")
            == "footprint:auth-flow"
        )
        assert candidate_state_key("no key here") == ""


# -- Documents ------------------------------------------------------------------------


def test_coding_document_has_headers_title_and_pitfall_count():
    text = render_coding_lesson_document(
        "Edit on .py files fails on identical strings → rewrite with a script.",
        session_id="s-1",
        lesson_type="pitfall",
        state_key="pitfall:Edit:no-changes-to-make:py",
        seen_in_sessions=2,
    )
    assert text == (
        "Lesson-Type: pitfall\n"
        "State-Key: pitfall:Edit:no-changes-to-make:py\n"
        "# Session learning (session s-1)\n\n"
        "Edit on .py files fails on identical strings → rewrite with a script. (seen in 2 sessions)\n"
    )


def test_non_pitfall_documents_carry_no_count_and_no_key_without_one():
    text = render_coding_lesson_document(
        "Routes live in api/routes/.", session_id="s-1", lesson_type="location", seen_in_sessions=3
    )
    assert (
        text
        == "Lesson-Type: location\n# Session learning (session s-1)\n\nRoutes live in api/routes/.\n"
    )


def test_pitfall_sessions_are_the_distinct_sessions_that_stored_the_key():
    rows = [
        SimpleNamespace(system_metadata={"session_id": "s-1"}),  # a row before the list
        SimpleNamespace(system_metadata={"session_id": "s-2", "seen_sessions": ["s-1", "s-2"]}),
    ]
    assert pitfall_sessions([], "s-1") == ["s-1"]
    assert pitfall_sessions(rows, "s-2") == ["s-1", "s-2"]  # re-distilling s-2 stays at 2
    assert pitfall_sessions(rows, "s-1") == ["s-1", "s-2"]  # re-distilling s-1 too
    assert pitfall_sessions(rows, "s-3") == ["s-1", "s-2", "s-3"]


def test_project_scope_ignores_the_distillations_own_node_sets():
    stored = '["session_learnings", "session_learnings:s-1", "proj:acme"]'
    assert project_node_sets(stored) == ("proj:acme",)
    assert project_node_sets(["session_learnings:s-9", "session_learnings"]) == ()
    assert project_node_sets(None) == ()


def test_lessons_in_scope_keep_generated_lessons_of_the_same_project_only():
    same = SimpleNamespace(
        node_set='["session_learnings", "proj:acme"]',
        system_metadata={"lesson_type": "state", "state_key": "test-baseline"},
    )
    other_project = SimpleNamespace(
        node_set='["session_learnings", "proj:other"]',
        system_metadata={"lesson_type": "state", "state_key": "test-baseline"},
    )
    # What a user with write access can plant: the lesson keys in the free-form
    # external_metadata, dated in the future. It is not a lesson, so it is never
    # counted, never wins the key and never gets the real lesson deleted.
    users_document = SimpleNamespace(
        id=uuid4(),
        node_set='["session_learnings", "proj:acme"]',
        external_metadata={
            "lesson_type": "state",
            "state_key": "test-baseline",
            "session_date": "9999-12-31",
            "seen_sessions": ["s-x", "s-y"],
        },
    )
    assert pitfall_sessions([users_document], "s-1") == ["s-1"]
    rows = [same, other_project, users_document]
    assert lessons_in_scope(rows, ["session_learnings", "session_learnings:s-2", "proj:acme"]) == [
        same
    ]
    assert lessons_in_scope(rows, ["session_learnings"]) == []


def test_one_run_stores_one_lesson_per_state_key():
    first = _written("state", "test-baseline", "pytest -q: 41 passed.")
    change = _written("change", statement="Routes moved to api/.")
    second = _written("state", "test-baseline", "pytest -q: 42 passed.")
    pitfall = _written("pitfall", "pitfall:Bash:cat-illegal-option:cat", "cat has no -z.")

    kept = newest_per_state_key([first, change, second, pitfall])
    assert [lesson.statement for lesson in kept] == [
        "Routes moved to api/.",
        "pytest -q: 42 passed.",
        "cat has no -z.",
    ]
    assert newest_per_state_key([change, pitfall]) == [change, pitfall]


def test_rows_to_delete_lets_the_newest_session_date_win():
    new_id = uuid4()
    older = SimpleNamespace(id=uuid4(), system_metadata={"session_date": "2026-11-01"})
    newer = SimpleNamespace(id=uuid4(), system_metadata={"session_date": "2026-11-05"})
    stored = SimpleNamespace(id=new_id, system_metadata={"session_date": "2026-11-03"})

    assert rows_to_delete([older, stored], {new_id: "2026-11-03"}) == [older]
    # A later session already stored the key: this session's lesson is the one dropped.
    assert rows_to_delete([older, newer, stored], {new_id: "2026-11-03"}) == [older, stored]
    # Without a date on both sides the lesson just stored wins.
    undated = SimpleNamespace(id=uuid4(), system_metadata={"session_date": ""})
    assert rows_to_delete([undated, newer, stored], {new_id: ""}) == [undated, newer]


def test_mentioned_files_keep_original_casing():
    statement = "README.md and src/Parser.py are read by run_tests.sh (e.g. v1.2.3 is not a file)."
    assert mentioned_files(statement) == ["README.md", "src/Parser.py", "run_tests.sh"]


def test_lesson_metadata_reads_the_plugins_qa_context():
    scope = SimpleNamespace(session_id="s-1")
    qa_context = {"date": "2026-11-03", "end_commit": "abc123", "branch": "dev", "cwd": "/x"}
    metadata = lesson_metadata(_written("location"), scope, qa_context, None)
    assert metadata == {
        "lesson_type": "location",
        "state_key": "",
        "session_id": "s-1",
        "session_date": "2026-11-03",
        "git_commit": "abc123",
        "git_branch": "dev",
        "files": ["cli/tasks.py"],
        "project_node_sets": [],
    }
    bare = lesson_metadata(_written("pitfall", "pitfall:Bash:x:cat"), scope, {}, ["s-0", "s-1"])
    assert bare["session_date"] == "" and bare["git_commit"] == ""
    assert bare["seen_sessions"] == ["s-0", "s-1"] and bare["seen_in_sessions"] == 2


def test_lesson_metadata_names_the_project_scope_without_the_distillation_sets():
    scope = SimpleNamespace(session_id="s-1")
    node_set = ["session_learnings", "session_learnings:s-1", "project-acme-0123"]
    metadata = lesson_metadata(_written("state", "test-baseline"), scope, {}, None, node_set)
    assert metadata["project_node_sets"] == ["project-acme-0123"]


# -- Publish ---------------------------------------------------------------------------


@pytest.fixture
def publish_env(monkeypatch):
    """Fake add/cognify/node-set/lesson-store; returns what publish did."""
    calls = SimpleNamespace(added=None, deleted=[], existing=[], finds=[], cognify_error=None)

    async def fake_add(data, **kwargs):
        calls.added = data
        calls.add_kwargs = kwargs
        # The rows are in the dataset from here on, as the replacement will find them.
        for item in data if isinstance(data, list) else []:
            if hasattr(item, "data_id"):
                calls.existing.append(
                    SimpleNamespace(
                        id=item.data_id,
                        node_set='["session_learnings"]',
                        external_metadata=item.external_metadata,
                        system_metadata=item.system_metadata,
                    )
                )

    async def fake_cognify(**kwargs):
        if calls.cognify_error is not None:
            raise calls.cognify_error

    import cognee.api.v1.add as add_module
    import cognee.api.v1.cognify as cognify_module

    monkeypatch.setattr(add_module, "add", fake_add)
    monkeypatch.setattr(cognify_module, "cognify", fake_cognify)
    monkeypatch.setattr(
        distill_module, "_lesson_node_set", AsyncMock(return_value=["session_learnings"])
    )

    async def fake_find(dataset_id, state_key, node_set):
        calls.finds.append(node_set)
        return [
            row
            for row in calls.existing
            if row.system_metadata["state_key"] == state_key and row.id not in calls.deleted
        ]

    async def fake_delete(*, dataset_id, data_id, user, delete_dataset_if_empty):
        calls.deleted.append(data_id)
        return {"status": "success"}

    monkeypatch.setattr(distill_module, "find_lessons_by_state_key", fake_find)
    monkeypatch.setattr(lesson_store, "find_lessons_by_state_key", fake_find)
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    monkeypatch.setattr(datasets_api, "delete_data", staticmethod(fake_delete))
    return calls


@pytest.mark.asyncio
async def test_publish_stores_typed_items_and_replaces_the_older_state_key(publish_env):
    scope = SimpleNamespace(session_id="s-2", user=USER, user_id=str(USER.id), dataset=DATASET)
    older = SimpleNamespace(
        id=uuid4(),
        system_metadata={
            "lesson_type": "state",
            "state_key": "test-baseline",
            "session_id": "s-1",
            "session_date": "2026-11-01",
        },
    )
    older_pitfall = SimpleNamespace(
        id=uuid4(),
        system_metadata={
            "lesson_type": "pitfall",
            "state_key": "pitfall:Edit:no-changes-to-make:py",
            "session_id": "s-1",
            "session_date": "2026-11-01",
            "seen_sessions": ["s-1"],
            "seen_in_sessions": 1,
        },
    )
    publish_env.existing = [older, older_pitfall]
    context = {"date": "2026-11-03", "end_commit": "abc", "branch": "dev"}
    accepted = [
        _written("state", "test-baseline", "pytest -q: 40 passed, 0 failed.", context),
        _written("pitfall", "pitfall:Edit:no-changes-to-make:py", "Edit fails → script.", context),
        _written("location", "", "Routes live in api/routes.py.", context),
    ]

    documents = await publish_distilled_lessons(scope, accepted, coding=True)

    items = publish_env.added
    # The lesson store reads the copy users cannot write.
    assert all(item.system_metadata == item.external_metadata for item in items)
    assert [item.external_metadata["lesson_type"] for item in items] == [
        "state",
        "pitfall",
        "location",
    ]
    assert items[1].external_metadata["seen_sessions"] == ["s-1", "s-2"]
    assert items[1].external_metadata["seen_in_sessions"] == 2
    assert "(seen in 2 sessions)" in items[1].data
    assert documents[0].startswith("Lesson-Type: state\nState-Key: test-baseline\n")
    assert all(item.literal_text for item in items)
    assert publish_env.add_kwargs["node_set"] == ["session_learnings"]
    assert sorted(publish_env.deleted) == sorted([older.id, older_pitfall.id])
    assert all(node_set == ["session_learnings"] for node_set in publish_env.finds)


@pytest.mark.asyncio
async def test_publish_keeps_the_pitfall_count_when_the_same_session_runs_again(publish_env):
    """A → B → B: the count reaches 2 and stays there, although A's row is gone."""
    pitfall = "pitfall:Edit:no-changes-to-make:py"
    scope_a = SimpleNamespace(session_id="s-a", user=USER, user_id=str(USER.id), dataset=DATASET)
    scope_b = SimpleNamespace(session_id="s-b", user=USER, user_id=str(USER.id), dataset=DATASET)

    await publish_distilled_lessons(
        scope_a, [_written("pitfall", pitfall, "Edit → script.")], coding=True
    )
    await publish_distilled_lessons(
        scope_b, [_written("pitfall", pitfall, "Edit → rewrite.")], coding=True
    )
    assert publish_env.added[0].external_metadata["seen_in_sessions"] == 2
    assert len(publish_env.deleted) == 1  # A's row

    await publish_distilled_lessons(
        scope_b, [_written("pitfall", pitfall, "Edit → rewrite.")], coding=True
    )
    assert publish_env.added[0].external_metadata["seen_in_sessions"] == 2
    assert "(seen in 2 sessions)" in publish_env.added[0].data
    assert len(publish_env.deleted) == 1  # nothing new to delete: the same row again


@pytest.mark.asyncio
async def test_publish_of_an_older_session_does_not_displace_a_newer_state(publish_env):
    scope = SimpleNamespace(session_id="s-old", user=USER, user_id=str(USER.id), dataset=DATASET)
    newer = SimpleNamespace(
        id=uuid4(),
        node_set='["session_learnings"]',
        system_metadata={
            "lesson_type": "state",
            "state_key": "test-baseline",
            "session_id": "s-new",
            "session_date": "2026-11-05",
        },
    )
    publish_env.existing = [newer]

    await publish_distilled_lessons(
        scope,
        [_written("state", "test-baseline", "pytest: 10 passed.", {"date": "2026-11-02"})],
        coding=True,
    )

    (stored,) = publish_env.added
    assert publish_env.deleted == [stored.data_id]


@pytest.mark.asyncio
async def test_publish_replaces_the_older_lesson_even_when_the_hook_after_cognify_fails(
    publish_env,
):
    from cognee.modules.pipelines.exceptions import AfterRunHookError

    scope = SimpleNamespace(session_id="s-2", user=USER, user_id=str(USER.id), dataset=DATASET)
    older = SimpleNamespace(
        id=uuid4(),
        node_set='["session_learnings"]',
        system_metadata={
            "lesson_type": "state",
            "state_key": "test-baseline",
            "session_id": "s-1",
        },
    )
    publish_env.existing = [older]
    publish_env.cognify_error = AfterRunHookError(
        [(DATASET.id, DATASET.name, RuntimeError("compaction"))]
    )

    with pytest.raises(AfterRunHookError):
        await publish_distilled_lessons(
            scope, [_written("state", "test-baseline", "pytest passes.")], coding=True
        )

    assert publish_env.deleted == [older.id]


@pytest.mark.asyncio
async def test_publish_keeps_the_same_sessions_identical_lesson(publish_env):
    scope = SimpleNamespace(session_id="s-1", user=USER, user_id=str(USER.id), dataset=DATASET)
    lesson = _written("state", "test-baseline", "pytest -q passes.")
    await publish_distilled_lessons(scope, [lesson], coding=True)
    (item,) = publish_env.added
    publish_env.existing = [SimpleNamespace(id=item.data_id, system_metadata=item.system_metadata)]

    await publish_distilled_lessons(scope, [lesson], coding=True)

    assert publish_env.deleted == []


@pytest.mark.asyncio
async def test_default_publish_adds_plain_documents(publish_env):
    scope = SimpleNamespace(session_id="s-1", user=USER, user_id=str(USER.id), dataset=DATASET)
    lesson = WrittenLesson(accept=True, statement="Plain lesson.", why_learned="while testing")

    documents = await publish_distilled_lessons(scope, [lesson])

    assert documents == ["# Session learning (session s-1)\n\nPlain lesson. (while testing.)\n"]
    assert publish_env.added == documents
    assert publish_env.deleted == []


# -- distill_session in coding mode ------------------------------------------------------


class _CodingSessionManager:
    def __init__(self, entries, qa_rows):
        self.store = [entry.model_dump() for entry in entries]
        self.qa_rows = qa_rows
        self.session_reads = []

    async def get_session_context_entries(self, **kwargs):
        return list(self.store)

    async def get_session(self, *, last_n=None, since=None, **kwargs):
        self.session_reads.append({"last_n": last_n, "since": since})
        rows = self.qa_rows[-last_n:] if last_n else self.qa_rows
        return [SimpleNamespace(model_dump=lambda row=row: dict(row)) for row in rows]

    async def update_session_context_entry(self, *, entry_id, merge, **kwargs):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, entry_dump, **kwargs):
        self.store.append(dict(entry_dump))
        return True


@pytest.mark.asyncio
async def test_distill_session_in_coding_mode_judges_each_candidate_by_its_turn(monkeypatch):
    entries = [
        _entry("Routes live in api/routes.py.", entry_id="c1", source_qa_ids=["q1"]),
        _entry("Tests live in tests/.", entry_id="c2"),  # no link: the newest turn
    ]
    qa_rows = [
        _row("q1", "Add a route", "2026-11-03T08:00:00", context='{"end_commit": "old"}'),
        _row("q2", "Unrelated", "2026-11-03T08:30:00"),
        _row("q3", "Add tests", "2026-11-03T09:00:00", context='{"end_commit": "abc"}'),
    ]
    manager = _CodingSessionManager(entries, qa_rows)
    monkeypatch.setattr(distill_module, "get_session_manager", lambda: manager)
    monkeypatch.setattr(distill_module, "coding_distillation_enabled", lambda: True)

    permissions_checked = []

    async def fake_datasets(requested, permission_type, user):
        permissions_checked.append(permission_type)
        return [DATASET]

    monkeypatch.setattr(distill_module, "get_authorized_existing_datasets", fake_datasets)
    seen = {}

    async def fake_propose(qa_rows, context_entries, *, coding=False):
        seen["propose"] = (qa_rows, coding)
        return distill_module.CuratorRun(
            proposed=[CodingProposedLesson(working_statement="x", member_entry_ids=["c1"])],
            failed_batches=0,
            curated_entry_ids={"c1"},
        )

    async def fake_accept(scope, proposed, context_entries, *, coding=False, qa_rows=None):
        seen["accept"] = (coding, [row["qa_id"] for row in qa_rows])
        return [_written("location")], 0

    async def fake_publish(scope, accepted, *, coding=False):
        seen["publish"] = coding
        return ["doc"]

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", fake_publish)

    result = await distill_module.distill_session("s-1", dataset=DATASET.name, user=USER)

    assert result.status == "completed"
    assert permissions_checked == ["write", "delete"]
    assert manager.session_reads == [{"last_n": None, "since": None}]
    # The turns the candidates came from, in time order; the unlinked turn q2 is not read.
    assert [row["question"] for row in seen["propose"][0]] == ["Add a route", "Add tests"]
    assert seen["propose"][1] is True
    assert seen["accept"] == (True, ["q1", "q3"]) and seen["publish"] is True


@pytest.mark.asyncio
async def test_coding_scope_requires_delete_permission(monkeypatch):
    from cognee.exceptions import CogneeValidationError

    async def write_only(requested, permission_type, user):
        return [DATASET] if permission_type == "write" else []

    monkeypatch.setattr(distill_module, "get_authorized_existing_datasets", write_only)

    scope = await distill_module.resolve_distillation_scope(
        session_id="s-1", dataset=DATASET.name, user=USER
    )
    assert scope.dataset is DATASET
    with pytest.raises(CogneeValidationError, match="delete permission"):
        await distill_module.resolve_distillation_scope(
            session_id="s-1", dataset=DATASET.name, user=USER, replaces_lessons=True
        )


# -- Duplicate proposals of several curator batches are folded (coding mode) --------


def _curator_fake(monkeypatch, per_batch, groups=()):
    """Stand-ins for the curator and the duplicate check.

    ``curate_batch`` returns the next item of ``per_batch``; ``find_duplicate_proposals``
    returns ``groups`` (an exception instance is raised) and records the proposals it got.
    """
    batch_calls, merge_calls = [], []
    outputs = iter(per_batch)

    async def fake_curate(batch_text, *, coding=False):
        batch_calls.append(batch_text)
        return next(outputs)

    async def fake_duplicates(proposed):
        merge_calls.append([lesson.working_statement for lesson in proposed])
        if isinstance(groups, Exception):
            raise groups
        return groups

    monkeypatch.setattr(distill_module, "curate_batch", fake_curate)
    monkeypatch.setattr(distill_module, "find_duplicate_proposals", fake_duplicates)
    return batch_calls, merge_calls


def _proposal(statement, members=(), state_key=""):
    return CodingProposedLesson(
        working_statement=statement, member_entry_ids=list(members), state_key=state_key
    )


def _seven_entries():
    """Seven candidates of one turn: two curator batches."""
    return [_entry(f"candidate {i}", entry_id=f"c{i}", source_qa_ids=["q1"]) for i in range(7)]


QA_ROW = [{"qa_id": "q1", "question": "Fix exits", "answer": "Command.execute exits 1."}]


@pytest.mark.asyncio
async def test_duplicates_across_batches_are_folded_and_the_rest_is_untouched(monkeypatch):
    """Two batches of one turn both propose the headline fact; one lesson remains, with
    the members of both, and the proposal nobody grouped is the object it was."""
    first = _proposal("Command.execute exits 1 on XitError.", ["c0"])
    other = _proposal("The CLI entry point is cli.py.", ["c1"])
    second = _proposal("Command.execute in commands.py exits with status 1.", ["c6"])
    batch_calls, merge_calls = _curator_fake(monkeypatch, [[first, other], [second]], [[1, 3]])

    run = await propose_lessons(QA_ROW, _seven_entries(), coding=True)

    assert len(batch_calls) == 2 and len(merge_calls) == 1
    assert [lesson.working_statement for lesson in run.proposed] == [
        "Command.execute exits 1 on XitError.",
        "The CLI entry point is cli.py.",
    ]
    assert run.proposed[0].member_entry_ids == ["c0", "c6"]
    assert run.proposed[1] is other
    assert run.curated_entry_ids == {f"c{i}" for i in range(7)}


@pytest.mark.asyncio
async def test_the_duplicate_check_sees_the_statements_only(monkeypatch):
    """The call's input is one numbered line per proposal: no turns, no candidates."""
    seen = {}

    async def fake_llm(*, text_input, system_prompt, response_model):
        seen.update(text=text_input, prompt=system_prompt, model=response_model)
        return response_model(groups=[[1, 2]])

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", fake_llm)
    proposals = [
        _proposal("[state-key: test-baseline] pytest -q:\n41 passed.", ["c0"], "test-baseline"),
        _proposal("Tests pass.", ["c6"]),
    ]

    groups = await distill_module.find_duplicate_proposals(proposals)

    assert seen["text"] == "1. pytest -q: 41 passed.\n2. Tests pass."
    assert seen["prompt"] == read_query_prompt("coding/session_distillation_merge_system.txt")
    assert seen["prompt"].strip() and "Propose at most" not in seen["prompt"]
    assert groups == [[1, 2]]


def test_a_group_keeps_its_keyed_proposal_and_never_spans_two_keys():
    """The state lesson survives its unkeyed restatement; a footprint and a pitfall the
    model grouped stay two lessons."""
    pitfall = _entry(
        "[state-key: pitfall:Edit:no-changes-to-make:py] Edit failed → Bash python",
        section="failure_lessons",
        source_trace_ids=["t1"],
        entry_id="pf",
    )
    proposals = [
        _proposal("Tests pass.", ["c0"]),
        _proposal("pytest -q: 41 passed.", ["c1"], "test-baseline"),
        _proposal("Export consists of export.py and cli.py.", ["c2"], "footprint:export"),
        _proposal("Edit on an unchanged string fails → use python.", ["pf"]),
    ]
    keys = [distill_module.proposal_key(lesson, {"pf": pitfall}) for lesson in proposals]

    merged = distill_module.merge_duplicate_proposals(proposals, [[1, 2], [3, 4]], keys)

    assert keys == ["", "test-baseline", "footprint:export", "pitfall:Edit:no-changes-to-make:py"]
    assert [lesson.working_statement for lesson in merged] == [
        "pytest -q: 41 passed.",
        "Export consists of export.py and cli.py.",
        "Edit on an unchanged string fails → use python.",
    ]
    assert merged[0].state_key == "test-baseline" and merged[0].member_entry_ids == ["c1", "c0"]
    assert merged[1] is proposals[2] and merged[2] is proposals[3]


def test_a_group_the_model_got_wrong_loses_nothing():
    """Unknown numbers, a number used twice and a group of one change nothing."""
    proposals = [_proposal(f"Lesson {i}.", [f"c{i}"]) for i in range(4)]
    keys = ["", "", "", ""]

    merged = distill_module.merge_duplicate_proposals(
        proposals, [[1, 2], [2, 3], [4], [0, 9, 4]], keys
    )

    assert [lesson.working_statement for lesson in merged] == [
        "Lesson 0.",
        "Lesson 2.",
        "Lesson 3.",
    ]
    assert merged[0].member_entry_ids == ["c0", "c1"]
    assert merged[1] is proposals[2] and merged[2] is proposals[3]


@pytest.mark.asyncio
async def test_one_batch_is_not_checked_for_duplicates(monkeypatch):
    entries = [_entry(f"candidate {i}", entry_id=f"c{i}") for i in range(3)]
    batch_calls, merge_calls = _curator_fake(
        monkeypatch, [[_proposal("A.", ["c0"]), _proposal("B.", ["c1"])]], [[1, 2]]
    )

    run = await propose_lessons([{"question": "q", "answer": "a"}], entries, coding=True)

    assert len(batch_calls) == 1 and merge_calls == [] and len(run.proposed) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("groups", [None, []])
async def test_a_failed_or_empty_duplicate_check_keeps_the_proposals(monkeypatch, groups):
    first, second = _proposal("A.", ["c0"]), _proposal("B.", ["c6"])
    _curator_fake(monkeypatch, [[first], [second]], groups)

    run = await propose_lessons(QA_ROW, _seven_entries(), coding=True)

    assert run.proposed == [first, second]
    assert run.failed_batches == 0 and not run.budget_exhausted


@pytest.mark.asyncio
async def test_an_exhausted_budget_in_the_duplicate_check_stops_the_run(monkeypatch):
    _curator_fake(
        monkeypatch,
        [[_proposal("A.", ["c0"])], [_proposal("B.", ["c6"])]],
        distill_module.LLMPaymentRequiredError("budget"),
    )

    run = await propose_lessons(QA_ROW, _seven_entries(), coding=True)

    assert run.budget_exhausted and len(run.proposed) == 2


@pytest.mark.asyncio
async def test_default_mode_never_checks_for_duplicates(monkeypatch):
    entries = [_entry(f"candidate {i}", entry_id=f"c{i}") for i in range(7)]
    batch_calls, merge_calls = _curator_fake(
        monkeypatch, [[_proposal("A.", ["c0"])], [_proposal("A.", ["c6"])]], [[1, 2]]
    )

    run = await propose_lessons([{"question": "q", "answer": "a", "time": "t"}], entries)

    assert len(batch_calls) == 2 and merge_calls == [] and len(run.proposed) == 2


# -- Lessons already in memory are the project's own ----------------------------------


@pytest.mark.asyncio
async def test_a_payload_search_keeps_the_hits_of_its_project_scope_only():
    """Repositories can share a dataset: a lesson of another project, or of none, is
    not one "already in memory" for this project."""
    seen = {}

    class Engine:
        async def search(self, collection, **kwargs):
            seen.update(kwargs)
            return [
                SimpleNamespace(
                    payload={"text": text, "belongs_to_set": ["session_learnings", *sets]}
                )
                for text, sets in (
                    ("Ours.", ["session_learnings:s-1", "proj:acme"]),
                    ("Theirs.", ["proj:other"]),
                    ("Both.", ["proj:acme", "proj:other"]),
                    ("Unscoped.", []),
                )
            ]

    texts = await distill_module.search_payload_texts(
        Engine(),
        "DocumentChunk_text",
        8,
        query_text="q",
        node_name=["session_learnings", "proj:acme"],
        node_name_filter_operator="AND",
        project_scope=("proj:acme",),
    )

    assert texts == ["Ours."]
    assert seen["node_name_filter_operator"] == "AND"
    assert await distill_module.search_payload_texts(
        Engine(), "DocumentChunk_text", 8, query_text="q", project_scope=()
    ) == ["Unscoped."]
