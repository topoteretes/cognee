"""BROAD's table lane: delimited records parsed and counted by code (SDK-324)."""

import pytest

from cognee.modules.retrieval import broad_retriever
from cognee.modules.retrieval.broad_retriever import BroadRetriever, CountPlan, Unit, _table_of
from cognee.modules.retrieval.broad_table import (
    Table,
    TableFilter,
    TableQuery,
    parse_table,
    run_query,
)


def _csv(rows: int) -> str:
    lines = ["issue,assignee,verdict,notes"]
    for i in range(rows):
        who = ["Ann", "ann", "Bo", "Cy"][i % 4]
        verdict = "yes" if i % 3 else "no"
        lines.append(f'{i},{who},{verdict},"checked, then merged\nafter review"')
    return "\n".join(lines)


def test_a_csv_with_quoted_commas_and_line_breaks_parses_into_rows():
    table = parse_table(_csv(30))

    assert table is not None
    assert table.columns == ["issue", "assignee", "verdict", "notes"]
    assert len(table.rows) == 30 and table.rows[0][3] == "checked, then merged\nafter review"


def test_a_pipe_log_without_a_header_gets_numbered_columns():
    text = "\n".join(
        f"a{i:04x} | 2026-06-{i % 28 + 1:02d} | {['Ann', 'Bo'][i % 2]} | fix: thing {i}"
        for i in range(40)
    )

    table = parse_table(text)

    assert table is not None and table.columns[0] == "column 1" and len(table.rows) == 40


def test_a_free_text_field_holding_the_separator_stays_one_field():
    text = "\n".join(
        f"h{i} | Ann | fix: a | b | c" if i == 3 else f"h{i} | Ann | fix" for i in range(30)
    )

    table = parse_table(text)

    assert table is not None and table.rows[3][2] == "fix: a|b|c"


def test_prose_is_not_a_table():
    """Prose puts a varying number of commas on each line; templated sentences that share
    their comma count still vary in one place only."""
    varied = "\n".join(
        ", ".join(["In spring the committee met"] + ["and argued"] * (i % 5) + [f"then left {i}."])
        for i in range(60)
    )
    templated = "\n".join(
        f"In spring, the committee met, argued, and adjourned; nothing, it seemed, changed {i}."
        for i in range(60)
    )

    assert parse_table(varied) is None
    assert parse_table(templated) is None


def test_a_query_is_evaluated_exactly_over_every_row():
    table = parse_table(_csv(30))
    assert table is not None

    yes = run_query(
        table,
        TableQuery(
            answerable=True, filters=[TableFilter(column="verdict", op="equals", value="YES")]
        ),
    )
    people = run_query(
        table, TableQuery(answerable=True, aggregate="count_distinct", column="assignee")
    )
    tally = run_query(table, TableQuery(answerable=True, group_by="assignee"))
    ann = run_query(table, TableQuery(answerable=True, group_by="assignee", target="ann"))

    assert yes.total == 20
    assert people.total == 3  # "Ann" and "ann" are one name
    assert tally.groups[0] == ("Ann", 16)
    assert ann.total == 16 and ann.target_values == ["Ann", "ann"]


def test_a_partial_name_finds_its_value():
    table = Table(
        columns=["who", "what"], rows=[["Roman Shkarin", "a"], ["Roman Shkarin", "b"], ["Ann", "c"]]
    )

    result = run_query(table, TableQuery(answerable=True, group_by="who", target="Roman"))

    assert result.total == 2


def test_a_column_the_table_lacks_raises():
    table = Table(columns=["who"], rows=[["Ann"]])

    with pytest.raises(KeyError):
        run_query(table, TableQuery(answerable=True, group_by="author"))


def test_the_tables_in_a_corpus_are_offered_even_beside_prose():
    text = _csv(30)
    half = len(text) // 2
    cut = half  # chunks cut mid-record, as the chunker does
    chunks = [
        Unit(id="c1", text=text[:cut], document="d1"),
        Unit(id="c2", text=text[cut:], document="d1"),
    ]

    prose = Unit(id="c3", text="A long narrative with no records at all.", document="d2")

    whole, left_out = _table_of(chunks)
    mixed, beside = _table_of([*chunks, prose])

    assert whole is not None and len(whole.rows) == 30 and left_out == 0
    assert mixed is not None and len(mixed.rows) == 30 and beside == 1
    assert _table_of([prose]) == (None, 0)
    assert _table_of([Unit(id="r1", text="a,b,c")]) == (None, 0)  # a DLT row has no document


@pytest.mark.asyncio
async def test_a_question_the_columns_cannot_answer_is_read_instead(monkeypatch):
    async def fake(text_input, system_prompt, response_model, **kwargs):
        return TableQuery(answerable=False, reason="needs the notes understood")

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)
    table = parse_table(_csv(30))
    assert table is not None

    result = await BroadRetriever().count_table("q", CountPlan(source="text", item="x"), table)

    assert result is None


@pytest.mark.asyncio
async def test_a_table_count_is_exact_and_says_so(monkeypatch):
    async def fake(text_input, system_prompt, response_model, **kwargs):
        return TableQuery(
            answerable=True, filters=[TableFilter(column="verdict", op="equals", value="yes")]
        )

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)
    table = parse_table(_csv(30))
    assert table is not None
    retriever = BroadRetriever()

    result = await retriever.count_table("How many yes?", CountPlan(source="text", item="x"), table)
    assert result is not None
    context = await retriever.get_context_from_objects("How many yes?", result)

    assert (result.method, result.total, result.llm_calls) == ("table", 20, 1)
    assert "Exact" in context and "TOTAL (count rows): 20" in context


def test_a_json_array_of_records_is_a_table():
    import json

    records = [
        {
            "number": i,
            "author": {"login": ["ann", "bo"][i % 2]},
            "labels": ["bug", "ui"][: i % 3],
            "draft": i % 4 == 0,
            "comments": i % 5,
        }
        for i in range(30)
    ]
    table = parse_table(json.dumps(records, indent=2))

    assert table is not None
    assert table.columns == ["number", "author.login", "labels", "draft", "comments"]
    no_comments = run_query(
        table,
        TableQuery(
            answerable=True, filters=[TableFilter(column="comments", op="equals", value="0")]
        ),
    )
    assert no_comments.total == 6


def test_json_lines_are_a_table_and_a_single_object_is_not():
    import json

    lines = "\n".join(json.dumps({"id": i, "level": ["info", "error"][i % 2]}) for i in range(25))

    assert parse_table(lines) is not None and len(parse_table(lines).rows) == 25
    assert parse_table(json.dumps({"name": "cognee", "version": "1.0"})) is None


# --- line shapes ----------------------------------------------------------------


def test_a_log_line_keeps_its_wording_and_masks_its_values():
    from cognee.modules.retrieval.broad_table import shape_of

    line = shape_of("[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6")
    block = shape_of("081109 203615 INFO Received block blk_-160899 of size 91 from /10.250.10.6")

    assert line.shape == "[$ ^ #] [error] mod_jk child workerEnv in error state #"
    assert ("code", "blk_-160899") in block.slots and ("dotted", "10.250.10.6") in block.slots


def test_a_word_with_digits_is_one_value():
    """ "test123" is a username and "subdir57" a directory: each is one value, never a
    word plus a number, and a word without digits stays part of the wording."""
    from cognee.modules.retrieval.broad_table import shape_of

    user = shape_of("Invalid user test123 from 1.2.3.4")
    folder = shape_of("Deleting subdir57 now")
    path = shape_of("Deleting file /data/current/subdir57/blk_9")

    assert user.shape == "Invalid user % from ~" and ("code", "test123") in user.slots
    assert folder.shape == "Deleting % now"
    assert path.shape == "Deleting file /" and path.slots == [
        ("path", "/data/current/subdir57/blk_9")
    ]
    assert shape_of("scored in the 12th minute").shape == "scored in the #th minute"


def test_prose_has_too_many_shapes_to_be_records():
    import random

    from cognee.modules.retrieval.broad_table import shaped_lines

    words = [
        "river",
        "stone",
        "quietly",
        "after",
        "the",
        "storm",
        "we",
        "walked",
        "toward",
        "a",
        "village",
        "where",
        "old",
        "bells",
        "rang",
    ]
    rng = random.Random(3)
    prose = "\n".join(
        " ".join(rng.choice(words) for _ in range(rng.randint(6, 14))) for _ in range(60)
    )
    logs = "\n".join(
        f"Jun {i % 28 + 1} 10:{i % 60:02d}:00 host sshd[{i}]: Failed password for root from 10.0.0.{i % 9}"
        for i in range(60)
    )

    assert shaped_lines(prose) is None
    assert shaped_lines(logs) is not None


def _ssh_log() -> str:
    return "\n".join(
        (
            f"Jun {i % 28 + 1} 10:00:0{i % 10} host sshd[{100 + i}]: "
            + (
                f"Failed password for root from 10.0.0.{i % 5} port {2000 + i}"
                if i % 3 == 1
                else f"Failed password for invalid user u{i % 7} from 10.0.0.{i % 2} port {i}"
                if i % 3 == 2
                else "Accepted password for bo"
            )
        )
        for i in range(90)
    )


def test_a_line_query_selects_every_form_of_a_line_and_captures_its_value():
    """ "Failed password for root" and "... for invalid user u3" are one event in two forms;
    one expression selects both, and code counts every such line."""
    from cognee.modules.retrieval.broad_table import LineQuery, matched_table, shaped_lines

    lines, _ = shaped_lines(_ssh_log())
    query = LineQuery(
        answerable=True,
        line_regex=r"Failed password for",
        value_regex=r"from (\d+\.\d+\.\d+\.\d+)",
        group=True,
    )

    table = matched_table(lines, query)
    tally = run_query(table, TableQuery(answerable=True, group_by="value"))

    assert len(table.rows) == 60
    assert dict(tally.groups) == {
        "10.0.0.0": 21,
        "10.0.0.1": 21,
        "10.0.0.2": 6,
        "10.0.0.3": 6,
        "10.0.0.4": 6,
    }


def test_every_match_on_a_line_is_a_value_and_exclusions_apply():
    from cognee.modules.retrieval.broad_table import LineQuery, matched_table, shaped_lines

    text = "\n".join(
        [f"INFO delete blk_{i}a blk_{i}b blk_{i}c" for i in range(10)]
        + [f"WARN lost blk_{i}a" for i in range(20)]
        + [f"WARN lost blk_x{i} (test)" for i in range(5)]
    )
    lines, _ = shaped_lines(text)
    blocks = matched_table(
        lines, LineQuery(answerable=True, line_regex="blk_", value_regex=r"(blk_\w+)")
    )
    warns = matched_table(
        lines, LineQuery(answerable=True, line_regex="^WARN", exclude_regex=r"\(test\)")
    )

    distinct = run_query(
        blocks, TableQuery(answerable=True, aggregate="count_distinct", column="value")
    )
    assert distinct.total == 45  # 30 on INFO lines, 10 more on WARN, 5 on test lines
    assert len(warns.rows) == 20


def test_an_expression_that_does_not_compile_raises():
    from cognee.modules.retrieval.broad_table import (
        LineQuery,
        LineQueryError,
        matched_table,
        shaped_lines,
    )

    lines, _ = shaped_lines(_ssh_log())

    with pytest.raises(LineQueryError):
        matched_table(lines, LineQuery(answerable=True, line_regex="Failed (password"))
    with pytest.raises(LineQueryError):
        matched_table(lines, LineQuery(answerable=True, line_regex="Failed", value_regex="from"))


def test_an_expression_that_backtracks_forever_is_stopped():
    """A model can write nested repetition; on a long line it would run for minutes."""
    import time

    from cognee.modules.retrieval.broad_table import Line, LineQuery, LineQueryError, matched_table

    lines = [Line(shape="", slots=[], text="a" * 60 + "!") for _ in range(30)]
    started = time.monotonic()

    with pytest.raises(LineQueryError):
        matched_table(lines, LineQuery(answerable=True, line_regex=r"(a|aa)+b"))
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_log_lines_are_counted_by_code_after_one_query_call(monkeypatch):
    from cognee.modules.retrieval.broad_table import LineQuery

    log = _ssh_log()
    units = [
        Unit(id="c1", text=log[:2500], document="d"),
        Unit(id="c2", text=log[2500:], document="d"),
    ]
    calls = []

    async def fake(text_input, system_prompt, response_model, **kwargs):
        calls.append(response_model)
        if response_model is CountPlan:
            return CountPlan(source="text", item="a failed password attempt")
        return LineQuery(answerable=True, line_regex="Failed password")

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)
    monkeypatch.setattr(
        broad_retriever.BroadRetriever, "load_text_units", lambda self, graph: _awaitable(units)
    )
    monkeypatch.setattr(
        broad_retriever.BroadRetriever, "load_entities", lambda self, graph: _awaitable({})
    )
    monkeypatch.setattr(
        broad_retriever,
        "get_unified_engine",
        lambda: _awaitable(type("E", (), {"graph": None})()),
    )

    result = await BroadRetriever().get_retrieved_objects("How many failed password attempts?")

    assert (result.method, result.total, result.units) == ("table", 60, 90)
    assert calls == [CountPlan, LineQuery]


async def _awaitable(value):
    return value


def test_grouping_by_a_value_counts_rows_per_value_even_if_asked_for_distinct():
    """ "Which IP made the most requests" grouped by IP: distinct IPs per IP is always 1."""
    table = Table(columns=["ip"], rows=[["a"], ["a"], ["b"]])

    result = run_query(
        table, TableQuery(answerable=True, aggregate="count_distinct", column="ip", group_by="ip")
    )

    assert result.groups[0] == ("a", 2)


@pytest.mark.asyncio
async def test_an_expression_matching_nothing_is_retried_with_real_lines(monkeypatch):
    from cognee.modules.retrieval.broad_table import LineQuery, shaped_lines

    lines, shapes = shaped_lines(_ssh_log())
    seen = []

    async def fake(text_input, system_prompt, response_model, **kwargs):
        seen.append(text_input)
        if len(seen) == 1:
            return LineQuery(answerable=True, line_regex=r"^\d+ Failed")  # lines start "Jun"
        return LineQuery(answerable=True, line_regex="Failed password")

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)

    result = await BroadRetriever().count_lines(
        "q", CountPlan(source="text", item="x"), lines, shapes
    )

    assert result is not None and result.total == 60
    assert "matched none of the 90 lines" in seen[1] and "Jun " in seen[1]


@pytest.mark.asyncio
async def test_a_distinct_count_without_a_captured_value_is_retried_not_raised(monkeypatch):
    """count_distinct needs a value to count; a query without one is sent back once and,
    if still unusable, left to the reading path instead of failing the search."""
    from cognee.modules.retrieval.broad_table import LineQuery, shaped_lines

    lines, shapes = shaped_lines(_ssh_log())
    answers = [
        LineQuery(answerable=True, line_regex="Failed", aggregate="count_distinct"),
        LineQuery(
            answerable=True,
            line_regex="Failed",
            aggregate="count_distinct",
            value_regex=r"from (\S+) port",
        ),
    ]

    async def fake(text_input, system_prompt, response_model, **kwargs):
        return answers.pop(0)

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)

    result = await BroadRetriever().count_lines(
        "q", CountPlan(source="text", item="x"), lines, shapes
    )

    assert result is not None and result.total == 5 and not answers


def test_lines_that_begin_alike_are_records_even_when_their_messages_vary():
    """A log whose messages are free text still starts every line with a timestamp and a
    host; prose does not start its lines alike."""
    import random

    from cognee.modules.retrieval.broad_table import shaped_lines

    rng = random.Random(1)
    words = ["disk", "fan", "usb", "wifi", "sleep", "wake", "thermal", "battery", "audio", "gpu"]
    log = "\n".join(
        f"Jul {i % 28 + 1} 10:{i % 60:02d}:00 host-{i % 3}-7 {rng.choice(words)}d[{i}]: "
        + " ".join(rng.choice(words) for _ in range(rng.randint(3, 9)))
        for i in range(200)
    )
    prose = "\n".join(
        " ".join(rng.choice(words + ["the", "a", "we", "then"]) for _ in range(rng.randint(5, 12)))
        for _ in range(200)
    )

    assert shaped_lines(log) is not None
    assert shaped_lines(prose) is None


@pytest.mark.asyncio
async def test_a_count_over_the_table_beside_prose_says_what_it_covered(monkeypatch):
    async def fake(text_input, system_prompt, response_model, **kwargs):
        return TableQuery(
            answerable=True, filters=[TableFilter(column="verdict", op="equals", value="yes")]
        )

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)
    table = parse_table(_csv(30))
    assert table is not None

    result = await BroadRetriever().count_table("q", CountPlan(source="text", item="x"), table)
    assert result is not None
    result.scope = (
        "only the table in the documents; 1 other documents (not tables) were not counted"
    )

    assert "1 other documents" in broad_retriever._provenance_note(result)


def test_a_json_map_of_records_is_a_table_and_nested_maps_stay_one_field():
    """A lockfile's packages are keyed by path; each package's dependencies are a map of
    names, which is one list-valued field, not a column per dependency."""
    import json

    packages = {
        f"node_modules/p{i}": {
            "version": f"1.{i}.0",
            "dev": i % 3 == 0,
            "dependencies": {f"d{(i + j) % 20}": "^1" for j in range(i % 4)},
        }
        for i in range(30)
    }
    table = parse_table(json.dumps({"name": "app", "packages": packages}))

    assert table is not None and len(table.rows) == 30
    assert table.columns == ["key", "version", "dev", "dependencies"]
    dev = run_query(
        table,
        TableQuery(answerable=True, filters=[TableFilter(column="dev", op="equals", value="true")]),
    )
    uses_d2 = run_query(
        table,
        TableQuery(
            answerable=True, filters=[TableFilter(column="dependencies", op="equals", value="d2")]
        ),
    )
    assert dev.total == 10 and uses_d2.total == sum(
        1 for i in range(30) if any((i + j) % 20 == 2 for j in range(i % 4))
    )


def test_an_iso_timestamp_is_one_value():
    from cognee.modules.retrieval.broad_table import shape_of

    line = shape_of("2026-09-25T19:52:32.9084314Z step finished")

    assert line.shape == "$ step finished" and line.slots[0][0] == "date"


def test_an_mbox_archive_is_one_row_per_message():
    messages = "".join(
        f"From dev{i % 3}@example.org Mon May  6 10:{i:02d}:00 2024\n"
        f"From: Dev {i % 3} <dev{i % 3}@example.org>\n"
        f"To: dev@list.org\nDate: Mon, 6 May 2024 10:{i:02d}:00 +0000\n"
        f"Subject: {'[jira] ' if i % 4 == 0 else 'Re: '}topic {i}\n\n"
        f"Body line one.\nFrom the logs we see nothing.\n\n"
        for i in range(24)
    )
    table = parse_table(messages)

    assert table is not None and len(table.rows) == 24
    by_sender = run_query(table, TableQuery(answerable=True, group_by="from_name"))
    jira = run_query(
        table,
        TableQuery(
            answerable=True,
            filters=[TableFilter(column="subject", op="starts_with", value="[jira]")],
        ),
    )
    assert by_sender.groups[0] == ("Dev 0", 8) and jira.total == 6


def test_mentions_matches_whole_words_and_plurals_contains_matches_parts():
    table = Table(
        columns=["text"],
        rows=[
            ["Oil prices rise"],
            ["A foiled plot"],
            ["Boilermakers win"],
            ["Rodents found"],
            ["rodent droppings"],
        ],
    )

    def count(op, value):
        return run_query(
            table,
            TableQuery(answerable=True, filters=[TableFilter(column="text", op=op, value=value)]),
        ).total

    assert count("mentions", "oil") == 1 and count("contains", "oil") == 3
    assert count("mentions", "rodent") == 2
