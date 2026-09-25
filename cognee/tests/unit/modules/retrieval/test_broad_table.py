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


def test_a_corpus_is_a_table_only_when_every_document_is_one():
    text = _csv(30)
    half = len(text) // 2
    cut = half  # chunks cut mid-record, as the chunker does
    chunks = [
        Unit(id="c1", text=text[:cut], document="d1"),
        Unit(id="c2", text=text[cut:], document="d1"),
    ]

    prose = Unit(id="c3", text="A long narrative with no records at all.", document="d2")

    assert _table_of(chunks) is not None and len(_table_of(chunks).rows) == 30
    assert _table_of([*chunks, prose]) is None
    assert _table_of([Unit(id="r1", text="a,b,c")]) is None  # a DLT row has no document


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
