"""BROAD's record store: documents as SQLite tables, queried read-only (SDK-324)."""

import json

import pytest

from cognee.modules.retrieval.broad_store import (
    BroadQueryError,
    RecordStore,
    _number,
    norm,
    parse_dlt_row,
    parse_records,
    shape_of,
)


def _csv(rows: int = 12) -> str:
    return "\n".join(["name,team,goals"] + [f"P{i},{'AB'[i % 2]},{i}" for i in range(rows)])


@pytest.fixture
def store():
    records = RecordStore()
    yield records
    records.close()


def test_delimited_json_markdown_and_mail_documents_are_records():
    markdown = "| name | goals |\n|---|---|\n" + "\n".join(f"| P{i} | {i} |" for i in range(12))
    mail = "".join(
        f"From a@x {i}\nFrom: Ann <a@x.org>\nSubject: s{i}\n\nbody {i}\n\n" for i in range(3)
    )

    assert parse_records("t", _csv()).columns == ["name", "team", "goals"]
    assert parse_records("m", markdown).columns == ["name", "goals"]
    assert len(parse_records("j", json.dumps([{"a": 1}, {"a": 2, "b": {"c": 3}}])).rows) == 2
    assert parse_records("e", mail).columns[:2] == ["from_name", "from_address"]
    assert parse_records("p", "A story.\nIt has lines, some with commas.\n" * 5) is None


def test_numbers_keep_ids_and_european_decimals_as_text():
    assert (_number("1,234"), _number("7"), _number("2.5")) == (1234, 7, 2.5)
    assert _number("00123") is None and _number("12,34") is None and _number("nan") is None


def test_a_dlt_row_gives_its_table_fields_and_foreign_keys():
    text = (
        "Table: orders\nColumns:\n  - id (INTEGER)\nForeign Keys:\n"
        "  - customer_id references customers.id\n\nRow Data:\n  id: 7\n  customer_id: 3"
    )

    assert parse_dlt_row(text) == (
        "orders",
        {"id": "7", "customer_id": "3"},
        ["customer_id references customers.id"],
    )


def test_a_line_shape_masks_names_and_anything_with_a_digit():
    assert shape_of("Ana María scored in the 12th minute.") == (
        "@ scored in the # minute.",
        ["Ana María", "12th"],
    )


def test_tables_are_queried_exactly_and_text_compares_without_case_or_accents(store):
    store.add_document("a.csv", _csv())
    store.add_document("b.csv", _csv())  # same columns: one table
    store.add_document("notes.txt", "Match 1, 7 December\nÉmile scored.\n\nMatch 2\nNobody.")

    assert store.run('SELECT COUNT(*), SUM("goals") FROM "a" WHERE norm("team") = norm(\'a\')')[
        1
    ] == [(12, 60)]
    assert store.run("SELECT COUNT(*) FROM lines WHERE norm(text) LIKE '%emile%'")[1] == [(1,)]
    assert store.run("SELECT block FROM lines WHERE text = 'Nobody.'")[1] == [("Match 2",)]
    assert store.run("SELECT words('The cat, the Cat.', 'cat')")[1] == [(2,)]


def test_the_schema_shows_values_spelled_like_words_of_the_question(store):
    store.add_document(
        "heroes.csv", "\n".join(["hero,publisher"] + [f"H{i},Marvel Comics" for i in range(12)])
    )

    schema = store.describe("How many heroes did marvel publish?")

    assert 'TABLE "heroes" (12 rows' in schema and "'Marvel Comics'" in schema


def test_only_one_read_only_select_runs(store):
    store.add_document("a.csv", _csv())

    for statement in (
        'DROP TABLE "a"',
        "INSERT INTO lines (text) VALUES ('x')",
        "ATTACH 'x.db' AS x",
        "PRAGMA writable_schema = 1",
        "SELECT 1; SELECT 2",
    ):
        with pytest.raises(BroadQueryError):
            store.run(statement)
    assert store.run('SELECT COUNT(*) FROM "a"')[1] == [(12,)]


def test_norm_folds_case_and_accents():
    assert norm("  Ünal  ÖZTÜRK ") == "unal ozturk"
