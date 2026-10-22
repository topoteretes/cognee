"""read_enola_docs and read_enola_fact_docs: the readers of enola's docs (enola >= 0.4.32).

docs.jsonl holds the whole comment of every documented symbol and module, one
JSON line per fact, joined to facts.jsonl by the fact ``id``. The reader takes
only a file the snapshot's receipt.json vouches for -- the rule flows.json
follows, shared through _receipt_vouches_for -- and names the snapshot the docs
belong to, so the code docs tasks can tell whether they describe the code graph
they hold. Declarations that share an id (overloads, a property's getter and
setter) each have a line there, but enola 0.4.32 can give them all one
declaration's comment; read_enola_fact_docs returns each one's own first
paragraph from facts.jsonl, by (id, line), to check them against.

Covered: lines in file order and the snapshot id; every reason the docs are
unavailable, a snapshot directory that is gone first; a missing or unparseable
receipt; which lines are skipped as malformed and how they are counted, lines
json.loads cannot decode included; the fact docs of the requested ids, the fact
lines that give none, and a facts.jsonl that is missing, unreadable or not
needed; flows.json vouching, unchanged; and the real enola binary on two tiny
Python packages.
"""

import importlib
import json
import os
import re
import sys
from unittest.mock import MagicMock

import pytest

from cognee.tasks.code_graph.enola import (
    DOCS_ARTIFACT,
    EnolaDocs,
    EnolaDocsUnavailable,
    EnolaNotInstalledError,
    _receipt_vouches_for_flows,
    find_enola_binary,
    parse_enola_snapshot,
    read_enola_docs,
    read_enola_fact_docs,
    run_enola_generate,
    snapshot_identity,
)
from cognee.tasks.code_graph.extract_code_graph import map_facts_to_data_points
from cognee.tests.utils.enola_snapshot import doc_line, sha256_label, write_enola_snapshot

enola_module = importlib.import_module("cognee.tasks.code_graph.enola")

REPO = "shop"
# write_enola_snapshot's default receipt snapshot_id.
SNAPSHOT_ID = "snapshot-1"
# The first enola release that writes docs.jsonl.
DOCS_RELEASE = (0, 4, 32)

TOTAL = doc_line(
    REPO,
    "symbol",
    "pricing/rules.total",
    "pricing/rules.py",
    "Sum the prices and take the discount off.\n\nA discount above one is a percentage.",
    line=4,
)
# An overload: the same repo, kind, name and file, so the same fact id.
TOTAL_OF_ONE_CART = doc_line(
    REPO,
    "symbol",
    "pricing/rules.total",
    "pricing/rules.py",
    "Sum the prices of one cart.",
    line=19,
)
PACKAGE = doc_line(REPO, "module", "pricing", "pricing", "Price a shopping cart.")

# The facts of those docs as enola writes them: the ``doc`` prop is the first
# paragraph of the comment, and a symbol declares its package by the package's
# id, so that id is in the symbol's facts.jsonl line too.
PACKAGE_FACT = {
    "kind": "module",
    "name": "pricing",
    "file": "pricing",
    "repo": REPO,
    "props": {"doc": "Price a shopping cart.", "language": "python"},
    "id": PACKAGE["id"],
}
TOTAL_FACT = {
    "kind": "symbol",
    "name": "pricing/rules.total",
    "file": "pricing/rules.py",
    "line": 4,
    "repo": REPO,
    "props": {"doc": "Sum the prices and take the discount off.", "language": "python"},
    "relations": [{"kind": "declares", "target": "pricing", "target_id": PACKAGE["id"]}],
    "id": TOTAL["id"],
}
TOTAL_OF_ONE_CART_FACT = {
    **TOTAL_FACT,
    "line": 19,
    "props": {"doc": "Sum the prices of one cart.", "language": "python"},
}
FACTS = [PACKAGE_FACT, TOTAL_FACT, TOTAL_OF_ONE_CART_FACT]
# read_enola_fact_docs of FACTS, every id asked for.
FACT_DOCS = {
    (PACKAGE["id"], None): "Price a shopping cart.",
    (TOTAL["id"], 4): "Sum the prices and take the discount off.",
    (TOTAL["id"], 19): "Sum the prices of one cart.",
}

# docs.jsonl bytes with a Latin-1 byte in a doc.
NOT_UTF8 = (json.dumps(TOTAL) + "\n").encode("utf-8").replace(b"Sum", b"S\xfcm")

# The most digits json.loads converts to an integer; 0 when this Python sets no
# such limit or it is switched off.
INT_DIGIT_LIMIT = getattr(sys, "get_int_max_str_digits", lambda: 0)()

_MISSING = object()


def _receipt_of(directory):
    return json.loads((directory / "receipt.json").read_text(encoding="utf-8"))


def _write_receipt(directory, receipt):
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def _snapshot_with_docs_bytes(directory, content):
    """A snapshot whose docs.jsonl is exactly ``content``, its receipt hashing those bytes."""
    write_enola_snapshot(directory, facts=FACTS, docs=[])
    (directory / DOCS_ARTIFACT).write_bytes(content)
    receipt = _receipt_of(directory)
    receipt["output_hashes"][DOCS_ARTIFACT] = sha256_label(content)
    _write_receipt(directory, receipt)
    return directory


def _jsonl(*entries):
    """JSON Lines bytes: a dict entry is dumped as one JSON line, a str entry is written as is."""
    lines = (entry if isinstance(entry, str) else json.dumps(entry) for entry in entries)
    return "".join(line + "\n" for line in lines).encode("utf-8")


def _write_facts(directory, *entries):
    """facts.jsonl of ``entries`` (see _jsonl): all of a snapshot read_enola_fact_docs reads."""
    (directory / "facts.jsonl").write_bytes(_jsonl(*entries))


def _variant(**changes):
    """TOTAL as one JSON line, with fields changed (or removed, when set to _MISSING)."""
    entry = {**TOTAL, **changes}
    return json.dumps({key: value for key, value in entry.items() if value is not _MISSING})


def _fact_variant(**changes):
    """TOTAL_FACT as one facts.jsonl line, with fields changed (or removed, when set to _MISSING)."""
    fact = {**TOTAL_FACT, **changes}
    return json.dumps({key: value for key, value in fact.items() if value is not _MISSING})


def _nested_past_the_recursion_limit(inner=""):
    """A JSON array around ``inner``, nested deeper than json.loads follows (RecursionError)."""
    return "[" * 100_000 + inner + "]" * 100_000


@pytest.fixture
def enola_logger(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(enola_module, "logger", mock_logger)
    return mock_logger


# --- a vouched file ------------------------------------------------------------


def test_a_vouched_docs_file_reads_into_its_lines_in_file_order(tmp_path):
    """Overloads share an id and stay apart; ``repo`` and ``line`` may be absent or null."""
    unlabelled = doc_line(None, "symbol", "tools.main", "tools.py", "Run the tools.", line=1)
    null_fields = {
        **doc_line(None, "module", "tools", "tools", "Command line tools."),
        "repo": None,
    }
    lines = [TOTAL_OF_ONE_CART, {**PACKAGE, "line": None}, TOTAL, unlabelled, null_fields]
    write_enola_snapshot(tmp_path, facts=FACTS, docs=lines)

    docs = read_enola_docs(str(tmp_path))

    assert isinstance(docs, EnolaDocs)
    assert docs.lines == tuple(lines)
    assert docs.snapshot_id == snapshot_identity(tmp_path, _receipt_of(tmp_path)) == SNAPSHOT_ID


def test_a_receipt_without_a_snapshot_id_names_the_snapshot_by_its_facts(tmp_path):
    """The same fallback add_code_graph_edges stamps as the repository's last_snapshot_id."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL], snapshot_id=None)

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (TOTAL,)
    assert docs.snapshot_id == sha256_label((tmp_path / "facts.jsonl").read_bytes())
    assert docs.snapshot_id == snapshot_identity(tmp_path, _receipt_of(tmp_path))


@pytest.mark.parametrize(
    "content", [pytest.param(b"", id="empty"), pytest.param(b"\n   \n", id="blank-lines")]
)
def test_an_empty_vouched_docs_file_documents_nothing(tmp_path, enola_logger, content):
    """Nothing documented, which is not "docs not written": no lines, and no refusal."""
    _snapshot_with_docs_bytes(tmp_path, content)

    assert read_enola_docs(tmp_path) == EnolaDocs(snapshot_id=SNAPSHOT_ID, lines=())
    enola_logger.warning.assert_not_called()


# --- unavailable docs -----------------------------------------------------------


def test_a_snapshot_directory_that_does_not_exist_is_unavailable(tmp_path):
    """A lost clone, or a code file's staging directory removed after its run: the
    reason says the directory is gone, not that a snapshot lacks docs.jsonl."""
    gone = tmp_path / "gone"

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(str(gone))

    assert f"the snapshot directory '{gone}' does not exist" in str(raised.value)
    assert f"no {DOCS_ARTIFACT}" not in str(raised.value)


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="needs POSIX permissions that bind the current user (not root)",
)
def test_a_snapshot_directory_that_cannot_be_searched_is_unavailable(tmp_path):
    """Checking whether the directory exists can fail itself (a parent the process
    may not search): that is an unreadable snapshot, not an error that escapes the
    extraction and costs every other repository its docs."""
    snapshot_dir = tmp_path / "locked" / "repo" / ".enola"
    write_enola_snapshot(snapshot_dir, facts=FACTS, docs=[TOTAL])
    (tmp_path / "locked").chmod(0)
    try:
        with pytest.raises(EnolaDocsUnavailable) as raised:
            read_enola_docs(snapshot_dir)
    finally:
        (tmp_path / "locked").chmod(0o700)

    assert f"could not read '{snapshot_dir}'" in str(raised.value)


def test_a_snapshot_without_docs_jsonl_is_unavailable(tmp_path):
    """A snapshot from enola before 0.4.32, or from a configuration without the docs renderer."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=None)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path)

    assert f"no {DOCS_ARTIFACT} in '{tmp_path}'" in str(raised.value)
    assert "does not exist" not in str(raised.value)


def test_a_docs_jsonl_that_cannot_be_read_is_unavailable(tmp_path):
    write_enola_snapshot(tmp_path, facts=FACTS, docs=None)
    (tmp_path / DOCS_ARTIFACT).mkdir()

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path)

    assert f"could not read '{tmp_path / DOCS_ARTIFACT}'" in str(raised.value)


def test_docs_bytes_the_receipt_hashes_differently_are_unavailable(tmp_path):
    """The file changed after the run the receipt describes."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL])
    with open(tmp_path / DOCS_ARTIFACT, "a", encoding="utf-8") as docs_file:
        docs_file.write(json.dumps(PACKAGE) + "\n")

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path)

    assert f"receipt.json in '{tmp_path}' does not list these {DOCS_ARTIFACT} bytes" in str(
        raised.value
    )


def test_output_hashes_without_docs_jsonl_refuse_it_whatever_else_they_list(tmp_path):
    """An earlier run's file. A receipt with output hashes is not overruled by its
    renderers, and another artifact's entry vouches for nothing, even with these bytes."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL], vouch_docs=False)
    receipt = _receipt_of(tmp_path)
    receipt["renderers"] = ["llm_context", "flows", "docs"]
    receipt["output_hashes"]["flows.json"] = sha256_label((tmp_path / DOCS_ARTIFACT).read_bytes())
    _write_receipt(tmp_path, receipt)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path)

    assert "does not list these" in str(raised.value)


@pytest.mark.parametrize(
    "receipt_fields",
    [
        pytest.param({"renderers": ["llm_context", "flows", "docs"]}, id="docs-renderer-ran"),
        pytest.param({"output_hashes": None, "renderers": ["docs"]}, id="null-output-hashes"),
    ],
)
def test_a_receipt_without_output_hashes_vouches_through_the_docs_renderer(
    tmp_path, receipt_fields
):
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL])
    _write_receipt(tmp_path, {"snapshot_id": SNAPSHOT_ID, "format_version": 1, **receipt_fields})

    assert read_enola_docs(tmp_path) == EnolaDocs(snapshot_id=SNAPSHOT_ID, lines=(TOTAL,))


@pytest.mark.parametrize(
    "receipt_fields",
    [
        pytest.param({"renderers": ["llm_context", "flows"]}, id="docs-renderer-did-not-run"),
        pytest.param({"renderers": "docs"}, id="renderers-not-a-list"),
        pytest.param({}, id="neither-hashes-nor-renderers"),
    ],
)
def test_a_receipt_without_output_hashes_or_the_docs_renderer_refuses_the_file(
    tmp_path, receipt_fields
):
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL])
    _write_receipt(tmp_path, {"snapshot_id": SNAPSHOT_ID, "format_version": 1, **receipt_fields})

    with pytest.raises(EnolaDocsUnavailable):
        read_enola_docs(tmp_path)


def test_docs_bytes_that_are_not_utf8_are_unavailable(tmp_path):
    _snapshot_with_docs_bytes(tmp_path, NOT_UTF8)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path)

    assert f"'{tmp_path / DOCS_ARTIFACT}' is not UTF-8" in str(raised.value)


# --- no receipt to check against ------------------------------------------------


def test_without_a_receipt_the_file_is_read_and_named_by_its_facts(tmp_path):
    """A writer that left no receipt: nothing to check, so the file is read, and the
    snapshot id falls back to the hash of facts.jsonl (None without it either)."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL, PACKAGE], receipt=False)

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (TOTAL, PACKAGE)
    assert docs.snapshot_id == sha256_label((tmp_path / "facts.jsonl").read_bytes())
    assert docs.snapshot_id == snapshot_identity(tmp_path, None)
    (tmp_path / "facts.jsonl").unlink()
    assert read_enola_docs(tmp_path) == EnolaDocs(snapshot_id=None, lines=(TOTAL, PACKAGE))


@pytest.mark.parametrize(
    "receipt_bytes",
    [
        pytest.param(b"{ not json", id="not-json"),
        pytest.param(b"\xff\xfe not utf-8", id="not-utf-8"),
        pytest.param(b'["snapshot-1"]', id="not-an-object"),
    ],
)
def test_an_unparseable_receipt_reads_like_no_receipt(tmp_path, receipt_bytes):
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL])
    (tmp_path / "receipt.json").write_bytes(receipt_bytes)

    docs = read_enola_docs(tmp_path)

    assert docs == EnolaDocs(
        snapshot_id=sha256_label((tmp_path / "facts.jsonl").read_bytes()), lines=(TOTAL,)
    )


# --- malformed lines ------------------------------------------------------------

MALFORMED_LINES = [
    pytest.param("{ not json", id="not-json"),
    pytest.param(json.dumps([TOTAL]), id="array"),
    pytest.param(json.dumps("Sum the prices."), id="string"),
    pytest.param("null", id="null"),
    pytest.param(_variant(id=_MISSING), id="id-missing"),
    pytest.param(_variant(id="A" * 32), id="id-uppercase"),
    pytest.param(_variant(id="a" * 31), id="id-short"),
    pytest.param(_variant(id="g" * 32), id="id-not-hex"),
    pytest.param(_variant(id=int("1" * 32)), id="id-number"),
    pytest.param(_variant(kind=_MISSING), id="kind-missing"),
    pytest.param(_variant(kind="file"), id="kind-file"),
    pytest.param(_variant(kind="Symbol"), id="kind-capitalized"),
    pytest.param(_variant(name=""), id="name-empty"),
    pytest.param(_variant(name=_MISSING), id="name-missing"),
    pytest.param(_variant(name=7), id="name-not-text"),
    pytest.param(_variant(file=""), id="file-empty"),
    pytest.param(_variant(file=_MISSING), id="file-missing"),
    pytest.param(_variant(doc=""), id="doc-empty"),
    pytest.param(_variant(doc=" \n\t "), id="doc-blank"),
    pytest.param(_variant(doc=_MISSING), id="doc-missing"),
    pytest.param(_variant(doc=["Sum", "the", "prices."]), id="doc-not-text"),
    pytest.param(_variant(line=True), id="line-bool"),
    pytest.param(_variant(line="4"), id="line-text"),
    pytest.param(_variant(line=4.0), id="line-float"),
    pytest.param(_variant(repo=7), id="repo-number"),
    pytest.param(_variant(repo=[REPO]), id="repo-list"),
]


@pytest.mark.parametrize("malformed", MALFORMED_LINES)
def test_a_malformed_line_is_skipped_and_the_lines_around_it_survive(
    tmp_path, enola_logger, malformed
):
    _snapshot_with_docs_bytes(tmp_path, _jsonl(PACKAGE, malformed, TOTAL))

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (PACKAGE, TOTAL)
    enola_logger.warning.assert_called_once()
    assert enola_logger.warning.call_args.args[1:] == (1, tmp_path / DOCS_ARTIFACT)


def test_malformed_lines_are_counted_in_one_warning_and_blank_lines_not_at_all(
    tmp_path, enola_logger
):
    malformed = [param.values[0] for param in MALFORMED_LINES]
    _snapshot_with_docs_bytes(tmp_path, _jsonl(PACKAGE, *malformed, "", "   ", TOTAL))

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (PACKAGE, TOTAL)
    enola_logger.warning.assert_called_once()
    assert enola_logger.warning.call_args.args[1:] == (len(malformed), tmp_path / DOCS_ARTIFACT)


@pytest.mark.parametrize(
    "kind", [pytest.param(["symbol"], id="list"), pytest.param({"symbol": True}, id="object")]
)
def test_a_kind_that_is_a_list_or_an_object_is_a_malformed_line(tmp_path, enola_logger, kind):
    """Any kind but "symbol" or "module" is skipped and counted, whatever its JSON type:
    an error out of the reader would end the code docs of every repository in the run."""
    _snapshot_with_docs_bytes(tmp_path, _jsonl(PACKAGE, _variant(kind=kind), TOTAL))

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (PACKAGE, TOTAL)
    enola_logger.warning.assert_called_once()
    assert enola_logger.warning.call_args.args[1] == 1


@pytest.mark.parametrize(
    "undecodable, error",
    [
        pytest.param(
            '{"line": ' + "1" * (INT_DIGIT_LIMIT + 1) + "}",
            ValueError,
            id="integer-past-the-digit-limit",
            marks=pytest.mark.skipif(
                not INT_DIGIT_LIMIT, reason="this Python converts integers of any length"
            ),
        ),
    ],
)
def test_a_line_json_cannot_decode_is_a_malformed_line_not_a_failure(
    tmp_path, enola_logger, undecodable, error
):
    """json.loads raises more than JSONDecodeError: a plain ValueError on an integer too
    long to convert. Out of the reader, it would end the code docs of every repository in
    the run."""
    # The line fails json.loads the way the case is named for.
    with pytest.raises(error) as decoding:
        json.loads(undecodable)
    assert type(decoding.value) is error
    _snapshot_with_docs_bytes(tmp_path, _jsonl(PACKAGE, undecodable, TOTAL))

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (PACKAGE, TOTAL)
    enola_logger.warning.assert_called_once()
    assert enola_logger.warning.call_args.args[1:] == (1, tmp_path / DOCS_ARTIFACT)


def test_a_line_nested_too_deep_is_a_malformed_line_not_a_failure(
    tmp_path, enola_logger, monkeypatch
):
    """json.loads raises RecursionError on nesting deeper than the stack holds; how deep
    that is depends on the machine (Python 3.14 measures the stack), so json.loads is
    made to raise it here."""
    loads = json.loads
    nested = "[[[]]]"

    def loads_past_the_stack(text, *args, **kwargs):
        if text == nested:
            raise RecursionError("maximum recursion depth exceeded")
        return loads(text, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads_past_the_stack)
    _snapshot_with_docs_bytes(tmp_path, _jsonl(PACKAGE, nested, TOTAL))

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (PACKAGE, TOTAL)
    enola_logger.warning.assert_called_once()
    assert enola_logger.warning.call_args.args[1:] == (1, tmp_path / DOCS_ARTIFACT)


@pytest.mark.parametrize(
    "line_break",
    [
        pytest.param("\u0085", id="next-line"),
        pytest.param("\u2028", id="line-separator"),
        pytest.param("\u2029", id="paragraph-separator"),
    ],
)
def test_a_doc_holding_a_unicode_line_break_stays_one_line(tmp_path, enola_logger, line_break):
    """Only a newline ends a docs.jsonl line. enola 0.4.32 writes U+0085 (NEXT LINE)
    inside a doc unescaped; U+2028 and U+2029 are as valid unescaped in a JSON string."""
    documented = {**TOTAL, "doc": f"Sum the prices.{line_break}Then take the discount off."}
    content = "".join(
        json.dumps(entry, ensure_ascii=False) + "\n" for entry in (PACKAGE, documented)
    ).encode("utf-8")
    _snapshot_with_docs_bytes(tmp_path, content)

    docs = read_enola_docs(tmp_path)

    assert docs.lines == (PACKAGE, documented)
    enola_logger.warning.assert_not_called()


# --- each declaration's own doc, from facts.jsonl --------------------------------


def test_fact_docs_are_those_of_the_requested_ids_by_id_and_line(tmp_path):
    """Overloads keep their own first paragraphs at their own lines, and a module, which
    has no line, is keyed by None. A fact that holds a requested id only as a relation
    target gets past the raw-text prefilter but is not one of that id's facts."""
    _write_facts(tmp_path, *FACTS)

    assert read_enola_fact_docs(tmp_path, {TOTAL["id"]}) == {
        (TOTAL["id"], 4): "Sum the prices and take the discount off.",
        (TOTAL["id"], 19): "Sum the prices of one cart.",
    }
    assert read_enola_fact_docs(str(tmp_path), {PACKAGE["id"], "0" * 32}) == {
        (PACKAGE["id"], None): "Price a shopping cart."
    }
    assert read_enola_fact_docs(tmp_path, {TOTAL["id"], PACKAGE["id"]}) == FACT_DOCS


def test_the_first_fact_of_an_id_and_line_gives_its_doc(tmp_path):
    """Should an (id, line) come twice, the first fact read keeps it; a null line is the
    same key as none."""
    _write_facts(
        tmp_path,
        *FACTS,
        {**TOTAL_FACT, "props": {"doc": "A second fact at line 4."}},
        {**PACKAGE_FACT, "line": None, "props": {"doc": "A second package fact."}},
    )

    assert read_enola_fact_docs(tmp_path, {TOTAL["id"], PACKAGE["id"]}) == FACT_DOCS


# Fact lines that hold TOTAL's quoted id but give no doc for it: not a JSON object,
# no doc prop of text, or no integer line. Those with a line to give carry a doc of
# their own, so one taken in TOTAL_FACT's place would show.
OTHER_PROPS = {"doc": "Not the doc of any declaration.", "language": "python"}
IGNORED_FACT_LINES = [
    pytest.param(
        '{"id": ' + json.dumps(TOTAL["id"]) + ', "line": 4, "props": {"doc": "Sum', id="not-json"
    ),
    pytest.param(
        _nested_past_the_recursion_limit(json.dumps(TOTAL["id"])),
        id="nested-past-the-recursion-limit",
    ),
    pytest.param(json.dumps([TOTAL_FACT]), id="array"),
    pytest.param(json.dumps(TOTAL["id"]), id="string"),
    # An id that is not a string, holding the requested one: no TypeError.
    pytest.param(json.dumps({**TOTAL_FACT, "id": [TOTAL["id"]]}), id="id-list"),
    pytest.param(json.dumps({**TOTAL_FACT, "id": {TOTAL["id"]: 1}}), id="id-object"),
    pytest.param(_fact_variant(props=_MISSING), id="props-missing"),
    pytest.param(_fact_variant(props="Sum the prices."), id="props-not-an-object"),
    pytest.param(_fact_variant(props={"language": "python"}), id="doc-missing"),
    pytest.param(_fact_variant(props={"doc": ""}), id="doc-empty"),
    pytest.param(_fact_variant(props={"doc": 7}), id="doc-number"),
    pytest.param(_fact_variant(props={"doc": ["Sum", "the", "prices."]}), id="doc-list"),
    pytest.param(_fact_variant(line="4", props=OTHER_PROPS), id="line-text"),
    pytest.param(_fact_variant(line=4.0, props=OTHER_PROPS), id="line-float"),
    pytest.param(_fact_variant(line=True, props=OTHER_PROPS), id="line-bool"),
]


@pytest.mark.parametrize("ignored", IGNORED_FACT_LINES)
def test_a_fact_line_without_a_doc_for_an_id_and_line_is_ignored(tmp_path, ignored):
    """The line holds the requested id, so it gets past the raw-text prefilter and the
    parsed line is what is turned down; the facts after it lose nothing, its (id, line)
    included."""
    assert json.dumps(TOTAL["id"]) in ignored
    _write_facts(tmp_path, PACKAGE_FACT, ignored, TOTAL_FACT, TOTAL_OF_ONE_CART_FACT)

    assert read_enola_fact_docs(tmp_path, {TOTAL["id"], PACKAGE["id"]}) == FACT_DOCS


def test_fact_docs_without_a_facts_jsonl_are_none(tmp_path):
    """Nothing to check docs.jsonl against, which is no error: the extraction keeps its text."""
    assert read_enola_fact_docs(tmp_path, {TOTAL["id"]}) == {}
    assert read_enola_fact_docs(tmp_path / "gone", {TOTAL["id"]}) == {}


def test_fact_docs_of_an_unreadable_facts_jsonl_are_none(tmp_path):
    """A facts.jsonl that cannot be opened, or is not UTF-8, is nothing to check against either."""
    facts_path = tmp_path / "facts.jsonl"
    facts_path.mkdir()

    assert read_enola_fact_docs(tmp_path, {TOTAL["id"]}) == {}

    facts_path.rmdir()
    facts_path.write_bytes(_jsonl(*FACTS) + b"\xff\n")

    assert read_enola_fact_docs(tmp_path, {TOTAL["id"]}) == {}


def test_no_fact_ids_open_no_facts_jsonl(tmp_path, monkeypatch):
    """Nothing asked, nothing read: facts.jsonl, which can be large, is not even opened."""
    _write_facts(tmp_path, *FACTS)
    opened = []

    def recording_open(file, *args, **kwargs):
        opened.append(file)
        return open(file, *args, **kwargs)

    monkeypatch.setattr(enola_module, "open", recording_open, raising=False)

    assert read_enola_fact_docs(tmp_path, set()) == {}
    assert opened == []
    # The recorder does see the read of a call that asks for an id.
    assert read_enola_fact_docs(tmp_path, {PACKAGE["id"]}) == {
        (PACKAGE["id"], None): "Price a shopping cart."
    }
    assert opened == [tmp_path / "facts.jsonl"]


# --- flows.json vouching, through the same rule ---------------------------------

FLOWS_BYTES = b'{"flows": []}'


@pytest.mark.parametrize(
    "receipt, vouches",
    [
        pytest.param(None, True, id="no-receipt"),
        pytest.param(
            {"output_hashes": {"flows.json": sha256_label(FLOWS_BYTES)}}, True, id="hash-matches"
        ),
        pytest.param(
            {"renderers": ["flows"], "output_hashes": {"flows.json": sha256_label(b"{}")}},
            False,
            id="hash-differs",
        ),
        pytest.param(
            {"renderers": ["flows"], "output_hashes": {DOCS_ARTIFACT: sha256_label(FLOWS_BYTES)}},
            False,
            id="hashes-without-flows-json",
        ),
        pytest.param({"renderers": ["llm_context", "flows"]}, True, id="flows-renderer-ran"),
        pytest.param(
            {"output_hashes": None, "renderers": ["flows"]}, True, id="null-output-hashes"
        ),
        pytest.param(
            {"renderers": ["llm_context", "docs"]}, False, id="flows-renderer-did-not-run"
        ),
        pytest.param({"renderers": "flows"}, False, id="renderers-not-a-list"),
        pytest.param({}, False, id="pre-flows-writer"),
    ],
)
def test_flows_json_vouching_is_unchanged_by_the_shared_rule(receipt, vouches):
    assert _receipt_vouches_for_flows(receipt, FLOWS_BYTES) is vouches


# --- the real enola binary --------------------------------------------------------


def _enola_available() -> bool:
    try:
        find_enola_binary()
        return True
    except EnolaNotInstalledError:
        return False


def _release(version) -> tuple[int, ...] | None:
    """(major, minor, patch) of an enola version string, None when it has none."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version) if isinstance(version, str) else None
    return tuple(int(part) for part in match.groups()) if match else None


@pytest.fixture
def enola_home(tmp_path, monkeypatch):
    """A home of the test's own: enola writes its state to ~/.enola on every run."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


async def _generate(repo_path):
    """enola's snapshot of ``repo_path`` and its receipt; a skip for an enola before docs.jsonl."""
    snapshot_dir = await run_enola_generate(repo_path)
    receipt = _receipt_of(snapshot_dir)
    release = _release(receipt.get("enola_version"))
    if release is not None and release < DOCS_RELEASE:
        pytest.skip(f"enola {receipt['enola_version']} predates docs.jsonl (0.4.32)")
    return snapshot_dir, receipt


@pytest.mark.skipif(not _enola_available(), reason="enola binary is not installed")
@pytest.mark.usefixtures("enola_home")
@pytest.mark.asyncio
async def test_real_enola_docs_carry_whole_comments_joined_to_their_facts(tmp_path):
    """enola on a tiny Python package: the function's line carries both paragraphs of
    its docstring, the module line the package docstring, each under its fact's id."""
    repo_path = tmp_path / REPO
    package = repo_path / "pricing"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '"""Price a shopping cart.\n\nEvery pricing rule lives in this package.\n"""\n',
        encoding="utf-8",
    )
    (package / "rules.py").write_text(
        "def total(prices, discount=0.0):\n"
        '    """Sum the prices and take the discount off.\n'
        "\n"
        "    A discount above one is a percentage, so 15 takes fifteen percent off.\n"
        '    """\n'
        "    if discount > 1:\n"
        "        discount = discount / 100\n"
        "    return sum(prices) * (1 - discount)\n",
        encoding="utf-8",
    )

    snapshot_dir, receipt = await _generate(repo_path)

    docs = read_enola_docs(snapshot_dir)

    by_name = {line["name"]: line for line in docs.lines}
    assert sorted(by_name) == ["pricing", "pricing/rules.total"]
    function, module = by_name["pricing/rules.total"], by_name["pricing"]
    assert (function["kind"], function["file"], function["line"]) == (
        "symbol",
        "pricing/rules.py",
        1,
    )
    assert function["doc"] == (
        "Sum the prices and take the discount off.\n\n"
        "A discount above one is a percentage, so 15 takes fifteen percent off."
    )
    assert (module["kind"], module["file"]) == ("module", "pricing")
    assert module["doc"] == "Price a shopping cart.\n\nEvery pricing rule lives in this package."
    assert docs.snapshot_id == snapshot_identity(snapshot_dir, receipt) == receipt["snapshot_id"]

    facts, _ = parse_enola_snapshot(snapshot_dir)
    nodes = {
        (type(point).__name__, point.name): point
        for point in map_facts_to_data_points(facts, repo_path=str(repo_path))
    }
    symbol = nodes[("CodeSymbol", "pricing/rules.total")]
    assert symbol.enola_id == function["id"]
    assert nodes[("CodeModule", "pricing")].enola_id == module["id"]
    assert function["repo"] == module["repo"] == symbol.repo


@pytest.mark.skipif(not _enola_available(), reason="enola binary is not installed")
@pytest.mark.usefixtures("enola_home")
@pytest.mark.asyncio
async def test_real_enola_facts_give_each_declaration_of_a_shared_id_its_own_first_paragraph(
    tmp_path,
):
    """A property's getter and setter share a fact id, so docs.jsonl has a line for each,
    which enola 0.4.32 can give one declaration's comment. facts.jsonl keeps each one's
    own first paragraph at its line, and read_enola_fact_docs returns it by (id, line)."""
    repo_path = tmp_path / REPO
    package = repo_path / "pricing"
    package.mkdir(parents=True)
    (package / "cart.py").write_text(
        "class Cart:\n"
        "    @property\n"
        "    def price(self):\n"
        '        """The cart\'s price, discount taken off.\n'
        "\n"
        "        Read it after every change.\n"
        '        """\n'
        "        return self._price\n"
        "\n"
        "    @price.setter\n"
        "    def price(self, value):\n"
        '        """Set the price before the discount.\n'
        "\n"
        "        Negative prices are refused.\n"
        '        """\n'
        "        self._price = value\n",
        encoding="utf-8",
    )

    snapshot_dir, _ = await _generate(repo_path)

    accessors = [
        line
        for line in read_enola_docs(snapshot_dir).lines
        if line["name"] == "pricing/cart.Cart.price"
    ]
    assert len(accessors) == 2
    getter, setter = sorted(accessors, key=lambda line: line["line"])
    assert getter["id"] == setter["id"]
    assert read_enola_fact_docs(snapshot_dir, {getter["id"]}) == {
        (getter["id"], getter["line"]): "The cart's price, discount taken off.",
        (setter["id"], setter["line"]): "Set the price before the discount.",
    }
