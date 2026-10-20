"""read_enola_docs: the reader of enola's docs.jsonl (enola >= 0.4.32).

docs.jsonl holds the whole comment of every documented symbol and module, one
JSON line per fact, joined to facts.jsonl by the fact ``id``. The reader takes
only a file the snapshot's receipt.json vouches for -- the rule flows.json
follows, shared through _receipt_vouches_for -- and names the snapshot the docs
belong to, so the code docs tasks can tell whether they describe the code graph
they hold. Covered: lines in file order and the snapshot id; ``parse`` False,
the availability check of the code docs sweep; every reason the docs are
unavailable; a missing or unparseable receipt; which lines are skipped as
malformed and how they are counted; flows.json vouching, unchanged; and one run
of the real enola binary on a tiny Python package.
"""

import importlib
import json
import re
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
FACTS = [
    {"kind": "module", "name": "pricing", "file": "pricing", "repo": REPO, "id": PACKAGE["id"]},
    {
        "kind": "symbol",
        "name": "pricing/rules.total",
        "file": "pricing/rules.py",
        "line": 4,
        "repo": REPO,
        "id": TOTAL["id"],
    },
]

# docs.jsonl bytes with a Latin-1 byte in a doc.
NOT_UTF8 = (json.dumps(TOTAL) + "\n").encode("utf-8").replace(b"Sum", b"S\xfcm")

BOTH_MODES = pytest.mark.parametrize("parse", [True, False], ids=["parsed", "check-only"])

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
    """docs.jsonl bytes: a dict entry is dumped as one JSON line, a str entry is written as is."""
    lines = (entry if isinstance(entry, str) else json.dumps(entry) for entry in entries)
    return "".join(line + "\n" for line in lines).encode("utf-8")


def _variant(**changes):
    """TOTAL as one JSON line, with fields changed (or removed, when set to _MISSING)."""
    entry = {**TOTAL, **changes}
    return json.dumps({key: value for key, value in entry.items() if value is not _MISSING})


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


def test_check_only_read_returns_no_lines_and_the_same_snapshot_id(tmp_path, enola_logger):
    """``parse`` False is the sweep's availability check: the lines are not even read."""
    _snapshot_with_docs_bytes(tmp_path, _jsonl(TOTAL, "{ not json"))

    checked = read_enola_docs(tmp_path, parse=False)

    assert checked == EnolaDocs(snapshot_id=SNAPSHOT_ID, lines=())
    enola_logger.warning.assert_not_called()
    assert read_enola_docs(tmp_path) == EnolaDocs(snapshot_id=SNAPSHOT_ID, lines=(TOTAL,))


@pytest.mark.parametrize(
    "content", [pytest.param(b"", id="empty"), pytest.param(b"\n   \n", id="blank-lines")]
)
def test_an_empty_vouched_docs_file_documents_nothing(tmp_path, enola_logger, content):
    """Nothing documented, which is not "docs not written": no lines, and no refusal."""
    _snapshot_with_docs_bytes(tmp_path, content)

    assert read_enola_docs(tmp_path) == EnolaDocs(snapshot_id=SNAPSHOT_ID, lines=())
    enola_logger.warning.assert_not_called()


# --- unavailable docs -----------------------------------------------------------


@BOTH_MODES
def test_a_snapshot_without_docs_jsonl_is_unavailable(tmp_path, parse):
    """A snapshot from enola before 0.4.32, or from a configuration without the docs renderer."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=None)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path, parse=parse)
    assert f"no {DOCS_ARTIFACT} in '{tmp_path}'" in str(raised.value)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path / "gone", parse=parse)
    assert f"no {DOCS_ARTIFACT} in '{tmp_path / 'gone'}'" in str(raised.value)


@BOTH_MODES
def test_a_docs_jsonl_that_cannot_be_read_is_unavailable(tmp_path, parse):
    write_enola_snapshot(tmp_path, facts=FACTS, docs=None)
    (tmp_path / DOCS_ARTIFACT).mkdir()

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path, parse=parse)

    assert f"could not read '{tmp_path / DOCS_ARTIFACT}'" in str(raised.value)


@BOTH_MODES
def test_docs_bytes_the_receipt_hashes_differently_are_unavailable(tmp_path, parse):
    """The file changed after the run the receipt describes."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL])
    with open(tmp_path / DOCS_ARTIFACT, "a", encoding="utf-8") as docs_file:
        docs_file.write(json.dumps(PACKAGE) + "\n")

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path, parse=parse)

    assert f"receipt.json in '{tmp_path}' does not list these {DOCS_ARTIFACT} bytes" in str(
        raised.value
    )


@BOTH_MODES
def test_output_hashes_without_docs_jsonl_refuse_it_whatever_else_they_list(tmp_path, parse):
    """An earlier run's file. A receipt with output hashes is not overruled by its
    renderers, and another artifact's entry vouches for nothing, even with these bytes."""
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL], vouch_docs=False)
    receipt = _receipt_of(tmp_path)
    receipt["renderers"] = ["llm_context", "flows", "docs"]
    receipt["output_hashes"]["flows.json"] = sha256_label((tmp_path / DOCS_ARTIFACT).read_bytes())
    _write_receipt(tmp_path, receipt)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path, parse=parse)

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


@BOTH_MODES
@pytest.mark.parametrize(
    "receipt_fields",
    [
        pytest.param({"renderers": ["llm_context", "flows"]}, id="docs-renderer-did-not-run"),
        pytest.param({"renderers": "docs"}, id="renderers-not-a-list"),
        pytest.param({}, id="neither-hashes-nor-renderers"),
    ],
)
def test_a_receipt_without_output_hashes_or_the_docs_renderer_refuses_the_file(
    tmp_path, receipt_fields, parse
):
    write_enola_snapshot(tmp_path, facts=FACTS, docs=[TOTAL])
    _write_receipt(tmp_path, {"snapshot_id": SNAPSHOT_ID, "format_version": 1, **receipt_fields})

    with pytest.raises(EnolaDocsUnavailable):
        read_enola_docs(tmp_path, parse=parse)


def test_docs_bytes_that_are_not_utf8_are_unavailable(tmp_path):
    _snapshot_with_docs_bytes(tmp_path, NOT_UTF8)

    with pytest.raises(EnolaDocsUnavailable) as raised:
        read_enola_docs(tmp_path)

    assert f"'{tmp_path / DOCS_ARTIFACT}' is not UTF-8" in str(raised.value)


def test_check_only_read_refuses_what_the_parsed_read_refuses(tmp_path):
    """The sweep (``parse`` False) must keep every repository the extraction leaves out:
    a repository it finds available while the extraction emitted nothing for it loses
    its stored docs."""
    _snapshot_with_docs_bytes(tmp_path, NOT_UTF8)

    with pytest.raises(EnolaDocsUnavailable):
        read_enola_docs(tmp_path, parse=False)


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
    "line_break",
    [
        pytest.param("\u0085", id="next-line"),
        pytest.param(" ", id="line-separator"),
        pytest.param(" ", id="paragraph-separator"),
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


@pytest.mark.skipif(not _enola_available(), reason="enola binary is not installed")
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

    snapshot_dir = await run_enola_generate(repo_path)
    receipt = _receipt_of(snapshot_dir)
    release = _release(receipt.get("enola_version"))
    if release is not None and release < DOCS_RELEASE:
        pytest.skip(f"enola {receipt['enola_version']} predates docs.jsonl (0.4.32)")

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
