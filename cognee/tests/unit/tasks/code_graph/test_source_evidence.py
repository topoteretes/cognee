import os
import time

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.tasks.code_graph.extract_code_graph import map_facts_to_data_points
from cognee.tasks.code_graph.models import CodeSource, CodeSymbol
from cognee.tasks.code_graph.source_evidence import (
    capture_sources,
    source_for_fact,
    source_node_id,
)

CLASS_FILE = """import os

class Store:
    def save(self):
        return 1

    def load(self):
        return 2


def helper():
    return 3
"""


def _symbol(name, line, file="store.py"):
    return {"kind": "symbol", "name": name, "file": file, "line": line}


def _points(facts, tmp_path):
    sources = capture_sources(facts, tmp_path, time.time_ns())
    points = map_facts_to_data_points(facts, source_files=sources)
    symbols = [p for p in points if isinstance(p, CodeSymbol)]
    return symbols, [p for p in points if isinstance(p, CodeSource)]


def test_source_is_its_own_node_and_changes_the_symbol_hash_with_the_text(tmp_path):
    path = tmp_path / "sync.py"
    path.write_text("def sync():\n    return True\n")
    fact = {"kind": "symbol", "name": "sync", "file": "sync.py", "line": 1, "end_line": 2}
    (first,), (first_source,) = _points([fact], tmp_path)
    assert first_source.text == path.read_text()
    assert first_source.id == source_node_id(first.id) and first_source.symbol_id == str(first.id)
    assert first.source_sha256 == first_source.text_sha256
    assert not hasattr(first, "source_evidence")
    path.write_text("def sync():\n    return False\n")
    (second,), (second_source,) = _points([fact], tmp_path)
    assert first.fact_hash != second.fact_hash
    assert first_source.fact_hash != second_source.fact_hash
    assert first_source.id == second_source.id


def test_the_fact_hash_does_not_depend_on_the_snapshot_or_commit(tmp_path):
    (tmp_path / "sync.py").write_text("def sync():\n    return True\n")
    fact = {"kind": "symbol", "name": "sync", "file": "sync.py", "line": 1}
    (first,), (first_source,) = _points([fact], tmp_path)
    (second,), (second_source,) = _points([fact], tmp_path)
    assert first.fact_hash == second.fact_hash
    assert first_source.fact_hash == second_source.fact_hash
    assert set(CodeSource.model_fields) >= {"text", "line", "end_line", "text_sha256"}


def test_editing_one_symbol_does_not_change_the_others_hash(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("def one():\n    return 1\n\n\ndef two():\n    return 2\n")
    facts = [_symbol("m.one", 1, "m.py"), _symbol("m.two", 5, "m.py")]
    before = {s.name: s.fact_hash for s in _points(facts, tmp_path)[0]}
    path.write_text("def one():\n    return 1\n\n\ndef two():\n    return 3\n")
    after = {s.name: s.fact_hash for s in _points(facts, tmp_path)[0]}
    assert before["m.one"] == after["m.one"] and before["m.two"] != after["m.two"]


def test_imported_snapshot_has_no_source_nodes(tmp_path):
    (tmp_path / "sync.py").write_text("unrelated later content")
    fact = {"kind": "symbol", "name": "sync", "file": "sync.py"}
    points = map_facts_to_data_points([fact], repo_path=tmp_path)
    assert not any(isinstance(p, CodeSource) for p in points)
    assert points[-1].source_sha256 is None


def test_changed_source_during_extraction_is_rejected(tmp_path):
    started = time.time_ns()
    (tmp_path / "sync.py").write_text("changed during indexing")
    with pytest.raises(CogneeValidationError, match="changed during extraction") as error:
        capture_sources([{"file": "sync.py"}], tmp_path, started)
    assert error.value.remediation


def test_a_permission_change_alone_is_not_a_change_of_the_source(tmp_path):
    path = tmp_path / "sync.py"
    path.write_text("def sync():\n    return True\n")
    started = time.time_ns()
    os.chmod(path, 0o600)
    assert "sync.py" in capture_sources([{"file": "sync.py"}], tmp_path, started)


def test_a_symlink_out_of_the_repository_is_not_read_and_does_not_stop_the_others(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "private.py"
    outside.write_text("private")
    (root / "escape.py").symlink_to(outside)
    (root / "own.py").write_text("def own():\n    return 1\n")
    sources = capture_sources([{"file": "escape.py"}, {"file": "own.py"}], root, time.time_ns())
    assert set(sources) == {"own.py"}


def test_a_file_dated_in_the_future_is_not_a_change_during_extraction(tmp_path):
    path = tmp_path / "sync.py"
    path.write_text("def sync():\n    return True\n")
    started = time.time_ns()
    future = time.time() + 3600
    os.utime(path, (future, future))
    assert "sync.py" in capture_sources([{"file": "sync.py"}], tmp_path, started)


def test_lines_are_counted_by_newline_only_like_the_extractor(tmp_path):
    (tmp_path / "a.py").write_text(
        'S = "x\u2028y"\n\nclass K:\n    pass\n\x0c\ndef g():\n    return 1\n'
    )
    facts = [_symbol("K", 3, "a.py"), _symbol("g", 6, "a.py")]
    sources = capture_sources(facts, tmp_path, time.time_ns())
    assert source_for_fact(facts[0], sources)["text"].startswith("class K:")
    assert source_for_fact(facts[1], sources)["text"] == "def g():\n    return 1\n"


def test_a_symbol_without_any_range_keeps_the_whole_file(tmp_path):
    (tmp_path / "sync.py").write_text("header\ndef sync():\n    return True\n")
    fact = {"file": "sync.py", "line": 2}
    source = source_for_fact(fact, capture_sources([fact], tmp_path, time.time_ns()))
    assert source["line"] == 1
    assert source["end_line"] == 3
    assert source["text"].startswith("header")


def test_a_symbol_runs_to_the_next_symbol_that_is_not_its_member(tmp_path):
    (tmp_path / "store.py").write_text(CLASS_FILE)
    facts = [
        _symbol("store.Store", 3),
        _symbol("store.Store.save", 4),
        _symbol("store.Store.load", 7),
        _symbol("store.helper", 11),
    ]
    sources = capture_sources(facts, tmp_path, time.time_ns())
    texts = {fact["name"]: source_for_fact(fact, sources)["text"] for fact in facts}
    # A class keeps its methods; a method stops where the next one starts; the last symbol
    # runs to the end of the file. None of them carries a sibling's body.
    assert "def load" in texts["store.Store"] and "def helper" not in texts["store.Store"]
    assert "return 1" in texts["store.Store.save"] and "def load" not in texts["store.Store.save"]
    assert "return 2" in texts["store.Store.load"] and "helper" not in texts["store.Store.load"]
    assert (
        texts["store.helper"].startswith("def helper")
        and "class Store" not in texts["store.helper"]
    )
    assert "import os" not in "".join(texts.values())


def test_the_extractors_own_end_line_wins_over_the_derived_one(tmp_path):
    (tmp_path / "store.py").write_text(CLASS_FILE)
    fact = {**_symbol("store.Store", 3), "end_line": 5}
    source = source_for_fact(fact, capture_sources([fact], tmp_path, time.time_ns()))
    assert (source["line"], source["end_line"]) == (3, 5)


DECORATED = """import os

TOP = 5


class A:
    # the name
    @property
    def b(self):
        return 1

    def c(self):
        return 2


# module level helper
@decorator
def d():
    return 3


if __name__ == "__main__":
    d()
"""


def test_decorators_and_comments_belong_to_their_symbol_and_trailing_code_to_none(tmp_path):
    (tmp_path / "m.py").write_text(DECORATED)
    facts = [
        _symbol("m.A", 6, "m.py"),
        _symbol("m.A.b", 9, "m.py"),
        _symbol("m.A.c", 12, "m.py"),
        _symbol("m.d", 18, "m.py"),
    ]
    sources = capture_sources(facts, tmp_path, time.time_ns())
    texts = {fact["name"]: source_for_fact(fact, sources)["text"] for fact in facts}
    assert texts["m.A.b"].startswith("    # the name\n    @property\n    def b")
    assert "def c" not in texts["m.A.b"] and "# module level" not in texts["m.A.c"]
    assert texts["m.A"].endswith("return 2\n") and "TOP" not in texts["m.A"]
    assert texts["m.d"] == "# module level helper\n@decorator\ndef d():\n    return 3\n"
    assert "__main__" not in "".join(texts.values())


def test_a_symbol_does_not_take_the_closing_brace_of_its_class(tmp_path):
    (tmp_path / "a.ts").write_text(
        "class Svc {\n  add(): number {\n    return 1;\n  }\n  del(): number {\n"
        "    return 2;\n  }\n}\n"
    )
    facts = [
        _symbol("Svc", 1, "a.ts"),
        _symbol("Svc.add", 2, "a.ts"),
        _symbol("Svc.del", 5, "a.ts"),
    ]
    sources = capture_sources(facts, tmp_path, time.time_ns())
    texts = {fact["name"]: source_for_fact(fact, sources)["text"] for fact in facts}
    assert texts["Svc.del"] == "  del(): number {\n    return 2;\n  }\n"
    assert texts["Svc"].endswith("}\n") and "class Svc" in texts["Svc"]


def test_sections_and_scopes_are_members_of_their_parent(tmp_path):
    (tmp_path / "README.md").write_text("# Top\n\nintro\n\n## Sub\n\ntext\n")
    (tmp_path / "c.cpp").write_text("class A {\npublic:\n  void m();\n};\nint z;\n")
    facts = [
        _symbol("README.md", 1, "README.md"),
        _symbol("README.md#sub", 5, "README.md"),
        _symbol("ns::A", 1, "c.cpp"),
        _symbol("ns::A::m", 3, "c.cpp"),
    ]
    sources = capture_sources(facts, tmp_path, time.time_ns())
    texts = {fact["name"]: source_for_fact(fact, sources)["text"] for fact in facts}
    assert "## Sub" in texts["README.md"] and texts["README.md#sub"].startswith("## Sub")
    assert "void m()" in texts["ns::A"] and "int z" not in texts["ns::A"]


def _texts(tmp_path, filename, content, symbols):
    (tmp_path / filename).write_text(content)
    facts = [_symbol(name, line, filename) for name, line in symbols]
    sources = capture_sources(facts, tmp_path, time.time_ns())
    return {fact["name"]: source_for_fact(fact, sources)["text"] for fact in facts}


def test_annotated_java_members_keep_their_declaration_and_body(tmp_path):
    texts = _texts(
        tmp_path,
        "A.java",
        "@Entity\npublic class A {\n\n    @Override\n    public int hashCode() {\n"
        "        return 1;\n    }\n\n    @Test\n    public void run() {\n    }\n}\n",
        [("A", 1), ("A.hashCode", 4), ("A.run", 9)],
    )
    assert texts["A"].startswith("@Entity\npublic class A {") and texts["A"].endswith("}\n")
    assert texts["A.hashCode"] == (
        "    @Override\n    public int hashCode() {\n        return 1;\n    }\n"
    )
    assert texts["A.run"] == "    @Test\n    public void run() {\n    }\n"


def test_kotlin_and_csharp_annotations_and_attributes_do_not_cut_the_declaration(tmp_path):
    kotlin = _texts(
        tmp_path,
        "K.kt",
        "object K {\n    @JvmStatic\n    fun f(): Int {\n        return 1\n    }\n}\n",
        [("K", 1), ("K.f", 2)],
    )
    assert kotlin["K.f"] == "    @JvmStatic\n    fun f(): Int {\n        return 1\n    }\n"
    csharp = _texts(
        tmp_path,
        "C.cs",
        "class C\n{\n    [Obsolete]\n    public void M()\n    {\n        Run();\n    }\n}\n",
        [("C", 1), ("C.M", 3)],
    )
    assert csharp["C.M"] == "    [Obsolete]\n    public void M()\n    {\n        Run();\n    }\n"
    assert csharp["C"].startswith("class C\n{") and csharp["C"].endswith("}\n")


def test_a_decorator_over_several_lines_belongs_to_its_symbol(tmp_path):
    python = _texts(
        tmp_path,
        "c.py",
        "import functools\n\n\ndef first():\n    return 1\n\n\n"
        "@functools.lru_cache(\n    maxsize=None,\n)\ndef cached():\n    return 2\n",
        [("c.first", 4), ("c.cached", 11)],
    )
    assert python["c.cached"].startswith("@functools.lru_cache(\n    maxsize=None,\n)\ndef cached")
    assert "lru_cache" not in python["c.first"]
    typescript = _texts(
        tmp_path,
        "c.ts",
        "export class Other {}\n\n@Component({\n  selector: 'x',\n})\nexport class Cmp {\n}\n",
        [("Other", 1), ("Cmp", 6)],
    )
    assert typescript["Cmp"].startswith("@Component({\n  selector: 'x',\n})\nexport class Cmp")
    assert "Component" not in typescript["Other"]


def test_code_before_a_decorator_is_not_mistaken_for_a_multi_line_decorator(tmp_path):
    python = _texts(
        tmp_path,
        "d.py",
        "def one():\n    return 1\n\n\nX = compute(\n    1,\n)\n\n\ndef two():\n    return 2\n",
        [("d.one", 1), ("d.two", 10)],
    )
    assert python["d.two"] == "def two():\n    return 2\n"
    assert "compute" not in python["d.one"]


def test_a_namespace_closing_brace_is_not_taken_by_the_last_class(tmp_path):
    texts = _texts(
        tmp_path,
        "n.cpp",
        "namespace ns {\nclass A {\npublic:\n  void m();\n};\n}\n",
        [("ns::A", 2), ("ns::A::m", 4)],
    )
    assert texts["ns::A"] == "class A {\npublic:\n  void m();\n};\n"


def test_the_next_line_brace_style_is_part_of_the_declaration(tmp_path):
    texts = _texts(
        tmp_path,
        "e.cs",
        "class E\n{\n    void Run()\n    {\n        Go();\n    }\n}\n\nclass F\n{\n}\n",
        [("E", 1), ("E.Run", 3), ("F", 9)],
    )
    assert texts["E.Run"] == "    void Run()\n    {\n        Go();\n    }\n"
    assert texts["F"] == "class F\n{\n}\n"


def test_a_source_node_follows_a_change_of_its_remote_repository(tmp_path):
    (tmp_path / "sync.py").write_text("def sync():\n    return True\n")
    fact = {"kind": "symbol", "name": "sync", "file": "sync.py", "line": 1}
    sources = capture_sources([fact], tmp_path, time.time_ns())
    plain = [
        p
        for p in map_facts_to_data_points([fact], source_files=sources)
        if isinstance(p, CodeSource)
    ]
    remote = [
        p
        for p in map_facts_to_data_points([fact], source_files=sources, repo_name="github.com/o/r")
        if isinstance(p, CodeSource)
    ]
    assert plain[0].fact_hash != remote[0].fact_hash
    assert remote[0].remote_repository == "github.com/o/r"


def test_a_try_or_else_block_after_a_nested_function_is_not_an_access_label(tmp_path):
    texts = _texts(
        tmp_path,
        "t.py",
        "def outer():\n    async def inner():\n        return 1\n\n    try:\n        run()\n"
        "    finally:\n        stop()\n",
        [("t.outer.inner", 2)],
    )
    assert texts["t.outer.inner"] == "    async def inner():\n        return 1\n"


def test_an_annotation_that_spans_lines_does_not_stand_in_for_the_declaration(tmp_path):
    java = _texts(
        tmp_path,
        "S.java",
        'class S {\n    @RequestMapping(\n        value = "/y",\n        method = {GET, POST}\n    )\n'
        '    public String several() {\n        return "x";\n    }\n}\n',
        [("S", 1), ("S.several", 2)],
    )
    assert java["S.several"] == (
        '    @RequestMapping(\n        value = "/y",\n        method = {GET, POST}\n    )\n'
        '    public String several() {\n        return "x";\n    }\n'
    )
    csharp = _texts(
        tmp_path,
        "T.cs",
        "class T {\n    [Theory]\n    [InlineData(1,\n        2)]\n    public void Multi(int a, int b) {\n"
        "        Run();\n    }\n}\n",
        [("T", 1), ("T.Multi", 2)],
    )
    assert csharp["T.Multi"].startswith("    [Theory]\n") and csharp["T.Multi"].endswith(
        "        Run();\n    }\n"
    )


def test_an_annotation_on_the_line_of_the_declaration_keeps_the_whole_body(tmp_path):
    java = _texts(
        tmp_path,
        "I.java",
        '@Entity public class I {\n    @Override public String toString() {\n        return "i";\n    }\n'
        "    @Test void inline() {\n        run();\n    }\n}\n",
        [("I", 1), ("I.toString", 2), ("I.inline", 5)],
    )
    assert java["I.toString"] == (
        '    @Override public String toString() {\n        return "i";\n    }\n'
    )
    assert java["I.inline"] == "    @Test void inline() {\n        run();\n    }\n"
    assert java["I"].endswith("    }\n}\n")
    swift = _texts(
        tmp_path,
        "s.swift",
        "class C {\n    @objc func f() {\n        go()\n    }\n}\n",
        [("C", 1), ("C.f", 2)],
    )
    assert swift["C.f"] == "    @objc func f() {\n        go()\n    }\n"


def test_a_php_attribute_belongs_to_the_function_it_precedes(tmp_path):
    php = _texts(
        tmp_path,
        "p.php",
        '<?php\nclass P {\n    #[Route("/x")]\n    public function f() {\n        return 1;\n    }\n}\n',
        [("P", 2), ("P.f", 3)],
    )
    assert php["P.f"] == '    #[Route("/x")]\n    public function f() {\n        return 1;\n    }\n'


def test_brackets_inside_strings_do_not_break_a_decorators_balance(tmp_path):
    python = _texts(
        tmp_path,
        "q.py",
        "import functools\n\n\n@functools.lru_cache(\n    maxsize=None,\n"
        '    typed=[")", "]"][0] == ")",\n)\ndef cached():\n    return 1\n',
        [("q.cached", 8)],
    )
    assert python["q.cached"].startswith("@functools.lru_cache(")


def test_a_list_above_a_symbol_is_not_taken_for_an_attribute(tmp_path):
    python = _texts(
        tmp_path,
        "l.py",
        "def one():\n    return foo(\n        1,\n    [1, 2])\n\n\nMATRIX = [\n    [1, 2],\n    [3, 4]]\n"
        "\n\ndef two():\n    return 2\n",
        [("l.one", 1), ("l.two", 12)],
    )
    assert python["l.two"] == "def two():\n    return 2\n"
    assert python["l.one"].endswith("[1, 2])\n")
    typescript = _texts(
        tmp_path,
        "l.ts",
        "function a() {\n  return 1;\n}\n[1, 2].forEach(go);\nfunction b() {\n  return 2;\n}\n",
        [("a", 1), ("b", 5)],
    )
    assert typescript["b"] == "function b() {\n  return 2;\n}\n"


def test_stacked_attributes_on_one_line_belong_to_their_symbol(tmp_path):
    csharp = _texts(
        tmp_path,
        "u.cs",
        "class U {\n    void Before() {\n    }\n\n    [A][B]\n    [C]\n    public void M() {\n    }\n}\n",
        [("U", 1), ("U.Before", 2), ("U.M", 5)],
    )
    assert csharp["U.M"].startswith("    [A][B]\n    [C]\n    public void M()")
    assert "[A]" not in csharp["U.Before"]


def test_a_source_node_follows_a_change_of_remote_repository_alone(tmp_path):
    (tmp_path / "sync.py").write_text("def sync():\n    return True\n")
    fact = {"kind": "symbol", "name": "sync", "file": "sync.py", "line": 1, "repo": "same"}
    sources = capture_sources([fact], tmp_path, time.time_ns())

    def source_node(repo_name):
        points = map_facts_to_data_points([fact], source_files=sources, repo_name=repo_name)
        return next(p for p in points if isinstance(p, CodeSource))

    plain, remote = source_node(None), source_node("github.com/o/r")
    assert plain.repo == remote.repo == "same"
    assert plain.fact_hash != remote.fact_hash
